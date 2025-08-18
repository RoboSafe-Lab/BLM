"""A script for evaluating closed-loop simulation"""
import argparse
import numpy as np
import json
import random
import importlib
import os
import torch
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env
from tbsim.utils.safety_critical_batch_utils import set_global_trajdata_batch_raster_cfg
from tbsim.configs.scene_edit_config import SceneEditingConfig
from tianshou.policy import PPOPolicy
from tianshou.data import Collector,ReplayBuffer
import wandb
from tianshou.trainer import onpolicy_trainer 
from tbsim.models.ppo_actor_critic import DiffusionBackbone,DiffusionActor,DiffusionCritic
from tbsim.datasets.factory import datamodule_factory
from tbsim.models.ppo_env import PPOEnv,preprocess_fn
from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily
import os, time, json, torch, pytorch_lightning as pl
def gmm_entropy_jensen_lower(mix: MixtureSameFamily):

    cat  = mix.mixture_distribution               # Categorical
    comp = mix.component_distribution             # Independent(Normal)
    base = comp.base_dist                         # Normal

    pi  = cat.probs                               # [..., K]
    mu  = base.loc                                # [..., K, D]
    var = (base.scale ** 2).clamp_min(1e-12)      # [..., K, D]

    # 形状一致性检查
    K = pi.shape[-1]
    assert mu.shape[-2] == K and var.shape[-2] == K, "K 维不一致"
    D = mu.shape[-1]
    assert var.shape[-1] == D, "D 维不一致"

    # 只围绕 K 维做两两组合：得到 [..., K_i, K_j, D]
    var_i   = var.unsqueeze(-2)                   # [..., K, 1, D]
    var_j   = var.unsqueeze(-3)                   # [..., 1, K, D]
    var_sum = (var_i + var_j).clamp_min(1e-12)    # [..., K, K, D]

    diff    = mu.unsqueeze(-2) - mu.unsqueeze(-3) # [..., K, K, D]
    maha    = (diff.pow(2) / var_sum).sum(-1)     # [..., K, K]
    logdet  = torch.log(2 * torch.pi * var_sum).sum(-1)  # [..., K, K]
    log_norm = -0.5 * (maha + logdet)             # [..., K, K]

    # 对 j 维做加权求和：log Σ_j π_j N(·)
    log_pi    = torch.log(pi.clamp_min(1e-12))    # [..., K]
    log_mix   = log_pi.unsqueeze(-2) + log_norm   # [..., K, K]  (π_j 广播到 j 维)
    log_sumexp = torch.logsumexp(log_mix, dim=-1) # [..., K]     (sum over j)

    # H ≥ - Σ_i π_i * log(...)
    H_lb = -(pi * log_sumexp).sum(dim=-1)         # [...]
    return H_lb

# 打补丁（在创建 PPOPolicy 之前执行）
MixtureSameFamily.entropy = gmm_entropy_jensen_lower


def build_save_best_fn(save_dir, actor, critic, wandb_run=None):

    os.makedirs(save_dir, exist_ok=True)
    policy_best = os.path.join(save_dir, "policy_best.ckpt")
    critic_best = os.path.join(save_dir, "critic_best.pth")
    def _lightning_policy_ckpt(policy_model: torch.nn.Module):

        sd = policy_model.state_dict()
        sd_prefixed = {f"nets.policy.{k}": v.detach().cpu() for k, v in sd.items()}
        
        # 创建完整的 Lightning checkpoint 格式
        ckpt = {
            "state_dict": sd_prefixed,
            "pytorch-lightning_version": pl.__version__,
            "epoch": 0,
            "global_step": 0,
            "lr_schedulers": [],
            "optimizer_states": [],
            "datamodule_hyper_parameters": {},
            "hyper_parameters": {},
        }
        return ckpt

    def save_best_fn(ts_policy):  # Tianshou 会把 policy 传进来，但我们直接用外层闭包里的 actor/critic
        # 1) 保存策略为 Lightning 兼容的 .ckpt
        policy_model = actor.backbone.ppo  # 就是 PPO_Diffuser
        ckpt = _lightning_policy_ckpt(policy_model)
        torch.save(ckpt, policy_best)
        print(f"[PPO] saved policy ckpt -> {policy_best}")

        # 2) critic 单存 .pth（训练时会再 load）
        torch.save({"critic": critic.state_dict()}, critic_best)
        print(f"[PPO] saved critic pth  -> {critic_best}")

        # 3) 可选：打到 wandb artifact
        if wandb_run is not None:
            import wandb
            art = wandb.Artifact("ppo_rl", type="model")
            art.add_file(policy_best)
            art.add_file(critic_best)
            wandb_run.log_artifact(art)

    return save_best_fn





def ppo_training(eval_cfg):

    # "trajdata_source_train": ["nusc_trainval-train","nusc_trainval-train_val"],
    # "trajdata_source_valid": ["nusc_trainval-val"],
    # "trajdata_source_train": ["nusc_mini-mini_train"],
    # "trajdata_source_valid": ["nusc_mini-mini_val"],
    set_global_batch_type("trajdata")
    set_global_trajdata_batch_env(eval_cfg.trajdata_source_test[0])
    np.random.seed(eval_cfg.seed)
    random.seed(eval_cfg.seed)
    torch.manual_seed(eval_cfg.seed)
    torch.cuda.manual_seed(eval_cfg.seed)
    # basic setup
    print('saving results to {}'.format(eval_cfg.results_dir))
    os.makedirs(eval_cfg.results_dir, exist_ok=True)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # create policy and rollout wrapper
    policy_composers = importlib.import_module("tbsim.evaluation.policy_composers")
    composer_class = getattr(policy_composers, eval_cfg.eval_class)
    # composer = composer_class(eval_cfg, device)
    # policy, exp_config = composer.get_policy()

    composer = composer_class(eval_cfg, device)
    policy, exp_config = composer.get_policy()

    exp_config.train.training.batch_size = 1
    exp_config.train.validation.batch_size = 1

    datamodule = datamodule_factory(
        cls_name=exp_config.train.datamodule_class, config=exp_config
    )
    datamodule.setup("fit")
    # determines cfg for rasterizing agents
    set_global_trajdata_batch_raster_cfg(exp_config.env.rasterizer)
    
    policy_model = policy.model.nets['policy']

    backbone = DiffusionBackbone(policy_model)
    actor  = DiffusionActor(backbone,n_diffusion_steps=int(exp_config.algo.n_diffusion_steps)).to(device)
    critic = DiffusionCritic(backbone,n_diffusion_steps=int(exp_config.algo.n_diffusion_steps),
                                        actor_hidden=exp_config.algo.ppo_actor_hidden).to(device)



    policy = PPOPolicy(
        actor =  actor,
        critic = critic,
        optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=eval_cfg.ppo["learning_rate"]),
        dist_fn = actor.dist_fn,
        discount_factor=eval_cfg.ppo["gamma"],
        gae_lambda=eval_cfg.ppo["gae_lambda"],
        max_grad_norm=eval_cfg.ppo["max_grad_norm"],
        eps_clip=eval_cfg.ppo["clip_ratio"],
        vf_coef=eval_cfg.ppo["vf_coef"],
        ent_coef=eval_cfg.ppo["ent_coef"],
    )

    transitions_per_episode  = eval_cfg.ppo["episodes_per_collect"]  *  exp_config.algo.ddim_steps #40*50=2000

    train_collector = Collector(policy,
                                PPOEnv(exp_config.algo, datamodule, datamodule.train_dataloader(), policy_model), 
                                buffer= ReplayBuffer(size=transitions_per_episode),
                                preprocess_fn=preprocess_fn)
    test_collector = Collector(policy,
                                PPOEnv(exp_config.algo, datamodule, datamodule.val_dataloader(), policy_model),
                                preprocess_fn=preprocess_fn)

    if not eval_cfg.debug:
        wandb.login()
        logger = WandbLogger(
            train_interval=1,
            update_interval=1,
            test_interval=1,
            project=eval_cfg.ppo['wandb_project'],
            entity=eval_cfg.ppo['wandb_entity'],
            name=eval_cfg.ppo['wandb_run_name'],
            save_interval=1,
            run_id=eval_cfg.ppo['wandb_id'],
            )

        writer = SummaryWriter(exp_config.training.output_dir)
        writer.add_text("config", str(exp_config))
        logger.load(writer)
    else:
        logger = None

    save_dir = os.path.join(eval_cfg.results_dir, "ppo_rl")
    wandb_run = logger.wandb_run if logger is not None else None
    save_best_fn = build_save_best_fn(save_dir, actor, critic, wandb_run)
    #--------------------- 3 * 1 * 50 --------------------------------------------
    result = onpolicy_trainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=eval_cfg.ppo["ppo_epochs"],#100
        repeat_per_collect=eval_cfg.ppo["update_per_collect"],#10 每次从buffer中取出minibatch,更新次数
        step_per_collect=transitions_per_episode,# buffer中存满这些数据后，开始更新, 
        step_per_epoch=transitions_per_episode * 10,# 一个epoch有多少条episode
        episode_per_test= eval_cfg.ppo["test_episodes"],
        batch_size=eval_cfg.ppo["mini_batch_size"], #64
        save_best_fn=save_best_fn,
        logger=logger,
        show_progress=True,

    )

    print(f"Finished training! Best reward: {result['best_reward']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="A json file containing evaluation configs"
    )


    parser.add_argument(
        "--eval_class",
        type=str,
        default=None,
        help="Optionally specify the evaluation class through argparse"
    )

    parser.add_argument(
        "--policy_ckpt_dir",
        type=str,
        default=None,
        help="Directory to look for saved checkpoints"
    )

    parser.add_argument(
        "--policy_ckpt_key",
        type=str,
        default=None,
        help="A string that uniquely identifies a checkpoint file within a directory, e.g., iter50000"
    )


    parser.add_argument(
        "--results_root_dir",
        type=str,
        default=None,
        help="Directory to save results and videos"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Root directory of the dataset"
    )



    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )


    parser.add_argument(
        "--registered_name",
        type=str,
        default='trajdata_nusc_diff',
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )




    args = parser.parse_args()

    cfg = SceneEditingConfig(registered_name=args.registered_name)

    if args.config_file is not None:
        external_cfg = json.load(open(args.config_file, "r"))
        cfg.update(**external_cfg)

    if args.eval_class is not None:
        cfg.eval_class = args.eval_class

    if args.policy_ckpt_dir is not None:
        assert args.policy_ckpt_key is not None, "Please specify a key to look for the checkpoint, e.g., 'iter50000'"
        cfg.ckpt.policy.ckpt_dir = args.policy_ckpt_dir
        cfg.ckpt.policy.ckpt_key = args.policy_ckpt_key


    if args.dataset_path is not None:
        cfg.dataset_path = args.dataset_path

    if cfg.name is None:
        cfg.name = cfg.eval_class


    if args.seed is not None:
        cfg.seed = args.seed
    if args.results_root_dir is not None:
        cfg.results_dir = os.path.join(args.results_root_dir, cfg.name)
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, cfg.name)

    cfg.debug = args.debug



    for k in cfg[cfg.env]:  # copy env-specific config to the global-level
        cfg[k] = cfg[cfg.env][k]

    cfg.pop("nusc")
    cfg.pop("trajdata")

    cfg.lock()
    ppo_training(cfg)
