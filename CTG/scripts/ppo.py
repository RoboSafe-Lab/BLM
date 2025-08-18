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
def ppo_training(eval_cfg):

        
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

    datamodule = datamodule_factory(
        cls_name=exp_config.train.datamodule_class, config=exp_config
    )

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
                                PPOEnv(exp_config.algo, datamodule, policy_model), 
                                buffer= ReplayBuffer(size=transitions_per_episode),
                                preprocess_fn=preprocess_fn)
    test_collector = Collector(policy,
                                PPOEnv(exp_config.algo, datamodule, policy_model),
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
        save_best_fn=None,
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
