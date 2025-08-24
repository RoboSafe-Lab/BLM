import pytorch_lightning as pl

from tbsim.models.ppo_diffuser import PPO_Diffuser
from tbsim.utils.safety_critical_batch_utils import parse_batch
import torch.optim as optim
import torch.nn as nn
from tbsim.policies.common import Plan, Action
from tbsim.utils.trajdata_utils import convert_scene_data_to_agent_coordinates,  add_scene_dim_to_agent_data, get_stationary_mask

from tbsim.models.ppo_latent_diffusion import PPO_LatentDiffusion
from tbsim.algos.vae_diffusion import TrajectoryVAE
class LatentDiffusion(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes):
        super().__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict()
        vae = TrajectoryVAE.load_from_checkpoint(
            algo_config.vae_ckpt_path,
            algo_config=algo_config,
            modality_shapes=modality_shapes,
            registered_name=None
        ).nets['policy']

        self._externals = {"vae": vae}
        self.nets['policy'] = PPO_LatentDiffusion(
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape = modality_shapes['image'],
            global_feature_dim = algo_config.global_feature_dim,
            grid_feature_dim = algo_config.grid_feature_dim,

            history_frames = int(algo_config.history_num_frames+1),
            center_history_out_dim = algo_config.center_history_out_dim,
            norm_info_center = algo_config.norm_info_center,

            state_encoder_out_dim = algo_config.state_encoder_out_dim,

            neighbor_history_out_dim = algo_config.neighbor_history_out_dim,
            norm_info_neighbor = algo_config.norm_info_neighbor,

            context_encoder_hidden_dim = algo_config.context_encoder_hidden_dim,
            context_encoder_out_dim = algo_config.context_encoder_out_dim,

            time_emb_dim = algo_config.time_emb_dim,
            diffusion_hidden_dim = algo_config.diffusion_hidden_dim,
            dilations = algo_config.dilations,
            num_heads = algo_config.num_heads,
       

            dynamics_kwargs = algo_config.Dynamics,
            n_timesteps = algo_config.n_diffusion_steps,

            dt = algo_config.dt,
            ddim_steps = algo_config.ddim_steps,
            latent_diffusion_dim = algo_config.vae['vae_latent_dim']
        )


        self.cur_train_step = 0

    def _freeze_vae(self):
        vae = self._externals["vae"]
        vae.to(self.device)     # 关键：迁移到当前 LightningModule 的 device
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

    def on_fit_start(self):
        self._freeze_vae()
    def on_validation_start(self):
        self._freeze_vae()
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.cur_train_step += 1


    def training_step(self, batch, batch_idx):
        batch = parse_batch(batch)
        B = batch['center_fut_positions'].shape[0]
        loss = self.nets['policy'].compute_loss(batch,self._externals["vae"])
        self.log('train_loss', loss, prog_bar=True, batch_size=B,on_step=True)

        return loss

    
    def validation_step(self, batch, batch_idx):
        batch = parse_batch(batch)
        B = batch['center_fut_positions'].shape[0]
        loss = self.nets['policy'].compute_loss(batch,self._externals["vae"])
        self.log('val_loss', loss, prog_bar=True, batch_size=B,on_epoch=True)
        return loss


    def configure_optimizers(self):
        optim_params = self.algo_config.optim_params
        return optim.Adam(
            params=self.nets["policy"].parameters(),
            lr=optim_params["learning_rate"]["initial"],
            weight_decay=optim_params["learning_rate"]['weight_decay']
        )


    def on_train_epoch_end(self):
        """记录当前学习率"""
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)
