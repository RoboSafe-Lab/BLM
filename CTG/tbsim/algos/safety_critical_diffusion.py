import pytorch_lightning as pl

from tbsim.models.ppo_diffuser import PPO_Diffuser
from tbsim.utils.safety_critical_batch_utils import parse_batch
import torch.optim as optim
import torch.nn as nn
from tbsim.policies.common import Plan, Action
from tbsim.utils.trajdata_utils import convert_scene_data_to_agent_coordinates,  add_scene_dim_to_agent_data, get_stationary_mask

from tbsim.utils.safety_critical_fig_vis import plot_trajdata_batch

class PPO_Diffusion_Trainer(pl.LightningModule):
    def __init__(self, algo_config, modality_shapes, registered_name):
        super().__init__()
        self.algo_config = algo_config
        self.nets = nn.ModuleDict() 
        self.disable_control_on_stationary = algo_config.disable_control_on_stationary
        self.moving_speed_th = algo_config.moving_speed_th
    

        self.nets['policy'] = PPO_Diffuser(
            map_encoder_model_arch=algo_config.map_encoder_model_arch,
            input_image_shape=modality_shapes["image"],
            map_feature_dim=algo_config.map_feature_dim,
            map_grid_feature_dim=algo_config.map_grid_feature_dim,
            hist_num_frames=algo_config.history_num_frames+1,
            agent_hist_feat_dim=algo_config.history_feature_dim,

            
            state_in_dim=algo_config.state_in_dim,
            state_out_feat_dim=algo_config.state_out_feat_dim,
            neighbor_hist_feat_dim=algo_config.history_feature_dim,
    
            cond_feature_dim=algo_config.cond_feat_dim,
            time_emb_dim=algo_config.time_emb_dim,
            attn_hidden_dim=algo_config.attn_hidden_dim,
            attn_dilations=algo_config.attn_dilations,
            attn_n_heads=algo_config.attn_n_heads,
            grid_map_traj_dim=algo_config.grid_map_traj_dim,
            mix_gauss=algo_config.mix_gauss,

            diffuser_norm_info=algo_config.norm_info['diffuser'],
            agent_hist_norm_info=algo_config.norm_info['agent_hist'],
            neighbor_hist_norm_info=algo_config.norm_info['neighbor_hist'],
  

            dynamics_type=algo_config.dynamics.type,
            dynamics_kwargs=algo_config.dynamics,

            n_timesteps=algo_config.n_diffusion_steps,
            ddim_steps=algo_config.ddim_steps,
        
            horizon=algo_config.horizon,
        )

        self.cur_train_step = 0
    
   

    def on_train_batch_start(self, batch, batch_idx):
        return parse_batch(batch)

    def training_step_end(self, batch_parts):
        self.cur_train_step += 1

    def training_step(self, batch, batch_idx):
        
        B = batch.center_fut_positions.shape[0]
        loss= self.nets['policy'].compute_loss(batch)
        self.log('train_loss', loss, prog_bar=True, batch_size=B,on_step=True)
        return loss
    
    def on_validation_batch_start(self, batch, batch_idx):
        return parse_batch(batch)
    
    def validation_step(self, batch, batch_idx):
        B = batch.center_fut_positions.shape[0]
        loss= self.nets['policy'].compute_loss(batch)
        self.log(f'val_loss', loss, prog_bar=True, batch_size=B,on_epoch=True)
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

    def forward(self,obs,global_t):
        if global_t ==0:
            self.stationary_mask = get_stationary_mask(obs, self.disable_control_on_stationary, self.moving_speed_th)
            B = self.stationary_mask.shape[0]
            stationary_mask_expand =  self.stationary_mask
        return self.nets['policy'](obs, stationary_mask=stationary_mask_expand, global_t=global_t)

    def get_action(self, obs_torch, **kwargs):
        # plot_trajdata_batch(obs_torch,None)
        preds = self(obs_torch, global_t=kwargs['step_index'])["predictions"]
        preds_positions = preds["positions"]
        # 
        preds_yaws = preds["yaws"]

        info = dict(
            action_samples=Action(
                positions=preds_positions,
                yaws=preds_yaws
            ).to_dict(),
        )
        action = Action(
            positions=preds_positions,
            yaws=preds_yaws
        )
        return action, info