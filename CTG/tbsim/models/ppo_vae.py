import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .ppo_diffusion_model import ConvCrossAttnDiffuser
import tbsim.models.base_models as base_models
from torch.distributions import Categorical, Normal,Independent,MixtureSameFamily
# from tbsim.models.diffuser_helpers import (
#     AgentHistoryEncoder,
#     NeighborHistoryEncoder,
#     MapEncoder,
# )
from tbsim.models.context_encoder import (
    MapEncoder,
    AgentHistoryEncoder,
    MLP,
    NeighborHistoryEncoder

)


import numpy as np
from tbsim.utils.geometry_utils import transform_points_tensor
from .diffuser_helpers import (
    cosine_beta_schedule,
    query_feature_grid,
    unicyle_forward_dynamics,
    extract
)
from tbsim.dynamics import Unicycle
import tbsim.utils.tensor_utils as TensorUtils


class TemporalEncoder(nn.Module):

    def __init__(self,
                 input_dim: int = 2,
                 enc_channels: int = 64,
                 latent_dim: int = 8,
                 downsample_stride: int = 4,
                 lowpass_kernel: int = 7):
        super().__init__()
        # kernel >= stride : 先低通 (平滑) 再降采样
        self.downsample = nn.Sequential(
            nn.Conv1d(input_dim, enc_channels,
                      kernel_size=lowpass_kernel,
                      stride=downsample_stride,
                      padding=lowpass_kernel // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(enc_channels, enc_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj_mu     = nn.Conv1d(enc_channels, latent_dim, kernel_size=1)
        self.proj_logvar = nn.Conv1d(enc_channels, latent_dim, kernel_size=1)

    def forward(self, x_bt2: torch.Tensor):
        # [B,T,C] -> [B,C,T]
        feat_bct = self.downsample(x_bt2.permute(0, 2, 1))     # [B, enc_channels, L]
        mu_bld   = self.proj_mu(feat_bct).permute(0, 2, 1)     # [B, L, latent_dim]
        lv_bld   = self.proj_logvar(feat_bct).permute(0, 2, 1) # [B, L, latent_dim]
        return mu_bld, lv_bld



class TemporalDecoderWithContext(nn.Module):

    def __init__(
        self,
        latent_dim: int = 8,       # 与 encoder 的 z 维度一致
        traj_feat_dim: int = 32,   # grid_map_traj_T 的通道数
        cond_dim: int = 128,       # cond_feat 的维度（你的 context_encoder_out_dim）
        hidden: int = 128,
        upsample_stride: int = 4,  # L -> T = L * stride
        out_dim: int = 2           # 动作维度（如 [a_lon, yaw_rate]）
    ):
        super().__init__()
        self.upsample_stride = upsample_stride

        # 1) 线性把 z、traj_feat 投到同一隐藏维
        self.z_proj    = nn.Linear(latent_dim, hidden)
        self.traj_proj = nn.Linear(traj_feat_dim, hidden)

        # 2) cond_feat 做 FiLM（轻量调制）
        self.cond_to_film = nn.Sequential(
            nn.Linear(cond_dim, hidden * 2)
        )

        # 3) 时间细化（小型 1D 卷积堆）
        self.refine = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 4) 输出头
        self.head = nn.Conv1d(hidden, out_dim, kernel_size=1)

    def forward(
        self,
        z_bld: torch.Tensor,            # [B, L, latent_dim]
        cond_feat: torch.Tensor,        # [B, cond_dim]
        grid_map_traj_T: torch.Tensor,  # [B, T, traj_feat_dim]
    ) -> torch.Tensor:                  # -> [B, T, out_dim]
        B, L, d = z_bld.shape
        T = grid_map_traj_T.size(1)
        s = self.upsample_stride
        assert T == L * s, f"T should equal L*stride, got T={T}, L={L}, stride={s}"

        # 上采样到 T（复制低频骨架）
        z_up = z_bld.repeat_interleave(s, dim=1)         # [B, T, d]

        # 投射到隐藏维并融合逐帧地图特征
        h = self.z_proj(z_up) 
        if grid_map_traj_T is not None:
            h = h + self.traj_proj(grid_map_traj_T)

        # FiLM 调制：cond_feat -> (γ, β)
        if cond_feat is not None:
            gamma, beta = self.cond_to_film(cond_feat).chunk(2, dim=-1)  # [B,H],[B,H]
            h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)

        # 时间细化
        h = self.refine(h.permute(0, 2, 1))       # [B,H,T]
        out = self.head(h).permute(0, 2, 1)       # [B,T,2]
        return out

class PPO_VAE(nn.Module):
    def __init__(
        self, 
        map_encoder_model_arch,
        input_image_shape,
        global_feature_dim,
        grid_feature_dim,

        history_frames,
        center_history_out_dim,
        norm_info_center,

        state_encoder_out_dim,

        neighbor_history_out_dim,
        norm_info_neighbor,

        context_encoder_hidden_dim,
        context_encoder_out_dim,

        vae_input_dim,
        vae_enc_channels,
        vae_latent_dim,
        vae_sample_stride,
        vae_lowpass_kernel,
        vae_dec_channels,
        vae_output_dim,
      
        dynamics_kwargs,
    

    ):
        super().__init__()

        cond_in_feat_size = 0

        self.map_encoder = MapEncoder(model_arch=map_encoder_model_arch,
                                    input_image_shape=input_image_shape,
                                    global_feature_dim=global_feature_dim,
                                    grid_feature_dim=grid_feature_dim)
        cond_in_feat_size += global_feature_dim
        
        self.center_hist = AgentHistoryEncoder( num_steps=history_frames,
                                                out_dim=center_history_out_dim,
                                                norm_info=norm_info_center)

        cond_in_feat_size += center_history_out_dim

        self.neighbor_hist = NeighborHistoryEncoder(num_steps=history_frames,
                                                    out_dim=neighbor_history_out_dim,
                                                    norm_info=norm_info_neighbor)
        cond_in_feat_size += neighbor_history_out_dim

        combine_layer_dims = (cond_in_feat_size, cond_in_feat_size, context_encoder_hidden_dim, context_encoder_hidden_dim)
        self.process_cond_mlp = MLP(in_dim = cond_in_feat_size,
                                    out_dim = context_encoder_out_dim,
                                    hidden_dims = combine_layer_dims)

        self.vae_encoder = TemporalEncoder(
            input_dim=vae_input_dim,
            enc_channels=vae_enc_channels,
            latent_dim=vae_latent_dim,
            downsample_stride=vae_sample_stride,
            lowpass_kernel=vae_lowpass_kernel
        )
        self.vae_decoder = TemporalDecoderWithContext(
            latent_dim=vae_latent_dim,
            traj_feat_dim=grid_feature_dim,
            cond_dim=context_encoder_out_dim,
            hidden=context_encoder_hidden_dim,
            upsample_stride=vae_sample_stride,
            out_dim=vae_output_dim
        )

        self._dynamics_kwargs = dynamics_kwargs
        self._create_dynamics(dynamics_kwargs)        

                              
     
        norm_add_coeffs = norm_info_center[0]
        norm_div_coeffs = norm_info_center[1]
        self.add_coeffs = np.array(norm_add_coeffs).astype('float32')
        self.div_coeffs = np.array(norm_div_coeffs).astype('float32')   
        print('self.add_coeffs', self.add_coeffs)
        print('self.div_coeffs', self.div_coeffs)    

     

        self.default_chosen_inds = [4, 5]
        self.input_image_shape = input_image_shape


    def _create_dynamics(self,config):
            self.dyn = Unicycle(
                        "dynamics",
                        max_steer=config["max_steer"],
                        max_yawvel=config["max_yawvel"],
                        acce_bound=config["acce_bound"],
                        vbound=config["vbound"])
        

    def scale_traj(self, target_traj_orig, chosen_inds=[]):

        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D
        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)
        target_traj = (target_traj_orig - dx_add) / dx_div

        return target_traj

    def descale_traj(self, target_traj_orig, chosen_inds=[]):

        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D

        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device) 

        target_traj = target_traj_orig * dx_div + dx_add

        return target_traj


    def query_map_feats(self, x, map_grid_feat, raster_from_agent):
        '''
        - x : (B, T, D)
        - map_grid_feat : (B, C, H, W)
        - raster_from_agent: (B, 3, 3)
        '''
        B, T, _ = x.size()
        _, _, Hfeat, Wfeat = map_grid_feat.size()

        raster_xy = transform_points_tensor(x, raster_from_agent)

        _, H_in, W_in = self.input_image_shape

        raster_xy[:,:,0] *= (Wfeat / W_in)
        raster_xy[:,:,1] *= (Hfeat / H_in)

        feats_out = query_feature_grid(raster_xy,map_grid_feat)
        feats_out = feats_out.reshape((B, T, -1))
        return feats_out

    def context_encoder(self,batch):
        global_feat, grid_feat = self.map_encoder(batch["maps"])
        center_hist = self.center_hist(batch["center_hist_positions"],
                                        batch["center_hist_speeds"],
                                        batch["center_hist_yaws"],
                                        batch["center_hist_acc_lons"],
                                        batch["center_hist_yaw_rates"],
                                        batch["extent"],
                                        batch["center_hist_availabilities"])

        
        neigh_hist = self.neighbor_hist(batch["neigh_hist_positions"],
                                        batch["neigh_hist_speeds"],
                                        batch["neigh_hist_yaws"],
                                        batch["neigh_hist_acc_lons"],
                                        batch["neigh_hist_yaw_rates"],
                                        batch["neigh_extent"][...,:2],
                                        batch["neigh_hist_availabilities"])
        cond_feat = self.process_cond_mlp(torch.cat([global_feat, center_hist, neigh_hist], -1))
        return cond_feat,grid_feat

    def compute_loss(self, batch):

        cond_feat, map_grid_feat = self.context_encoder(batch)
        
        center_fut_action = torch.cat([batch['center_fut_acc_lons'].unsqueeze(-1),
                                batch['center_fut_yaw_rates'].unsqueeze(-1)],dim=-1)
        
        x = self.scale_traj(center_fut_action)

        mu, logvar = self.vae_encoder(x) 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps   

        grid_map_traj_T = self.query_map_feats(
            batch['center_fut_positions'].detach(),   # 训练时用 GT 位置
            map_grid_feat,
            batch['raster_from_center'])

        x_hat = self.vae_decoder(
            z_bld=z,
            cond_feat=cond_feat,
            grid_map_traj_T=grid_map_traj_T
        )
        mask = batch['center_fut_availabilities'].unsqueeze(-1)
        mse_elem = F.mse_loss(x_hat, x, reduction='none')  
        mse_masked = mse_elem * mask 
        denom = (mask.sum() * x.size(-1)).clamp_min(1.0)                       # 防 0
        recon = mse_masked.sum() / denom

        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
    

        return recon, kl
    
        

    
