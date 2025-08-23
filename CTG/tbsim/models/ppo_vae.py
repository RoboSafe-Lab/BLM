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



class TemporalDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int = 8,
                 hidden_dim: int = 64,
                 output_dim: int = 2,
                 upsample_stride: int = 4,
                 # context 维度
                 cond_dim: int = 0,              # 全局 cond_feat
                 grid_traj_dim: int = 0,         # 逐帧局部地图特征维度 (T, Cg)
                 grid_map_dim: int = 32,         # 全图特征每像素的通道（与 MapEncoder 对齐）
                 n_heads: int = 4):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.upsample_stride = upsample_stride
        self.use_film = cond_dim > 0
        self.use_traj_ctx = grid_traj_dim > 0

        # 1) latent -> T 的“粗特征”
        self.up = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, hidden_dim,
                               kernel_size=8, stride=upsample_stride, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 2) 全局 context → FiLM(γ, β) 调制通道
        if self.use_film:
            self.cond_to_film = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
            )

        # 3) 逐帧局部地图特征 (T, Cg) → 投影到 hidden_dim，逐时刻相加
        if self.use_traj_ctx:
            self.traj_proj = nn.Linear(grid_traj_dim, hidden_dim)

        # 4) 全图 Cross-Attn（整图 token + 2D 位置）
        self.grid_map_proj = nn.Linear(grid_map_dim, hidden_dim)
        self.grid_map_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, batch_first=True
        )
        self.grid_map_gate = nn.Parameter(torch.zeros(1))
        self.grid_map_norm = nn.LayerNorm(hidden_dim)

        # 5) 最终读出
        self.to_out = nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1)

        self.phase_emb = nn.Embedding(upsample_stride, hidden_dim)

        self.register_buffer("grid_pos_cache", None, persistent=False)

    # ---- 生成 2D 位置编码 ----
    def _get_pos_emb(self, B, Hm, Wm, device):
        if (self.grid_pos_cache is None) or (self.grid_pos_cache.shape[1] != Hm * Wm):
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, Hm, device=device),
                torch.linspace(-1, 1, Wm, device=device),
                indexing='ij'
            )
            pos = torch.stack([yy, xx], dim=-1).view(1, Hm * Wm, 2)  # (1, N, 2)
            self.grid_pos_cache = pos
        return self.grid_pos_cache.expand(B, -1, -1)                 # (B, N, 2)

    def forward(self,
                z_bld: torch.Tensor,                    # [B, L, d]
                cond_feat: torch.Tensor = None,         # [B, cond_dim]
                grid_map_feat: torch.Tensor = None,     # [B, Cg, H, W]
                grid_map_traj_T: torch.Tensor = None):  # [B, T, Cg]
        B, L, d = z_bld.shape

        # 1) latent 上采样到 T：h [B, T, H]
        h = self.up(z_bld.permute(0, 2, 1)).permute(0, 2, 1)   # [B,H,T] -> [B,T,H]

        
        T = h.size(1)
        phase = torch.arange(T, device=h.device) % self.upsample_stride  # [T]
        h = h + self.phase_emb(phase).unsqueeze(0)                        # [B,T,H]

        # 2) 逐帧局部地图：投影后逐时刻相加

        traj_ctx = self.traj_proj(grid_map_traj_T)        # [B,T,H]
        h = h + traj_ctx

        # 3) 全局条件 FiLM：通道调制
  
        film = self.cond_to_film(cond_feat)               # [B,2H]
        gamma, beta = film.chunk(2, dim=-1)               # [B,H], [B,H]
        h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)    # [B,T,H]

        # 4) 全图 Cross-Attn：query=h(T步)，key/value=整图token（带2D位置）
      
        Bm, Cg, Hm, Wm = grid_map_feat.shape
        assert Bm == B, "grid_map_feat batch mismatch"
        # flatten map -> (B, N, Cg) -> proj (B,N,H)
        m_flat = grid_map_feat.view(B, Cg, -1).permute(0, 2, 1)          # (B, N, Cg)
        m_proj = self.grid_map_proj(m_flat)                               # (B, N, H)
        # 2D 位置编码
        pos = self._get_pos_emb(B, Hm, Wm, grid_map_feat.device)          # (B, N, 2)
        pos_emb = nn.functional.relu(nn.Linear(2, self.hidden_dim, bias=False).to(h.device)(pos))
        m_proj = m_proj + pos_emb                                         # (B, N, H)

        attn, _ = self.grid_map_attn(query=h, key=m_proj, value=m_proj)   # (B, T, H)
        h = h + torch.sigmoid(self.grid_map_gate) * attn
        h = self.grid_map_norm(h)

        # 5) 输出到动作维
        x_hat = self.to_out(h.permute(0, 2, 1)).permute(0, 2, 1)              # [B,T,output_dim]
        return x_hat


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
        vae_downsample_stride,
        vae_lowpass_kernel,
        vae_dec_channels,
        vae_output_dim,
      
        dynamics_kwargs,
        beta

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
            downsample_stride=vae_downsample_stride,
            lowpass_kernel=vae_lowpass_kernel
        )
        self.vae_decoder = TemporalDecoder(
            latent_dim=vae_latent_dim,
            hidden_dim=vae_dec_channels,
            output_dim=vae_output_dim,
            upsample_stride=vae_downsample_stride,
            cond_dim=context_encoder_out_dim,
            grid_traj_dim=grid_feature_dim,
            grid_map_dim=grid_feature_dim,
        )

        self._dynamics_kwargs = dynamics_kwargs
        self._create_dynamics(dynamics_kwargs)        

                              
        self.beta = beta
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
        _, C, Hfeat, Wfeat = map_grid_feat.size()

        # unscale to agent coords
        pos_traj = self.descale_traj(x.detach())[:,:,:2]
        # convert to raster frame
        raster_pos_traj = transform_points_tensor(pos_traj, raster_from_agent)

        # scale to the feature map size
        _, H, W = self.input_image_shape
        xscale = Wfeat / W
        yscale = Hfeat / H
        raster_pos_traj[:,:,0] = raster_pos_traj[:,:,0] * xscale
        raster_pos_traj[:,:,1] = raster_pos_traj[:,:,1] * yscale

        # interpolate into feature grid
        feats_out = query_feature_grid(
                            raster_pos_traj,
                            map_grid_feat
                            )
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
            grid_map_feat=map_grid_feat,
            grid_map_traj_T=grid_map_traj_T
        )

        recon = F.mse_loss(x_hat, x)

        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        loss = recon + self.beta * kl

        return loss
    
        

    
