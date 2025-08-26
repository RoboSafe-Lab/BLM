import torch.nn as nn
import torch
from .diffuser_helpers import extract
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily



def to_device_tensor_batch(obs: dict, device: torch.device) -> dict:
    out = {}
    for k, v in obs.items():
        if isinstance(v, torch.Tensor):
            t = v
        else:
            # numpy 或 list
            t = torch.as_tensor(v)  # 更鲁棒：既能接 numpy 也能接 tensor/list
        if k in ["current_step", "next_step"]:
            t = t.to(device=device, dtype=torch.long)
        else:
            t = t.to(device=device, dtype=torch.float32)
        out[k] = t
    return out




class DiffusionBackbone(nn.Module):
    def __init__(self, latent_diffusion,vae):
        super().__init__()
        self.latent_diffusion = latent_diffusion
        self.vae = vae

    def head(self, x_t, cond_feat, t_float, map_grid_feat):
        return self.latent_diffusion.model(x_t, cond_feat, t_float, map_grid_feat)

    def features(self, x_t, cond_feat, t_float, map_grid_feat):
        return self.latent_diffusion.model.get_backbone(x_t, cond_feat, t_float, map_grid_feat)


class DiffusionActor(nn.Module):
    def __init__(self, backbone: DiffusionBackbone, n_diffusion_steps):
        super().__init__()
        self.backbone = backbone
        # 超参
        self.n_timesteps = n_diffusion_steps
        self.eta = 0.3
        # 缓存给 dist_fn 用
        self._mu_t = None
        self._sigma_t = None

    @torch.no_grad()
    def forward(self, obs, state=None, info=None):

        device = next(self.parameters()).device
        obs = to_device_tensor_batch(obs, device=device)

        z_t   = obs["z_t"]              # [B, T, 2]
        cond_feat  = obs["cond_feat"]        # [B, Cctx]
        map_grid_feat  = obs["map_grid_feat"]    # [B, C, H, W]
        t     = obs["current_step"]     # [B]
        t_next= obs["next_step"]        # [B]
        t_float = t.float() / (self.n_timesteps - 1)




        # 1) 条件、无条件两路
        logits_pi, mu, log_sigma = self.backbone.head(z_t, cond_feat, t_float, map_grid_feat)
        
        sigma = torch.exp(log_sigma)

        latent_diffusion = self.backbone.latent_diffusion
        
        alpha_t    = extract(latent_diffusion.alphas_cumprod, t,      z_t.shape)
        alpha_next = extract(latent_diffusion.alphas_cumprod, t_next, z_t.shape)

        sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(-1)
        sqrt_one_minus_alpha_t = torch.sqrt((1 - alpha_t).clamp_min(1e-8)).unsqueeze(-1)
        
        # 计算eps_pred
        eps_pred = (z_t.unsqueeze(2) - sqrt_alpha_t * mu) / sqrt_one_minus_alpha_t
        
        # DDIM方差
        sigma_t = self.eta * torch.sqrt(
            ((1 - alpha_t / alpha_next) * (1 - alpha_next) / (1 - alpha_t)).clamp_min(1e-8)
        ).unsqueeze(-1)
        
        # 均值计算
        self._mu_t = (
            torch.sqrt(alpha_next).unsqueeze(-1) * mu
            + torch.sqrt((1 - alpha_next).unsqueeze(-1) - sigma_t**2) * eps_pred
        )

        var_combined = alpha_next.unsqueeze(-1) * (sigma**2) + sigma_t**2 
        self._sigma_t = torch.sqrt(torch.clamp(var_combined, min=1e-6))

        return logits_pi, None  # Tianshou 期望返回 (logits, state)
  

    def dist_fn(self, raw_logits):
        pi = F.softmax(raw_logits, dim=-1)  # [B, T, K]
        comp = Independent(Normal(self._mu_t, self._sigma_t), 1)  # 事件维=2
        return MixtureSameFamily(Categorical(pi), comp)


# ---------- Critic ----------
class DiffusionCritic(nn.Module):

    def __init__(self, backbone: DiffusionBackbone, n_diffusion_steps, actor_hidden):
        super().__init__()
        self.backbone = backbone
        self.n_timesteps = n_diffusion_steps
        self.value_head = nn.Sequential(
            nn.Linear(actor_hidden, actor_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(actor_hidden, 1)
        )
    

    def forward(self, obs, state=None, info=None):
        device = next(self.parameters()).device
        obs = to_device_tensor_batch(obs, device=device)

        z_t   = obs["z_t"]              # [B, T, 2]
        cond_feat  = obs["cond_feat"]        # [B, Cctx]
        map_grid_feat  = obs["map_grid_feat"]    # [B, C, H, W]
        t     = obs["current_step"]  
        t_float = t.float() / (self.n_timesteps - 1)

     
     

        feat = self.backbone.features(z_t, cond_feat, t_float, map_grid_feat)  # [B, T, H]
        feat_global = feat.mean(dim=1)
        v = self.value_head(feat_global).squeeze(-1)    # [B]
        return v
