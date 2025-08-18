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
    def __init__(self, ppo_diffuser):
        super().__init__()
        self.ppo = ppo_diffuser

    def head(self, x_t, cond_feat, t_float, map_grid_feat, map_grid_traj):
        return self.ppo.model(x_t, cond_feat, t_float, map_grid_feat, map_grid_traj)

    def features(self, x_t, cond_feat, t_float, map_grid_feat, map_grid_traj):
        return self.ppo.model.get_backbone(x_t, cond_feat, t_float, map_grid_feat, map_grid_traj)


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

        x_t   = obs["x_t"]              # [B, T, 2]
        cond  = obs["cond_feat"]        # [B, Cctx]
        grid  = obs["map_grid_feat"]    # [B, C, H, W]
        traj  = obs["map_grid_traj"]    # [B, T, Cmap]
        t     = obs["current_step"]     # [B]
        t_next= obs["next_step"]        # [B]
        t_float = t.float() / (self.n_timesteps - 1)

        # 1) 条件、无条件两路
        raw_logits, mu, log_sigma = self.backbone.head(x_t, cond, t_float, grid, traj)
        log_sigma = torch.clamp(log_sigma, min=-20, max=2)
        sigma = torch.exp(log_sigma)

        ppo = self.backbone.ppo

        alpha_t   = extract(ppo.alphas_cumprod, t,      x_t.shape)
        alpha_nxt = extract(ppo.alphas_cumprod, t_next, x_t.shape)
        sqrt_at   = torch.sqrt(alpha_t)
        sqrt_one_minus_at = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-8))

        eps_pred = (x_t.unsqueeze(2) - sqrt_at.unsqueeze(-2) * mu) / (sqrt_one_minus_at.unsqueeze(-2))

        frac = (1.0 - alpha_t / torch.clamp(alpha_nxt, min=1e-8)) \
             * (1.0 - alpha_nxt) / torch.clamp(1.0 - alpha_t, min=1e-8)
        sigma_t = self.eta * torch.sqrt(torch.clamp(frac, min=1e-8)).unsqueeze(-1)  # [B, T, 1, 2]

        # 均值 mu_t: [B, T, K, 2]
        mu_t = torch.sqrt(torch.clamp(alpha_nxt, min=1e-8)).unsqueeze(-1) * mu + \
               torch.sqrt(torch.clamp(1.0 - alpha_nxt, min=1e-8) - sigma_t**2).unsqueeze(-1) * eps_pred

        # 合成方差（随机 DDIM）：[B, T, K, 2]
        var_combined = alpha_nxt.unsqueeze(-1).unsqueeze(-1) * (sigma ** 2) + sigma_t ** 2
        sigma_combined = torch.sqrt(torch.clamp(var_combined, min=1e-12))

        # 缓存给 dist_fn
        self._mu_t = mu_t
        self._sigma_t = sigma_combined

        return raw_logits, None  # Tianshou 期望返回 (logits, state)
  

    def dist_fn(self, raw_logits):
        assert self._mu_t is not None and self._sigma_t is not None, \
            "Call actor.forward() before dist_fn so that mu/sigma are cached."
        pi = F.softmax(raw_logits, dim=-1)  # [B, T, K]
        comp = Independent(Normal(self._mu_t, self._sigma_t), 1)  # 事件维=2
        return MixtureSameFamily(Categorical(pi), comp)


# ---------- Critic ----------
class DiffusionCritic(nn.Module):

    def __init__(self, backbone: DiffusionBackbone, n_diffusion_steps, actor_hidden):
        super().__init__()
        self.backbone = backbone
        self.n_timesteps = n_diffusion_steps
        self.value_head = nn.Linear(actor_hidden, 1)

    def forward(self, obs, state=None, info=None):
        device = next(self.parameters()).device
        obs = to_device_tensor_batch(obs, device=device)

        x_t   = obs["x_t"]
        cond  = obs["cond_feat"]
        grid  = obs["map_grid_feat"]
        traj  = obs["map_grid_traj"]
        t     = obs["current_step"]
        t_float = t.float() / (self.n_timesteps - 1)

        feat = self.backbone.features(x_t, cond, t_float, grid, traj)  # [B, T, H]
        v = self.value_head(feat[:, -1, :]).squeeze(-1)                # [B]
        return v
