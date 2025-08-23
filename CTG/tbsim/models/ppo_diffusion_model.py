import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffuser_helpers import SinusoidalPosEmb


class ResidualDilatedConv1d(nn.Module):
    """真正的多尺度膨胀卷积；输入 (B, C, T) → 输出 (B, H, T)"""
    def __init__(self, in_channels, out_channels, dilations=(1,2,4), kernel_size=3):
        super().__init__()
        self.dilations = dilations
        self.branches = nn.ModuleList()
        for d in dilations:
            part = out_channels // len(dilations)
            pad = d * (kernel_size - 1) // 2
            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_channels // len(dilations), 
                            kernel_size, padding=pad, dilation=d),
                nn.GroupNorm(1,part),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        if out_channels % len(dilations) != 0:
            extra_channels = out_channels % len(dilations)
            pad = (kernel_size - 1) // 2
            self.extra_branch = nn.Sequential(
                nn.Conv1d(in_channels, extra_channels, kernel_size, padding=pad),
                nn.GroupNorm(1,extra_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.extra_branch = None

        self.shortcut = (nn.Conv1d(in_channels, out_channels, 1) 
                        if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        if self.extra_branch is not None:
            branch_outputs.append(self.extra_branch(x))
        
        parallel_out = torch.cat(branch_outputs, dim=1)  # (B, H, T)
        
        res = self.shortcut(x)
        
        return parallel_out + res  


class ConvCrossAttnDiffuser(nn.Module):
    def __init__(self,
                in_dim,        
                cond_dim,      
                time_emb_dim,  
                hidden_dim=64, 
                out_dim=2, 
                dilations=(1,2,4),
                n_heads=4,
                grid_map_dim=32,
                grid_map_traj_dim=32,
                ):
        super().__init__()
       

        # 1) 时序特征提取
        self.temporal_conv = ResidualDilatedConv1d(
            in_channels=in_dim,
            out_channels=hidden_dim,
            dilations=dilations,
            kernel_size=3
        )

        # 2) 时间嵌入 → FiLM 参数
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),         # (B, time_emb_dim)
            nn.Linear(time_emb_dim, hidden_dim*2),  # 拆成 gamma_t, beta_t
        )

        # 3) 全局条件 Cond - 用于Cross-Attention
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.cond_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.cond_norm = nn.LayerNorm(hidden_dim)

        self.cond_ffn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
        self.cond_ffn_norm = nn.LayerNorm(hidden_dim)
        
        # 3) 静态网格 (整图) Cross-Attn
        self.grid_map_pos_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.grid_map_gate = nn.Parameter(torch.zeros(1))
        self.grid_map_proj = nn.Linear(grid_map_dim, hidden_dim)
        self.grid_map_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.grid_map_norm = nn.LayerNorm(hidden_dim)

        
        # 4) grid_map_traj Cross-Attention
        self.register_buffer("grid_pos_cache", None, persistent=False)
        self.grid_traj_proj = nn.Linear(grid_map_traj_dim, hidden_dim)
        self.grid_traj_norm = nn.LayerNorm(hidden_dim)
        self.traj_pos_emb = nn.Embedding(512, hidden_dim)  # 可选：显式时间步编码
        

        # 5) 输出头
        self.out_dim = nn.Linear(hidden_dim, out_dim)

    def _get_pos_emb(self, B, Hm, Wm, device):
        if (self.grid_pos_cache is None) or (self.grid_pos_cache.shape[1] != Hm*Wm):
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, Hm, device=device),
                torch.linspace(-1, 1, Wm, device=device),
                indexing='ij'
            )
            pos = torch.stack([yy, xx], dim=-1).view(1, Hm*Wm, 2)
            self.grid_pos_cache = pos  # (1, N, 2)
        return self.grid_pos_cache.expand(B, -1, -1)

    def forward(self, x, cond_feat, t, grid_map_feat, grid_map_traj):
        h = self.get_backbone(x, cond_feat, t, grid_map_feat, grid_map_traj)
        # 5) 输出
        
        out = self.out_dim(h)
        return out
    
    def get_backbone(self,x,cond_feat,t,grid_map_feat,grid_map_traj):
        h = self.apply_conv(x)      # [B, T, H]
        h = self.apply_time(h, t)   # [B, T, H]
        h = self.apply_cond(h, cond_feat)  # [B, T, H]
        h = self.apply_grid_map(h, grid_map_feat)  # [B, T, H]
        h = self.apply_grid_traj(h, grid_map_traj)  # [B, T, H]
        return h

    def apply_conv(self,x):
        h = x.permute(0,2,1)
        h = self.temporal_conv(h)        # → (B, H, T)
        h = h.permute(0,2,1)             # → (B, T, H)
        return h
    
    def apply_time(self,x,t):
        t_emb = self.time_mlp(t)
        gamma_t, beta_t = t_emb.chunk(2, dim=-1)  # 各 (B, H)
        gamma_t = gamma_t.unsqueeze(1)
        beta_t  = beta_t .unsqueeze(1)
        return gamma_t * x + beta_t  # 时间调制
    
    def apply_cond(self,h,cond_feat):
        cond_token = self.cond_proj(cond_feat).unsqueeze(1)  # (B, 1, H)
        attn_out, _ = self.cond_attn(h, cond_token, cond_token)
        h = self.cond_norm(h + attn_out)  # 残差连接 + LayerNorm
        cond_ffn_out = self.cond_ffn(h)
        return self.cond_ffn_norm(h + cond_ffn_out)
    
    def apply_grid_map(self, h, grid_map_feat):
        B, Cmap, Hm, Wm = grid_map_feat.shape
        m_flat = grid_map_feat.view(B, Cmap, -1).permute(0,2,1) 
        m_proj = self.grid_map_proj(m_flat)  # (B, N_loc, hidden_dim)
        pos = self._get_pos_emb(B, Hm, Wm, grid_map_feat.device)
        pos_emb = self.grid_map_pos_mlp(pos) 
        m_proj = m_proj + pos_emb
        
        map_attn_out, _ = self.grid_map_attn(query=h, key=m_proj, value=m_proj)
        h = h + torch.sigmoid(self.grid_map_gate) * map_attn_out
        return self.grid_map_norm(h)
    
    def apply_grid_traj(self, h, grid_map_traj):
        m = self.grid_traj_proj(grid_map_traj)     # [B, T, H]
        T = h.size(1)
        step_idx = torch.arange(T, device=h.device)
        h  = h + self.traj_pos_emb(step_idx).unsqueeze(0)
        h = h + m
        return self.grid_traj_norm(h)

    def apply_mix_gauss(self, h):
        B, T, H = h.shape
        h = h.permute(0,2,1)
        raw_logits_pi = self.logits_pi(h).permute(0,2,1).view(B,T,self.mix_gauss)
        
        mu_raw = self.logits_mu(h).permute(0,2,1)\
        .view(B,T,self.mix_gauss,self.out_dim)

        log_sigma_raw = self.logits_sigma(h).permute(0,2,1)\
            .view(B,T,self.mix_gauss,self.out_dim)
        
        log_sigma_raw = torch.clamp(log_sigma_raw, min=-5, max=0)
        
        return raw_logits_pi, mu_raw, log_sigma_raw