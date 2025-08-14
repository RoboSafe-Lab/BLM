import torch
import torch.nn as nn
import torch.nn.functional as F
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
from .context_encoder import ContextEncoder
class PPO_Diffuser(nn.Module):
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

        time_emb_dim,
        diffusion_hidden_dim,
        dilations,
        num_heads,
        num_Gaussian,


        dynamics_kwargs,
        n_timesteps,

        dt,
     


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

        # self.center_state = MLP(in_dim = 4,
        #                         out_dim = state_encoder_out_dim,
        #                         hidden_dims = (state_encoder_out_dim,state_encoder_out_dim))
        # cond_in_feat_size += state_encoder_out_dim

        self.neighbor_hist = NeighborHistoryEncoder(num_steps=history_frames,
                                                    out_dim=neighbor_history_out_dim,
                                                    norm_info=norm_info_neighbor)
        cond_in_feat_size += neighbor_history_out_dim

        combine_layer_dims = (cond_in_feat_size, cond_in_feat_size, context_encoder_hidden_dim, context_encoder_hidden_dim)
        self.process_cond_mlp = MLP(in_dim = cond_in_feat_size,
                                    out_dim = context_encoder_out_dim,
                                    hidden_dims = combine_layer_dims)




        self.model = ConvCrossAttnDiffuser( in_dim=2,
                                            cond_dim = context_encoder_out_dim,
                                            time_emb_dim = time_emb_dim,
                                            hidden_dim = diffusion_hidden_dim,
                                            out_dim = 2,
                                            dilations = dilations,
                                            n_heads = num_heads,
                                            grid_map_traj_dim = grid_feature_dim,
                                            mix_gauss = num_Gaussian)


        self._dynamics_kwargs = dynamics_kwargs
        self._create_dynamics(dynamics_kwargs)        

        self.dt = dt                               

        norm_add_coeffs = norm_info_center[0]
        norm_div_coeffs = norm_info_center[1]
        self.add_coeffs = np.array(norm_add_coeffs).astype('float32')
        self.div_coeffs = np.array(norm_div_coeffs).astype('float32')   
        print('self.add_coeffs', self.add_coeffs)
        print('self.div_coeffs', self.div_coeffs)    



        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)


        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        self.sqrt_alphas_over_one_minus_alphas_cumprod = torch.sqrt(alphas_cumprod / (1.0 - alphas_cumprod))
        self.sqrt_recip_one_minus_alphas_cumprod = 1.0 / torch.sqrt(1. - alphas_cumprod)

        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

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

    def q_sample(self, x_start, t, noise=None):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

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
        # center_state = self.center_state(torch.cat([batch["center_curr_positions"],
        #                                             batch["center_curr_speeds"].unsqueeze(-1),
        #                                             batch["center_curr_yaws"].unsqueeze(-1)],-1))
        
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
        
        scaled_action = self.scale_traj(center_fut_action).detach()

        t = torch.randint(0, self.n_timesteps, (scaled_action.shape[0],), device=scaled_action.device)

        t_float = t.float() / (self.n_timesteps - 1) 
        noise_init = torch.randn_like(scaled_action)
        noised_action = self.q_sample(scaled_action, t, noise_init)

        
        map_grid_traj = self.query_map_feats(batch['center_fut_positions'].detach(),
                                                        map_grid_feat,
                                                        batch['raster_from_center'],
                                                        )#(B,T,32)
        
   
        raw_logits_pi, mu_raw, log_sigma_raw = self.model(noised_action, cond_feat,t_float,map_grid_feat,map_grid_traj)  # 使用修正后的地图特征
        
        pi = F.softmax(raw_logits_pi, dim=-1)
        sigma = torch.exp(log_sigma_raw)

        gmm = MixtureSameFamily(Categorical(pi), Independent(Normal(mu_raw, sigma), 1))
        logp = gmm.log_prob(scaled_action)

        return -logp.mean()





    def make_ddim_timesteps(self, ddim_steps: int, n_timesteps: int):
        c = torch.linspace(n_timesteps - 1, 0, ddim_steps, device=self.betas.device).long()
        next_c = list(c[1:].tolist()) + [0]
        return c.tolist(), next_c

    def convert_action_to_state_and_action(self, x_out, curr_states, scaled_input=True, descaled_output=False):
 
        dim = len(x_out.shape)
        if dim == 4:
            B, N, T, _ = x_out.shape
            x_out = TensorUtils.join_dimensions(x_out,0,2)

        if scaled_input:
            x_out = self.descale_traj(x_out, [4, 5])
        x_out_state = unicyle_forward_dynamics(
            dyn_model=self.dyn,
            initial_states=curr_states,
            actions=x_out,
            step_time=self.dt,
            mode='parallel'
        )

        x_out_all = torch.cat([x_out_state, x_out], dim=-1)
        if scaled_input and not descaled_output:
            x_out_all = self.scale_traj(x_out_all, [0, 1, 2, 3, 4, 5])

        if dim == 4:
            x_out_all = x_out_all.reshape([B, N, T, -1])
        return x_out_all

    def forward(self, obs, stationary_mask, global_t=0):
        if global_t == 0:
            self.stationary_mask = stationary_mask
        return self.sample_ddim(obs)

    @torch.no_grad()
    def sample_ddim(self, batch, num_samples: int = 1, ddim_steps: int = 50, 
                    eta: float = 0.0, cfg_scale: float = 1.5, use_cfg: bool = True):
        # 1) 条件编码
        cond_feat, map_grid_feat = self.context_encoder(batch)
        B = batch['center_fut_positions'].size(0)
        T = 50
        curr_state = torch.cat([
            batch['center_curr_positions'],
            batch['center_curr_speeds'].unsqueeze(-1),
            batch['center_curr_yaws'].unsqueeze(-1)
        ], dim=-1)

        # 2) 生成DDIM时间步
        timesteps, next_timesteps = self.make_ddim_timesteps(
            self.ddim_steps, self.n_timesteps
        )
           # for _ in range(num_samples):
            # 3) 初始化 x_T
        x_t = torch.randn((B, T, 2), device=curr_state.device)
        for t, t_next in zip(timesteps, next_timesteps):
            t_tensor      = torch.full((B,), t,      dtype=torch.long, device=x_t.device)
            t_next_tensor = torch.full((B,), t_next, dtype=torch.long, device=x_t.device)
            
            t_float = t_tensor.float() / (self.n_timesteps - 1)

            # 4) 解码当前位置用于地图特征查询
            pred_positions = self.convert_action_to_state_and_action(x_t, curr_state,True,True)[..., :2]  # [B, T, 2]
            
            # 5) 查询地图轨迹特征
            map_grid_traj = self.query_map_feats(
                pred_positions, map_grid_feat,
                batch['raster_from_center']
            )
            raw_logits_cond, mu_cond, log_sigma_cond= self.model(x_t, cond_feat, t_float, map_grid_feat, map_grid_traj)
            if use_cfg:
                cond_zero      = torch.zeros_like(cond_feat)
                map_grid_traj_zero = torch.zeros_like(map_grid_traj)

                raw_logits_un, mu_un, log_sigma_un = self.model(x_t, cond_zero, t_float, map_grid_feat, map_grid_traj_zero)
                
                raw_logits = raw_logits_un + cfg_scale * (raw_logits_cond - raw_logits_un)
                mu         = mu_un + cfg_scale * (mu_cond   - mu_un)
                log_sigma  = log_sigma_un + cfg_scale * (log_sigma_cond - log_sigma_un)
            else:
                raw_logits = raw_logits_cond
                mu         = mu_cond
                log_sigma  = log_sigma_cond
            
            pi = F.softmax(raw_logits, dim=-1)
            
            log_sigma = torch.clamp(log_sigma, min=-20, max=2)
            sigma = torch.exp(log_sigma)
            


            mix_dist = self.mix_dist(x_t, pi, mu, sigma,t_tensor, t_next_tensor, eta)
            x_t = mix_dist.sample()

            if self.stationary_mask is not None:
                x_stationary = x_t[self.stationary_mask]
                x_stationary = self.descale_traj(x_stationary, [4, 5])
                x_stationary[...] = 0
                x_stationary = self.scale_traj(x_stationary, [4, 5])
                x_t[self.stationary_mask] = x_stationary



        # 8) 最后解码完整状态
 
        traj = self.convert_action_to_state_and_action(x_t, curr_state,True,True)  # [B, T, 4]
        traj = traj[..., [0, 1, 3]] #(x,y,yaw)

        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]

        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
        }
        return out_dict
    
    def mix_dist(self,x_t, pi, mu, sigma_gmm, t_tensor, t_next_tensor, eta):
 
        
        alpha_t = extract(self.alphas_cumprod, t_tensor, x_t.shape)
        alpha_next = extract(self.alphas_cumprod, t_next_tensor, x_t.shape)
        
        sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(-1)
        sqrt_one_minus_alpha_t = torch.sqrt((1 - alpha_t).clamp_min(1e-8)).unsqueeze(-1)
        
        # 计算eps_pred
        eps_pred = (x_t.unsqueeze(2) - sqrt_alpha_t * mu) / sqrt_one_minus_alpha_t
        
        # DDIM方差
        sigma_t = eta * torch.sqrt(
            ((1 - alpha_t / alpha_next) * (1 - alpha_next) / (1 - alpha_t)).clamp_min(1e-8)
        ).unsqueeze(-1)
        
        # 均值计算
        mu_t = (
            torch.sqrt(alpha_next).unsqueeze(-1) * mu
            + torch.sqrt((1 - alpha_next).unsqueeze(-1) - sigma_t**2) * eps_pred
        )
        
        if eta == 0:
            # 确定性DDIM：每个分量都是确定性的
            # 使用很小的方差来近似Dirac delta分布
            min_sigma = 1e-8
            sigma_combined = torch.full_like(mu_t, min_sigma)
        else:
            # 随机DDIM：组合GMM方差和DDIM方差
            var_combined = alpha_next.unsqueeze(-1) * (sigma_gmm**2) + sigma_t**2
            sigma_combined = torch.sqrt(var_combined)
        
        # 构造混合分布
        comp_dist = Independent(Normal(mu_t, sigma_combined), 1)
        mix_dist = MixtureSameFamily(
            mixture_distribution=Categorical(pi),
            component_distribution=comp_dist
        )
        return mix_dist