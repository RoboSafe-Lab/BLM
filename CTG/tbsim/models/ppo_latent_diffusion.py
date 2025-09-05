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
from copy import deepcopy

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


from tbsim.utils.guidance_loss import PerturbationGuidance, verify_guidance_config_list, verify_constraint_config,apply_constraints
class PPO_LatentDiffusion(nn.Module):
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
        ddim_steps,
        latent_diffusion_dim
    

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

        self.model = ConvCrossAttnDiffuser( in_dim=latent_diffusion_dim,
                                            cond_dim = context_encoder_out_dim,
                                            time_emb_dim = time_emb_dim,
                                            hidden_dim = diffusion_hidden_dim,
                                            out_dim = latent_diffusion_dim,
                                            dilations = dilations,
                                            n_heads = num_heads,
                                            grid_map_dim = grid_feature_dim,
                                            grid_map_traj_dim = grid_feature_dim,
                                            mix_gauss = num_Gaussian
                                            )

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
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.default_chosen_inds = [4, 5]
        self.input_image_shape = input_image_shape
        self.ddim_steps = ddim_steps


        self.stride = 1
        self.apply_guidance_output = False
        self.guidance_optimization_params = None
        self.current_constraints = None
        self.transform = self.latent_grad_inner_transform
        self.transform_params = {'scaled_input': True, 'scaled_output': True, 'vae': None}

        # wrapper for optimization using current_guidance
        self.current_perturbation_guidance = PerturbationGuidance(self.transform, self.transform_params, self.scale_traj, self.descale_traj)

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

    def q_sample(self, x_start, t, noise=None):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def compute_loss(self, batch,vae):

        cond_feat, map_grid_feat = self.context_encoder(batch)
        
        center_fut_action = torch.cat([batch['center_fut_acc_lons'].unsqueeze(-1),
                                batch['center_fut_yaw_rates'].unsqueeze(-1)],dim=-1)
        
        x = self.scale_traj(center_fut_action)

        with torch.no_grad():
            mu, logvar = vae.vae_encoder(x) 
            std = (0.5 * logvar).exp()
            eps_post = torch.randn_like(std)
            z0 = mu + std * eps_post
    
        
 
        t = torch.randint(0, self.n_timesteps, (z0.shape[0],), device=z0.device)
        t_float = t.float() / (self.n_timesteps - 1)
    
        noise = torch.randn_like(z0)
        noised_action = self.q_sample(z0, t, noise)

        # map_grid_traj = self.query_map_feats(batch['center_fut_positions'],map_grid_feat,batch['raster_from_center'])#(B,T,32)

        raw_logits_pi, mu_raw, log_sigma_raw = self.model(noised_action, cond_feat,t_float,map_grid_feat)  
     
        pi = F.softmax(raw_logits_pi, dim=-1)
        sigma = torch.exp(log_sigma_raw)

        gmm = MixtureSameFamily(Categorical(pi), Independent(Normal(mu_raw, sigma), 1))
        logp = gmm.log_prob(z0)

        return -logp.mean()

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



    def forward(self, obs, vae, stationary_mask, global_t=0):
            if global_t == 0:
                self.stationary_mask = stationary_mask
            center_fut_action = torch.cat([obs['center_fut_acc_lons'].unsqueeze(-1),
                                obs['center_fut_yaw_rates'].unsqueeze(-1)],dim=-1)
        
            x = self.scale_traj(center_fut_action)

            with torch.no_grad():
                mu, logvar = vae.vae_encoder(x) 
                std = (0.5 * logvar).exp()
                eps_post = torch.randn_like(std)
                z0 = mu + std * eps_post
     
                
            return self.sample_ddim(obs,eta=0.3,z0=z0,vae=vae)

    def make_ddim_timesteps(self):
            c = torch.linspace(self.n_timesteps - 1, 0, self.ddim_steps, device=self.betas.device).long()
            next_c = list(c[1:].tolist()) + [0]
            return c.tolist(), next_c

    @torch.no_grad()
    def sample_ddim(self, batch, eta=0, z0=None, vae=None):

        # 1) 条件编码
        
        cond_feat, map_grid_feat = self.context_encoder(batch)
        self.transform_params['vae'] = vae
        B, L, d_latent = z0.shape
        

        # 2) 生成DDIM时间步
        timesteps, next_timesteps = self.make_ddim_timesteps()

        # 3) 初始化 x_T
        z_t = torch.randn((B, L, d_latent), device=z0.device)

        for t, t_next in zip(timesteps, next_timesteps):
            t_tensor      = torch.full((B,), t,      dtype=torch.long, device=cond_feat.device)
            t_next_tensor = torch.full((B,), t_next, dtype=torch.long, device=cond_feat.device)        
            t_float = t_tensor.float() / (self.n_timesteps - 1)



            logits_pi, mu, log_sigma = self.model(z_t, cond_feat,t_float,map_grid_feat)  

            pi = F.softmax(logits_pi, dim=-1)

            sigma = torch.exp(log_sigma)

            mix_dist = self.mix_dist(z_t, pi, mu, sigma,t_tensor, t_next_tensor, eta)

            z_next = mix_dist if isinstance(mix_dist, torch.Tensor) else mix_dist.sample()

            #------apply guidance------
            # is_last = (t_next == 0)
            # if (getattr(self, 'apply_guidance_intermediate', False) and not is_last) \
            # or (getattr(self, 'apply_guidance_output', False) and is_last):

            #     # 可用 σ_t 作为步长/裁剪的尺度（与上面的 t,t_next 对齐）
            #     alpha_t    = extract(self.alphas_cumprod, t_tensor,      z_t.shape)
            #     alpha_next = extract(self.alphas_cumprod, t_next_tensor, z_t.shape)
            #     sigma_t = eta * torch.sqrt(
            #         ((1 - alpha_t/alpha_next) * (1 - alpha_next) / (1 - alpha_t)).clamp_min(1e-8)
            #     )

            #     opt_params = deepcopy(self.guidance_optimization_params)
            #     # 若未显式给 lr/perturb_th，则用 σ_t 做自适应尺度
            #     if opt_params.get('lr', None) is None: opt_params['lr'] = sigma_t
            #     if opt_params.get('perturb_th', None) is None: opt_params['perturb_th'] = sigma_t

            #     # 重要：让 z 有梯度，PG 会在 transform(z, ...) 上回传并更新 z
            #     z_var = z_next.clone().detach().requires_grad_(True)
            #     z_guided, _ = self.current_perturbation_guidance.perturb(
            #         z_var, data_batch=batch, opt_params=opt_params, num_samp=1, return_grad_of=z_var
            #     )
            #     z_next = z_guided.detach()
            #------apply guidance------end

            z_t = z_next
        with torch.no_grad():
            x_t = vae.vae_decoder(z_bld=z_t,
                                cond_feat=None,
                                grid_map_traj_T=None) 

            if self.stationary_mask is not None:
                x_stationary = x_t[self.stationary_mask]
                x_stationary = self.descale_traj(x_stationary, [4, 5])
                x_stationary[...] = 0
                x_stationary = self.scale_traj(x_stationary, [4, 5])
                x_t[self.stationary_mask] = x_stationary

                
        curr_state = torch.cat([batch['center_curr_positions'],
                                batch['center_curr_speeds'].unsqueeze(-1),
                                batch['center_curr_yaws'].unsqueeze(-1)], dim=-1)


    
        traj = self.convert_action_to_state_and_action(x_t, curr_state,True,True)  # [B, T, 4]
        traj = traj[..., [0, 1, 3]] #(x,y,yaw)

        pred_positions = traj[..., :2]
        pred_yaws = traj[..., 2:3]

        out_dict = {
            "trajectories": traj,
            "predictions": {"positions": pred_positions, "yaws": pred_yaws},
        }
        return out_dict
    
    def mix_dist(self, x_t, pi, mu, sigma_gmm, t_tensor, t_next_tensor, eta):
        alpha_t    = extract(self.alphas_cumprod, t_tensor,      x_t.shape)
        alpha_next = extract(self.alphas_cumprod, t_next_tensor, x_t.shape)
        sqrt_alpha_t = torch.sqrt(alpha_t).unsqueeze(-1)
        sqrt_one_minus_alpha_t = torch.sqrt((1 - alpha_t).clamp_min(1e-8)).unsqueeze(-1)

        # eps_pred (x0-parameterization)
        eps_pred = (x_t.unsqueeze(2) - sqrt_alpha_t * mu) / sqrt_one_minus_alpha_t

        sigma_t = eta * torch.sqrt(
            ((1 - alpha_t / alpha_next) * (1 - alpha_next) / (1 - alpha_t)).clamp_min(1e-8)
        ).unsqueeze(-1)

        # 主路径均值
        sqrt_term = torch.sqrt(((1 - alpha_next).unsqueeze(-1) - sigma_t**2).clamp_min(0.0))
        mu_t = torch.sqrt(alpha_next).unsqueeze(-1) * mu + sqrt_term * eps_pred  # [B,T,K,dim]

        if eta == 0:
            # 方案B：MAP 分量
            k_star = pi.argmax(dim=-1)  # [B,T]
            x0_hat = torch.gather(
                mu, 2, k_star.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, mu.size(-1))
            ).squeeze(2)  # [B,T,dim]
            sqrt_a_t     = torch.sqrt(alpha_t)
            sqrt_1ma_t   = torch.sqrt((1 - alpha_t).clamp_min(1e-8))
            sqrt_a_next  = torch.sqrt(alpha_next)
            sqrt_1manext = torch.sqrt((1 - alpha_next).clamp_min(1e-8))
            eps_hat = (x_t - sqrt_a_t * x0_hat) / sqrt_1ma_t
            x_next  = sqrt_a_next * x0_hat + sqrt_1manext * eps_hat    # [B,T,dim]
            return x_next   # 直接返回张量（确定性）

        # eta>0: 仍然随机
        var_combined   = alpha_next.unsqueeze(-1) * (sigma_gmm**2) + sigma_t**2
        sigma_combined = torch.sqrt(var_combined.clamp_min(1e-12))
        comp = Independent(Normal(mu_t, sigma_combined), 1)
        mix = MixtureSameFamily(Categorical(pi), comp)
        return mix  # 返回分布


    def set_guidance(self, guidance_config, example_batch=None):
        '''
        Instantiates test-time guidance functions using the list of configs (dicts) passed in.
        '''
        if guidance_config is not None:
            if len(guidance_config) > 0 and verify_guidance_config_list(guidance_config):
                print('Instantiating test-time guidance with configs:')
                print(guidance_config)
                self.current_perturbation_guidance.set_guidance(guidance_config, example_batch)


    def set_constraints(self, constraint_config):
        '''
        Instantiates test-time hard constraints using the config (dict) passed in.
        '''
        if constraint_config is not None and len(constraint_config) > 0:
            verify_constraint_config(constraint_config)
            print('Instantiating test-time constraints with config:')
            print(constraint_config)
            self.current_constraints = constraint_config
    def update_guidance(self, **kwargs):
        if self.current_perturbation_guidance.current_guidance is not None:
            self.current_perturbation_guidance.update(**kwargs)

    def clear_guidance(self):
        self.current_perturbation_guidance.clear_guidance()

    def set_guidance_optimization_params(self, guidance_optimization_params):
        self.guidance_optimization_params = guidance_optimization_params

    def set_diffusion_specific_params(self, diffusion_specific_params):
        self.apply_guidance_intermediate = diffusion_specific_params['apply_guidance_intermediate']
        self.apply_guidance_output = diffusion_specific_params['apply_guidance_output']
        self.final_step_opt_params = diffusion_specific_params['final_step_opt_params']
        self.stride = diffusion_specific_params['stride']

    def state_action_grad_inner_transform(self, x_guidance, data_batch, transform_params, **kwargs):
        bsize = kwargs.get('bsize', x_guidance.shape[0])
        num_samp = kwargs.get('num_samp', 1)

        curr_states  = torch.cat([data_batch['center_curr_positions'],
                                data_batch['center_curr_speeds'].unsqueeze(-1),
                                data_batch['center_curr_yaws'].unsqueeze(-1)], dim=-1)
        expand_states = curr_states.unsqueeze(1).expand((bsize, num_samp, 4)).reshape((bsize*num_samp, 4))

        x_all = self.convert_action_to_state_and_action(x_guidance, expand_states, scaled_input=transform_params['scaled_input'], descaled_output=transform_params['scaled_output'])
        return x_all

# ------------------- 新增：latent 版 transform -------------------
    def latent_grad_inner_transform(self, z_guidance, data_batch, transform_params, **kwargs):

        bsize = kwargs.get('bsize', z_guidance.shape[0])
        num_samp = kwargs.get('num_samp', 1)
        vae = transform_params['vae']             # 从 transform_params 里拿到 VAE

        # 1) latent → 动作 (缩放域)
        x_action = vae.vae_decoder(z_bld=z_guidance, cond_feat=None, grid_map_traj_T=None)
        # 2) 动作 → state+action（输入是缩放域动作，所以 scaled_input=True）
        curr_states = torch.cat(
            [data_batch['center_curr_positions'],
            data_batch['center_curr_speeds'].unsqueeze(-1),
            data_batch['center_curr_yaws'].unsqueeze(-1)], dim=-1
        )
        expand_states = curr_states.unsqueeze(1).expand((bsize, num_samp, 4)).reshape((bsize*num_samp, 4))
        x_all = self.convert_action_to_state_and_action(
            x_action, expand_states, scaled_input=True, descaled_output=True
        )
        return x_all
