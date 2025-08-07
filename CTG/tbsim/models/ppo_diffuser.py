import torch
import torch.nn as nn
import torch.nn.functional as F
from .ppo_diffusion_model import ConvCrossAttnDiffuser
import tbsim.models.base_models as base_models
from torch.distributions import Categorical, Normal,Independent,MixtureSameFamily
from tbsim.models.diffuser_helpers import (
    AgentHistoryEncoder,
    NeighborHistoryEncoder,
    MapEncoder,
)
import numpy as np
import tbsim.dynamics as dynamics
from tbsim.utils.geometry_utils import transform_points_tensor
from .diffuser_helpers import (
    cosine_beta_schedule,
    query_feature_grid,
    unicyle_forward_dynamics,
    extract
)
import tbsim.utils.tensor_utils as TensorUtils
class PPO_Diffuser(nn.Module):
    def __init__(
        self, 
        map_encoder_model_arch,
        input_image_shape,
        map_feature_dim,
        map_grid_feature_dim,
        hist_num_frames,
        agent_hist_feat_dim,
        state_in_dim,
        state_out_feat_dim,
        neighbor_hist_feat_dim,

        cond_feature_dim,
        time_emb_dim,
        attn_hidden_dim,
        attn_dilations,
        attn_n_heads,
        grid_map_traj_dim,
        mix_gauss,

        horizon,
        ddim_steps,

        diffuser_norm_info,
        agent_hist_norm_info,
        neighbor_hist_norm_info,
        dynamics_type=None,
        dynamics_kwargs={},

        n_timesteps=100,

        dt=0.1,
     


    ):
        super().__init__()

        cond_in_feat_size = 0

        self.map_encoder = MapEncoder(model_arch=map_encoder_model_arch,
                            input_image_shape=input_image_shape,
                            global_feature_dim=map_feature_dim,
                            grid_feature_dim=map_grid_feature_dim)
        cond_in_feat_size += map_feature_dim
        
        self.center_hist = AgentHistoryEncoder(num_steps=hist_num_frames,
                                                out_dim=agent_hist_feat_dim,
                                                norm_info=agent_hist_norm_info)
        cond_in_feat_size += agent_hist_feat_dim

        self.center_state = base_models.MLP(state_in_dim,
                                        state_out_feat_dim,
                                        (state_out_feat_dim,state_out_feat_dim),
                                        normalization=True)
        cond_in_feat_size += state_out_feat_dim

        self.neighbor_hist = NeighborHistoryEncoder(num_steps=hist_num_frames,
                                                    out_dim=neighbor_hist_feat_dim,
                                                    norm_info=neighbor_hist_norm_info)
        cond_in_feat_size += neighbor_hist_feat_dim

        combine_layer_dims = (cond_in_feat_size, cond_in_feat_size, cond_feature_dim, cond_feature_dim)
        self.process_cond_mlp = base_models.MLP(cond_in_feat_size,
                                                cond_feature_dim,
                                                combine_layer_dims,
                                                normalization=True)

        self._dynamics_type = dynamics_type
        self._dynamics_kwargs = dynamics_kwargs
        self._create_dynamics()        

        self.dt = dt                               

        norm_add_coeffs = diffuser_norm_info[0]
        norm_div_coeffs = diffuser_norm_info[1]
        self.add_coeffs = np.array(norm_add_coeffs).astype('float32')
        self.div_coeffs = np.array(norm_div_coeffs).astype('float32')   
        print('self.add_coeffs', self.add_coeffs)
        print('self.div_coeffs', self.div_coeffs)    

        self.horizon = horizon

        self.model = ConvCrossAttnDiffuser( in_dim=2,
                                            cond_dim=cond_feature_dim,
                                            time_emb_dim = time_emb_dim,
                                            hidden_dim = attn_hidden_dim,
                                            out_dim = 2,
                                            dilations = attn_dilations,
                                            n_heads = attn_n_heads,
                                            grid_map_traj_dim = grid_map_traj_dim,
                                            mix_gauss = mix_gauss)

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
        self.ddim_steps = ddim_steps
        self.input_image_shape = input_image_shape
        


    def _create_dynamics(self):
        if self._dynamics_type in ["Unicycle", dynamics.DynType.UNICYCLE]:
            self.dyn = dynamics.Unicycle(
                "dynamics",
                max_steer=self._dynamics_kwargs["max_steer"],
                max_yawvel=self._dynamics_kwargs["max_yawvel"],
                acce_bound=self._dynamics_kwargs["acce_bound"]
            )
        else:
            self.dyn = None

    def scale_traj(self, target_traj_orig, chosen_inds=[]):

        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D
        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device)
        target_traj = (target_traj_orig + dx_add) / dx_div

        return target_traj

    def descale_traj(self, target_traj_orig, chosen_inds=[]):

        if len(chosen_inds) == 0:
            chosen_inds = self.default_chosen_inds
        add_coeffs = self.add_coeffs[chosen_inds][None,None] # 1 x 1 x D
        div_coeffs = self.div_coeffs[chosen_inds][None,None] # 1 x 1 x D

        device = target_traj_orig.get_device()
        dx_add = torch.tensor(add_coeffs, device=device)
        dx_div = torch.tensor(div_coeffs, device=device) 

        target_traj = target_traj_orig * dx_div - dx_add

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
    def context_encoder(self, batch):
        global_feat, grid_feat = self.map_encoder(batch.maps)
        center_hist = self.center_hist(batch.center_hist_positions,       
                                        batch.center_hist_yaws.unsqueeze(-1),
                                        batch.center_hist_speeds,
                                        batch.center_extent,
                                        batch.center_hist_availabilities)
        center_state = self.center_state(torch.cat([batch.center_curr_positions,
                                                    batch.center_curr_speeds.unsqueeze(-1),
                                                    batch.center_curr_yaws.unsqueeze(-1)],-1))
        
        neigh_hist = self.neighbor_hist(batch.neigh_hist_positions,
                                        batch.neigh_hist_yaws.unsqueeze(-1),
                                        batch.neigh_hist_speeds,
                                        batch.neigh_extent,
                                        batch.neigh_hist_availabilities)
        cond_feat = self.process_cond_mlp(torch.cat([global_feat, center_hist, center_state, neigh_hist], -1))
        return cond_feat,grid_feat

    def compute_loss(self, batch,p_drop: float = 0.1):

        cond_feat, map_grid_feat = self.context_encoder(batch)
        
        center_fut_action = torch.cat([batch.center_fut_acc_lons.unsqueeze(-1),
                                batch.center_fut_yaw_rates.unsqueeze(-1)],dim=-1)
        
        scaled_action = self.scale_traj(center_fut_action).detach()

        t = torch.randint(0, self.n_timesteps, (scaled_action.shape[0],), device=scaled_action.device)

        t_float = t.float() / (self.n_timesteps - 1) 
        noise_init = torch.randn_like(scaled_action)
        noised_action = self.q_sample(scaled_action, t, noise_init)

        map_grid_traj = self.query_map_feats(batch.center_fut_positions.detach(),
                                                        map_grid_feat,
                                                        batch.raster_from_center)#(B,T,32)
        
        uncond_idx = torch.rand(scaled_action.shape[0], device=scaled_action.device) < p_drop
        cond_feat_zero = cond_feat.clone()
        map_grid_traj_zero = map_grid_traj.clone()
        cond_feat_zero[uncond_idx] = 0.0
        map_grid_traj_zero[uncond_idx] = 0.0

        raw_logits_pi, mu_raw, log_sigma_raw = self.model(noised_action, 
                            cond_feat_zero,  # 使用修正后的条件
                            t_float,
                            map_grid_feat,
                            map_grid_traj_zero)  # 使用修正后的地图特征
        
        pi = F.softmax(raw_logits_pi, dim=-1)
        sigma = torch.exp(log_sigma_raw)

        gmm = MixtureSameFamily(Categorical(pi), 
                                Independent(Normal(mu_raw, sigma), 1))
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


    @torch.no_grad()
    def sample_ddim(self, batch, num_samples: int = 1, ddim_steps: int = 50, 
                    eta: float = 0.0, cfg_scale: float = 1.5, use_cfg: bool = True):
        # 1) 条件编码
        cond_feat, map_grid_feat = self.context_encoder(batch)
        B = batch.center_fut_positions.size(0)
        T = 50
        curr_state = torch.cat([
            batch.center_curr_positions,
            batch.center_curr_speeds.unsqueeze(-1),
            batch.center_curr_yaws.unsqueeze(-1)
        ], dim=-1)

        # 2) 生成DDIM时间步
        timesteps, next_timesteps = self.make_ddim_timesteps(
            self.ddim_steps, self.n_timesteps
        )
        results = []

        for _ in range(num_samples):
            # 3) 初始化 x_T
            x_t = torch.randn((B, T, 2), device=curr_state.device)
            for t, t_next in zip(timesteps, next_timesteps):
                t_tensor      = torch.full((B,), t,      dtype=torch.long, device=x_t.device)
                t_next_tensor = torch.full((B,), t_next, dtype=torch.long, device=x_t.device)
                
                t_float = t_tensor.float() / (self.n_timesteps - 1)

                # 4) 解码当前位置用于地图特征查询
                action_denorm = self.descale_traj(x_t)
                pred_positions = self.convert_action_to_state_and_action(
                    self.dyn, action_denorm, curr_state
                )[..., :2]  # [B, T, 2]
                
                # 5) 查询地图轨迹特征
                map_grid_traj = self.query_map_feats(
                    pred_positions, map_grid_feat,
                    batch.raster_from_center, batch.maps.shape[1:]
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



            # 8) 最后解码完整状态
            u_denorm = self.descale_traj(x_t)
            full_states = self.convert_action_to_state_and_action(
                self.dyn, u_denorm, curr_state
            )  # [B, T, 4]
            results.append({'full_states': full_states})

        return results
    
