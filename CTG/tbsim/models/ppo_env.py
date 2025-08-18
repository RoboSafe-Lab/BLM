from itertools import cycle
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tianshou.data import Batch
from tbsim.utils.safety_critical_batch_utils import parse_batch
import torch
from tbsim.utils.geometry_utils import transform_points_tensor
def preprocess_fn(
    obs=None, obs_next=None, rew=None, done=None, info=None,
    policy=None, env_id=None, act=None
):

    def to_numpy_and_squeeze(x):
        if hasattr(x, 'cpu') and hasattr(x, 'numpy'):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
        if isinstance(arr, np.ndarray) and arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        return arr
    result = {}

    for name, item in zip(['obs', 'obs_next'], [obs, obs_next]):
        if item is not None:
            item_list = item.tolist() if isinstance(item, np.ndarray) else item
            keys = item_list[0].keys()
            batch_dict = {
                k: np.stack([to_numpy_and_squeeze(o[k]) for o in item_list], axis=0)
                for k in keys
            }
            result[name] = Batch(batch_dict)

    return result if result else None

def _to_device_batch(batch: dict, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            # 尝试转换为张量，如果失败则保持原值
            try:
                if not isinstance(v, str) and v is not None:
                    t = torch.as_tensor(v)
                    out[k] = t.to(device)
                else:
                    out[k] = v
            except (ValueError, TypeError, RuntimeError):
                out[k] = v
    return out

class PPOEnv(gym.Env):
    def __init__(self,cfg,data_module, model):             
        super().__init__()
        # —— 1) 数据源 —— 
        self.data_module = data_module
        self.data_module.setup("fit")
        self.data_loader = data_module.train_dataloader()
        self.iterator = cycle(self.data_loader)

        self.model = model
        # —— 2) 核心模型 —— 
        self.device = next(model.parameters()).device

        self.ddim_steps = model.ddim_steps
        self.n_timesteps = model.n_timesteps

        self.T = cfg.future_num_frames

        Cmap = cfg.grid_feature_dim
        Cctx = cfg.context_encoder_out_dim
        H = W = 56 

        self.observation_space = spaces.Dict({
            "x_t":               spaces.Box(-np.inf, np.inf, shape=(self.T, 2), dtype=np.float32),
            "cond_feat":         spaces.Box(-np.inf, np.inf, shape=(Cctx,), dtype=np.float32),
            "map_grid_feat":     spaces.Box(-np.inf, np.inf, shape=(Cmap, H, W), dtype=np.float32),
            "raster_from_center":spaces.Box(-np.inf, np.inf, shape=(3, 3), dtype=np.float32),
            "curr_state":        spaces.Box(-np.inf, np.inf, shape=(4,), dtype=np.float32),
            "current_step":      spaces.Discrete(self.n_timesteps + 1),
            "next_step":         spaces.Discrete(self.n_timesteps + 1),
        })
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(self.T, 2), dtype=np.float32)

        self.w_road = cfg.w_road
        self.w_proximity = cfg.w_proximity

    @staticmethod
    def _to_np(x):
        return x.detach().cpu().numpy().astype(np.float32)

    def _obs(self, terminated=False):
        cs = int(self.timesteps[self.t_idx]) if not terminated else 0
        ns = int(self.next_timesteps[self.t_idx]) if not terminated else 0
        return {
            "x_t":               self._to_np(self.x_t[0]),
            "cond_feat":         self._to_np(self._cond_feat[0]),
            "map_grid_feat":     self._to_np(self._map_grid_feat[0]),
            "raster_from_center":self._to_np(self._raster_from_center[0]),
            "curr_state":        self._to_np(self._curr_state[0]),
            "current_step":      np.array(cs, dtype=np.int64),  # 确保是形状[1]的数组
            "next_step":         np.array(ns, dtype=np.int64),  # 确保是形状[1]的数组
        }

    def reset(self, seed=42, options=None):
        # —— 1) 拿下一批数据，并搬到 GPU —— 
        batch = next(self.iterator)
            
        batch = parse_batch(batch)
        batch    = _to_device_batch(batch, self.device)

        # —— 2) 条件编码 —— 
        cond_feat, map_grid_feat = self.model.context_encoder(batch)

        # —— 3) 当前物理状态，用于最终轨迹解码 ——  
        curr_state = torch.cat([
            batch['center_curr_positions'],
            batch['center_curr_speeds'].unsqueeze(-1),
            batch['center_curr_yaws'].unsqueeze(-1)
        ], dim=-1)
        
        # —— 4) 初始化 DDIM 时间表 —— 
        self.timesteps, self.next_timesteps = self.model.make_ddim_timesteps(self.ddim_steps, self.n_timesteps)
        
        self.t_idx = 0
        # —— 5) 初始化噪声 x_T —— 

        self.x_t = torch.randn((1, self.T, 2), device=self.device)

        self._cond_feat = cond_feat
        self._map_grid_feat = map_grid_feat
        self._curr_state = curr_state
        self._raster_from_center  = batch['raster_from_center']

        self._drivable_map = batch['drivable_map']
        self._neigh_fut_positions = batch['neigh_fut_positions']
        self._neigh_fut_availabilities = batch['neigh_fut_availabilities']
        return self._obs(terminated= False), {}



    def step(self, action):
        # action 即下一步的 x_t（来自 Actor 的采样）
        if isinstance(action, np.ndarray):
            action = torch.as_tensor(action)
        action = action.to(self.device, dtype=torch.float32).unsqueeze(0)
        self.x_t = action

        # 推进一步
        self.t_idx += 1
        terminated = (self.t_idx == len(self.timesteps))
        truncated = False

        # 终止时解码一次，计算 reward（非终止步 reward=0；如需 shaping 可自行加）
        if terminated:
            # 解码到 agent frame 的 (x,y)，再算任务奖励
            x_denorm = self.model.descale_traj(self.x_t)  # [1,T,2]
            pred_pos = self.model.convert_action_to_state_and_action(
                x_denorm, self._curr_state,
                scaled_input=True, descaled_output=True
            )[..., :2]  # [1,T,2]

            # 你自己的奖励函数（下面按你原先签名示例）
            reward = compute_reward(
                pred_pos,
                self._drivable_map,
                self._neigh_fut_positions,
                self._neigh_fut_availabilities,
                self._raster_from_center,
                self.w_road,
                self.w_proximity
            )
            reward = float(reward)
        else:
            reward = 0.0

        return self._obs(terminated=terminated), reward, bool(terminated), bool(truncated), {}
    
def compute_road_reward(pred_positions,
                        drivable_map,
                        raster_from_center,               
                        r_in: float = 0.1,                 # 在道路内给的正奖励
                        r_out: float = -1.0 ):               # 出界惩罚

    
    
    ego_px = transform_points_tensor(pred_positions, raster_from_center)  # (T,2)
    ix = ego_px[..., 0].clamp(0, drivable_map.shape[1]-1).long()        # (T)
    iy = ego_px[..., 1].clamp(0, drivable_map.shape[0]-1).long()        # (T)
    flags = drivable_map[iy, ix]                                        # (T)
    # 4) 映射到 [-1, r_in] 区间：在道内≈+r_in，出界≈r_out
    reward_road = torch.where(flags > 0.5,
                            torch.full_like(flags, r_in, dtype=torch.float32),
                            torch.full_like(flags, r_out, dtype=torch.float32)).mean()
    return reward_road



def compute_proximity_reward(
        pred_positions, neigh_fut_positions, neigh_fut_availabilities,
        d_col: float = 0.5,     # 碰撞阈
        d_tar: float = 1.5,     # 理想 near‑miss
        sigma: float = 0.4,
        hard_penalty: float = -1.0):

    T = min(pred_positions.shape[-2], neigh_fut_positions.shape[-2])

    dists = torch.norm(pred_positions[:T]- neigh_fut_positions[:, :T, :], dim=-1) # (N,T)

    mask = neigh_fut_availabilities[:, :T].bool()
    dists = torch.where(mask, dists, torch.full_like(dists, float('inf')))

    dist_min = dists.min(dim=0).values  # (T)

    # ---------- soft 奖励 ----------
    r_soft = torch.exp(- (dist_min - d_tar) ** 2 / (2 * sigma ** 2))
    # 只在 d_col < d <= d_tar 区间给奖励，其余置 0
    r_soft = torch.where((dist_min > d_tar) | (dist_min < d_col),torch.zeros_like(r_soft), r_soft)

    # ---------- hard 碰撞惩罚 ----------
    r_hard = torch.where(dist_min < d_col,
                        torch.full_like(r_soft, hard_penalty),
                        torch.zeros_like(r_soft))

    return (r_soft + r_hard).mean()

def compute_reward(pred_positions,
                drivable_map,
                neigh_fut_positions,
                neigh_fut_availabilities,
                raster_from_center,
                w_road: float = 1.0,
                w_proximity: float = 1.0):
    road_r = compute_road_reward(pred_positions,drivable_map,raster_from_center)

    prox_r = compute_proximity_reward(pred_positions.squeeze(0),neigh_fut_positions.squeeze(0),
                                    neigh_fut_availabilities)
    # 3) 加权合成
    reward = w_road * road_r  + w_proximity * prox_r
    return reward.detach().cpu().numpy()
