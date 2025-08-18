from itertools import cycle
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from tianshou.data import Batch
from tbsim.utils.safety_critical_batch_utils import parse_batch
import torch
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

        self.w_road = cfg.ppo.w_road
        self.w_proximity = cfg.ppo.w_proximity

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
            "current_step":      cs,
            "next_step":         ns,
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
                scaled_input=False, descaled_output=True
            )[..., :2]  # [1,T,2]

            # 你自己的奖励函数（下面按你原先签名示例）
            reward = compute_reward(
                pred_pos[0],
                # None if self._drivable_map is None else self._drivable_map[0],
                # None if self._neigh_pos   is None else self._neigh_pos[0],
                # None if self._neigh_avail is None else self._neigh_avail[0],
                # self._raster_from_center[0],
                # self.w_road, self.w_prox
            )
            reward = float(reward)
        else:
            reward = 0.0

        return self._obs(terminated=terminated), reward, bool(terminated), bool(truncated), {}
    
def compute_reward(pred_pos):
    return np.float32(1.0)