from tbsim.utils.batch_utils import BatchUtils
import torch


import tbsim.utils.tensor_utils as TensorUtils


import tbsim.utils.geometry_utils as GeoUtils
from trajdata.data_structures.batch import AgentBatch, SceneBatch
import numpy as np
from trajdata.data_structures import SceneBatchElement,AgentBatchElement
from trajdata.utils.arr_utils import angle_wrap
BATCH_RASTER_CFG = None
def set_global_trajdata_batch_raster_cfg(raster_cfg):
    global BATCH_RASTER_CFG
    assert "include_hist" in raster_cfg
    assert "pixel_size" in raster_cfg
    assert "raster_size" in raster_cfg
    assert "ego_center" in raster_cfg
    assert "num_sem_layers" in raster_cfg
    assert "no_map_fill_value" in raster_cfg
    assert "drivable_layers" in raster_cfg
    BATCH_RASTER_CFG = raster_cfg
def get_drivable_region_map(maps):
    if isinstance(maps, torch.Tensor):
        drivable = torch.amax(maps[..., -3:, :, :], dim=-3).bool()
    else:
        drivable = np.amax(maps[..., -3:, :, :], axis=-3).astype(bool)
    return drivable

def trajdata2posyawspeed_acc(state, dt=0.1):
    # —— 1) 原始提取 —— #

    pos = state[...,:2]                   # [B, N, T, 2]
    s = state[..., -2]   # sin
    c = state[..., -1]   # cos
    yaw  = angle_wrap(torch.atan2(s,c))
    
    speed = state[...,2]*c + state[...,3]*s #(B,N,T)

    xddydd = state[...,4:6]

    heading_vec = torch.stack([c, s], dim=-1)

    acc_lon = (xddydd * heading_vec).sum(dim=-1)        # [B, N, T]
    

    # —— 2) 统一计算 mask —— #
    # mask的True代表该帧有效
    mask = torch.bitwise_not(torch.max(torch.isnan(state), dim=-1)[0])#(B,N,T)

    # —— 3) 计算 yaw_rate —— #
    # 使用纯PyTorch实现的unwrap函数
    yaw_unwrapped = unwrap_torch(yaw)
    yaw_pad = torch.cat([torch.zeros_like(yaw[..., :1]), yaw_unwrapped], dim=-1)  # [B, N, T+1]
    yaw_diff = yaw_pad[..., 1:] - yaw_pad[..., :-1]                     # [B, N, T]
    yaw_rate = yaw_diff / dt                                           # [B, N, T]


    # —— 4) 把无效帧都置零 —— #
    pos      = torch.where(mask.unsqueeze(-1), pos,      torch.zeros_like(pos))
    speed    = torch.where(mask,           speed,      torch.zeros_like(speed))
    yaw      = torch.where(mask,           yaw,        torch.zeros_like(yaw))
    acc_lon  = torch.where(mask,           acc_lon,    torch.zeros_like(acc_lon))
    yaw_rate = torch.where(mask,           yaw_rate,   torch.zeros_like(yaw_rate))

    return pos, speed, yaw, acc_lon, yaw_rate, mask

@torch.no_grad()
def parse_scene_centric(batch: SceneBatch):
    num_agents = batch.num_agents
    fut_pos, fut_speed, fut_yaw, fut_acc_lon, fut_yaw_rate, fut_mask = trajdata2posyawspeed_acc(batch.agent_fut)
    hist_pos, hist_speed,hist_yaw, hist_acc_lon, hist_yaw_rate, hist_mask = trajdata2posyawspeed_acc(batch.agent_hist)

    curr_pos = hist_pos[:,:,-1]
    curr_yaw = hist_yaw[:,:,-1]

    curr_speed = hist_speed[..., -1]
    centered_state = batch.centered_agent_state
    assert torch.all(centered_state[:, -1] == centered_state.heading[...,0])#Note:obs_format is x,y,xd,yd,xdd,ydd,h, .heading is h
    assert torch.all(centered_state[:, :2] == centered_state.position)
    centered_yaw = centered_state.heading[...,0]
    centered_pos = centered_state.position

    # convert nuscenes types to l5kit types
    agent_type = batch.agent_type
    agent_type[agent_type < 0] = 0
    agent_type[agent_type == 1] = 3
    # mask out invalid extents
    agent_hist_extent = batch.agent_hist_extent
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.


    centered_world_from_agent = torch.inverse(batch.centered_agent_from_world_tf)

    # map-related
    if batch.maps is not None:
        map_res = batch.maps_resolution[0,0]
        h, w = batch.maps.shape[-2:]
        # TODO: pass env configs to here
        
        centered_raster_from_agent = torch.Tensor([
            [map_res, 0, 0.25 * w],
            [0, map_res, 0.5 * h],
            [0, 0, 1]
        ]).to(centered_state.device)
        b,a = curr_yaw.shape[:2]
        centered_agent_from_raster,_ = torch.linalg.inv_ex(centered_raster_from_agent)
        
        agents_from_center = (GeoUtils.transform_matrices(-curr_yaw.flatten(),torch.zeros(b*a,2,device=curr_yaw.device))
                                @GeoUtils.transform_matrices(torch.zeros(b*a,device=curr_yaw.device),-curr_pos.reshape(-1,2))).reshape(*curr_yaw.shape[:2],3,3)
        center_from_agents = GeoUtils.transform_matrices(curr_yaw.flatten(),curr_pos.reshape(-1,2)).reshape(*curr_yaw.shape[:2],3,3)
        raster_from_center = centered_raster_from_agent @ agents_from_center
        center_from_raster = center_from_agents @ centered_agent_from_raster

        raster_from_world = batch.rasters_from_world_tf
        world_from_raster,_ = torch.linalg.inv_ex(raster_from_world)
        raster_from_world[torch.isnan(raster_from_world)] = 0.
        world_from_raster[torch.isnan(world_from_raster)] = 0.

 
        drivable_map = get_drivable_region_map(batch.maps)
    else:
        maps = None
        drivable_map = None
        raster_from_agent = None
        agent_from_raster = None
        raster_from_world = None

    extent_scale = 1.0


    d = dict(
        future_positions=fut_pos,
        future_speeds=fut_speed,
        future_yaws=fut_yaw,
        future_acc_lons=fut_acc_lon,
        future_yaw_rates=fut_yaw_rate,
        future_availabilities=fut_mask,

        history_positions=hist_pos,
        history_speeds=hist_speed,
        history_yaws=hist_yaw,
        history_acc_lons=hist_acc_lon,
        history_yaw_rates=hist_yaw_rate,
        history_availabilities=hist_mask,

        current_positions=curr_pos,
        current_speeds=curr_speed,
        current_yaws=curr_yaw,
        
        centroid=centered_pos,
        yaw=centered_yaw,
        type=agent_type,
        history_extent=agent_hist_extent * extent_scale,
        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        map_names=batch.map_names,
        drivable_map=drivable_map,
        raster_from_agent=centered_raster_from_agent,
        agent_from_raster=centered_agent_from_raster,
        raster_from_center=raster_from_center,
        center_from_raster=center_from_raster,
        agents_from_center = agents_from_center,
        center_from_agents = center_from_agents,
        raster_from_world=raster_from_world,
        agent_from_world=batch.centered_agent_from_world_tf,
        world_from_agent=centered_world_from_agent,
    )
    return d

@torch.no_grad()
def parse_node_centric(batch: dict):
    maybe_pad_neighbor(batch)
    center_fut_pos, center_fut_speed, center_fut_yaw, center_fut_acc_lon, center_fut_yaw_rate, center_fut_mask = trajdata2posyawspeed_acc(batch['agent_fut'],0.1)
    center_hist_pos, center_hist_speed, center_hist_yaw, center_hist_acc_lon, center_hist_yaw_rate, center_hist_mask = trajdata2posyawspeed_acc(batch['agent_hist'],0.1)
    
    neigh_fut_pos, neigh_fut_speed, neigh_fut_yaw, neigh_fut_acc_lon, neigh_fut_yaw_rate, neigh_fut_mask = trajdata2posyawspeed_acc(batch['neigh_fut'],0.1)
    neigh_hist_pos, neigh_hist_speed, neigh_hist_yaw, neigh_hist_acc_lon, neigh_hist_yaw_rate, neigh_hist_mask = trajdata2posyawspeed_acc(batch['neigh_hist'],0.1)
    
    # ego_fut_pos, ego_fut_speed, ego_fut_yaw, ego_fut_acc_lon, ego_fut_yaw_rate, ego_fut_mask = trajdata2posyawspeed_acc(batch.robot_fut,0.1)
    curr_state = batch["curr_agent_state"]
    curr_yaw = curr_state.heading[...,0]
    curr_pos = curr_state.position

    center_curr_pos = center_hist_pos[:, -1]
    center_curr_yaw = center_hist_yaw[:, -1]
    center_curr_speed = center_hist_speed[:, -1]

    neigh_curr_pos = neigh_hist_pos[:,:, -1]
    neigh_curr_yaw = neigh_hist_yaw[:,:, -1]
    neigh_curr_speed = neigh_hist_speed[:,:, -1]


    agent_hist_extent = batch['agent_hist_extent']
    agent_hist_extent[torch.isnan(agent_hist_extent)] = 0.

    # mask out invalid extents
    neigh_hist_extents = batch['neigh_hist_extents']
    neigh_hist_extents[torch.isnan(neigh_hist_extents)] = 0.

    world_from_agents = torch.inverse(batch['agents_from_world_tf'])
    raster_cfg = BATCH_RASTER_CFG
    map_res = 1.0 / raster_cfg["pixel_size"]
    h = w = raster_cfg["raster_size"]
    ego_cent = raster_cfg["ego_center"]
    raster_from_agent = torch.Tensor([
        [map_res, 0, ((1.0 + ego_cent[0])/2.0) * w],
        [0, map_res, ((1.0 + ego_cent[1])/2.0) * h],
        [0, 0, 1]
    ]).to(center_fut_pos.device)
    
    bsize = batch['agents_from_world_tf'].shape[0]
    agent_from_raster = torch.inverse(raster_from_agent)
    raster_from_agent = TensorUtils.unsqueeze_expand_at(raster_from_agent, size=bsize, dim=0)
    agent_from_raster = TensorUtils.unsqueeze_expand_at(agent_from_raster, size=bsize, dim=0)
    raster_from_world = torch.bmm(raster_from_agent, batch['agents_from_world_tf'])



    drivable_map = None
    if batch['maps'] is not None:
        drivable_map = get_drivable_region_map(batch['maps'])

    extent_scale = 1.0
    d = dict(
        maps=batch['maps'],
        map_names=batch['map_names'],
        drivable_map=drivable_map,
        
        center_fut_positions=center_fut_pos,
        center_fut_speeds=center_fut_speed,
        center_fut_yaws=center_fut_yaw,
        center_fut_acc_lons=center_fut_acc_lon,
        center_fut_yaw_rates=center_fut_yaw_rate,
        center_fut_availabilities=center_fut_mask,

        center_hist_positions=center_hist_pos,
        center_hist_speeds=center_hist_speed,
        center_hist_yaws=center_hist_yaw,
        center_hist_acc_lons=center_hist_acc_lon,
        center_hist_yaw_rates=center_hist_yaw_rate,
        center_hist_availabilities=center_hist_mask,

        center_curr_positions=center_curr_pos,
        center_curr_speeds=center_curr_speed,
        center_curr_yaws=center_curr_yaw,
        
        neigh_curr_positions=neigh_curr_pos,
        neigh_curr_speeds=neigh_curr_speed,
        neigh_curr_yaws=neigh_curr_yaw,
        
        neigh_fut_positions=neigh_fut_pos,
        neigh_fut_speeds=neigh_fut_speed,
        neigh_fut_yaws=neigh_fut_yaw,
        neigh_fut_acc_lons=neigh_fut_acc_lon,
        neigh_fut_yaw_rates=neigh_fut_yaw_rate,
        neigh_fut_availabilities=neigh_fut_mask,

        neigh_hist_positions=neigh_hist_pos,
        neigh_hist_speeds=neigh_hist_speed,
        neigh_hist_yaws=neigh_hist_yaw,
        neigh_hist_acc_lons=neigh_hist_acc_lon,
        neigh_hist_yaw_rates=neigh_hist_yaw_rate,
        neigh_hist_availabilities=neigh_hist_mask,

        # ego_fut_positions=ego_fut_pos,
        # ego_fut_speeds=ego_fut_speed,
        # ego_fut_yaws=ego_fut_yaw,
        # ego_fut_acc_lons=ego_fut_acc_lon,
        # ego_fut_yaw_rates=ego_fut_yaw_rate,
        # ego_fut_availabilities=ego_fut_mask,
        centroid=curr_pos,
        yaw=curr_yaw,
        curr_agent_state = curr_state,

        agent_fut = batch['agent_fut'],

        extent=agent_hist_extent.max(dim=-2)[0] * extent_scale,
        neigh_extent=neigh_hist_extents.max(dim=-2)[0] * extent_scale,
        
        raster_from_center=raster_from_agent,
        center_from_raster=agent_from_raster,
        
        raster_from_world=raster_from_world,
        
        center_from_world=batch['agents_from_world_tf'],
        world_from_agent=world_from_agents,

    )
    return d
def maybe_pad_neighbor(batch):
    """Pad neighboring agent's history to the same length as that of the ego using NaNs"""
    hist_len = batch['agent_hist'].shape[1]
    fut_len = batch['agent_fut'].shape[1]
    b, a, neigh_len, _ = batch['neigh_hist'].shape
    device = batch['neigh_hist'].device
    empty_neighbor = a == 0
    device = batch['neigh_hist'].device
    if empty_neighbor:
        batch['neigh_hist'] = torch.ones(b, 1, hist_len, batch['neigh_hist'].shape[-1]).to(device) * torch.nan
        batch['neigh_fut'] = torch.ones(b, 1, fut_len, batch['neigh_fut'].shape[-1]).to(device) * torch.nan
        batch['neigh_types'] = torch.zeros(b, 1).to(device)
        batch['neigh_hist_extents'] = torch.zeros(b, 1, hist_len, batch['neigh_hist_extents'].shape[-1]).to(device)
        batch['neigh_fut_extents'] = torch.zeros(b, 1, fut_len, batch['neigh_hist_extents'].shape[-1]).to(device)
    elif neigh_len < hist_len:
        hist_pad = torch.ones(b, a, hist_len - neigh_len, batch['neigh_hist'].shape[-1], device=device).to(device) * torch.nan
        batch['neigh_hist'] = torch.cat((hist_pad, batch['neigh_hist']), dim=-2)
        hist_pad = torch.zeros(b, a, hist_len - neigh_len, batch['neigh_hist_extents'].shape[-1], device=device).to(device)
        batch['neigh_hist_extents'] = torch.cat((hist_pad, batch['neigh_hist_extents']), dim=-2)
    


def transform_fn_filterout_stationary_agents_Scene(batch_elem: SceneBatchElement) -> SceneBatchElement:
    """过滤掉静止不动的代理"""
    # 获取所有agent的元数据
    agent_meta_dicts = batch_elem.agent_meta_dicts
    
    # 找出非静止代理的索引
    moving_indices = [i for i, meta in enumerate(agent_meta_dicts) if not meta['is_stationary']]#TODO:加入判断，保留ego agent
    
    # 如果没有需要过滤的代理，直接返回
    if len(moving_indices) == len(agent_meta_dicts):
        return batch_elem
    
    # 更新agent数量
    batch_elem.num_agents = len(moving_indices)
    
    # 过滤列表类型数据
    batch_elem.agent_meta_dicts = [batch_elem.agent_meta_dicts[i] for i in moving_indices]
    batch_elem.agent_names = [batch_elem.agent_names[i] for i in moving_indices]
    batch_elem.agent_histories = [batch_elem.agent_histories[i] for i in moving_indices]
    batch_elem.agent_history_extents = [batch_elem.agent_history_extents[i] for i in moving_indices]
    batch_elem.agent_futures = [batch_elem.agent_futures[i] for i in moving_indices]
    batch_elem.agent_future_extents = [batch_elem.agent_future_extents[i] for i in moving_indices]
    
    # 过滤numpy数组
    batch_elem.agent_types_np = batch_elem.agent_types_np[moving_indices]
    batch_elem.agent_history_lens_np = batch_elem.agent_history_lens_np[moving_indices]
    batch_elem.agent_future_lens_np = batch_elem.agent_future_lens_np[moving_indices]
    
    return batch_elem

def transform_fn_filterout_stationary_neighbor_Agent(batch_elem: AgentBatchElement) -> AgentBatchElement:
    """过滤掉静止不动的代理"""
    # 获取所有agent的元数据
    neighbor_meta_dicts = batch_elem.neighbor_meta_dicts
    
    # 找出非静止代理的索引
    moving_indices = [i for i, meta in enumerate(neighbor_meta_dicts) if not meta['is_stationary']]
    
    # 如果没有需要过滤的代理，直接返回
    if len(moving_indices) == len(neighbor_meta_dicts):
        return batch_elem
    
    # 更新agent数量
    batch_elem.num_neighbors = len(moving_indices)
    
    # 过滤列表类型数据
    batch_elem.neighbor_meta_dicts = [batch_elem.neighbor_meta_dicts[i] for i in moving_indices]
    batch_elem.neighbor_histories = [batch_elem.neighbor_histories[i] for i in moving_indices]
    batch_elem.neighbor_history_extents = [batch_elem.neighbor_history_extents[i] for i in moving_indices]
    batch_elem.neighbor_futures = [batch_elem.neighbor_futures[i] for i in moving_indices]
    batch_elem.neighbor_future_extents = [batch_elem.neighbor_future_extents[i] for i in moving_indices]
    
    # 过滤numpy数组
    batch_elem.neighbor_types_np = batch_elem.neighbor_types_np[moving_indices]
    batch_elem.neighbor_history_lens_np = batch_elem.neighbor_history_lens_np[moving_indices]
    batch_elem.neighbor_future_lens_np = batch_elem.neighbor_future_lens_np[moving_indices]
    
    return batch_elem

def dataset_filter_only_moving_central_agent(batch_elem: AgentBatchElement) -> bool:
    return not batch_elem.agent_meta_dict['is_stationary'] 

def unwrap_torch(angles, dim=-1):

    pi = torch.tensor(np.pi, device=angles.device, dtype=angles.dtype)
    
    # 创建一个移位版本的angles用于计算差值
    # 我们需要沿着指定维度移位
    shape = list(angles.shape)
    if dim < 0:
        dim = len(shape) + dim
    
    # 如果dim维度长度为1或0，直接返回原始角度
    if shape[dim] <= 1:
        return angles.clone()
    
    # 计算相邻元素差值
    diff = torch.zeros_like(angles)
    
    # 创建索引来获取除了第一个元素外的所有元素
    idx_first = [slice(None)] * len(shape)
    idx_first[dim] = slice(1, None)
    
    # 创建索引来获取除了最后一个元素外的所有元素
    idx_last = [slice(None)] * len(shape)
    idx_last[dim] = slice(0, -1)
    
    # 计算差值
    diff[tuple(idx_first)] = angles[tuple(idx_first)] - angles[tuple(idx_last)]
    
    # 找出差值大于pi的点
    jumps = torch.zeros_like(diff)
    jumps[diff < -pi] = 2 * pi
    jumps[diff > pi] = -2 * pi
    
    # 累积跳变
    result = angles.clone()
    cumsum_jumps = torch.cumsum(jumps, dim=dim)
    result = result + cumsum_jumps
    
    return result





@torch.no_grad()
def parse_batch(data_batch):
        if hasattr(data_batch, "num_agents"):
            parsed_batch =  parse_scene_centric(data_batch)
        else:
            parsed_batch =  parse_node_centric(data_batch)
            
        # 添加 extras 到 parsed_batch
        if 'extras' in data_batch:
            parsed_batch['extras'] = data_batch['extras']
            
        return parsed_batch
        # for k in list(vars(data_batch).keys()):
        #     if k not in parsed_batch:
        #         delattr(data_batch, k)
        # for k, v in parsed_batch.items():
        #     setattr(data_batch, k, v)
        # for k in parsed_batch:
        #     val = getattr(data_batch, k)
        #     if isinstance(val, torch.Tensor):
        #         setattr(data_batch, k, val.nan_to_num(0.0))
        # return data_batch
        
    
