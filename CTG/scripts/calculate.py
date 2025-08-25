from tbsim.datasets.cld_datamodules import FilteredUnifiedDataset,create_dataset
from tbsim.utils.safety_critical_batch_utils import trajdata2posyawspeed_acc
import os
import json
import math
from typing import Dict, Any, Tuple, Optional
from trajdata.data_structures import  AgentMetadata, AgentType
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union






def main():
    KWARGS=dict(
        cache_location='~/cld_cache', 
        desired_data=['nusc_trainval-train', 'nusc_trainval-train_val'], 
        data_dirs={'nusc_trainval': '../nuscenes', 'nusc_test': '../nuscenes', 'nusc_mini': '../nuscenes'}, 
        desired_dt=0.1, 
        only_types=[AgentType.VEHICLE], 
        only_predict=[AgentType.VEHICLE], 
        future_sec=(5.2, 5.2), 
        history_sec=(3.0, 3.0), 
        incl_robot_future=True, 
        ego_only=False, 
        agent_interaction_distances=defaultdict(lambda: 30), 
        incl_raster_map=True, 
        raster_map_params={
            'px_per_m': 1, 
            'map_size_px': 224, 
            'return_rgb': True, 
            'offset_frac_xy':[-0.5, 0.0], 
            'no_map_fill_value':-1.0
            }, 
        centric='agent', 
        standardize_data=True, 
        verbose=True, 
        max_agent_num=None, 
        num_workers=24, 
        rebuild_cache=False, 
        rebuild_maps=False)

    # 初始化统计变量
    dt = KWARGS["desired_dt"]
    
    # Center agent 统计变量
    c_sum_x = c_sum_y = c_sum_spd = c_sum_acc = c_sum_yawr = 0.0
    c_sumsq_x = c_sumsq_y = c_sumsq_spd = c_sumsq_acc = c_sumsq_yawr = 0.0
    c_n_x = c_n_y = c_n_spd = c_n_acc = c_n_yawr = 0
    c_sum_sin = c_sum_cos = c_n_yaw = 0.0
    
    # Neighbor agents 统计变量
    n_sum_x = n_sum_y = n_sum_spd = n_sum_acc = n_sum_yawr = 0.0
    n_sumsq_x = n_sumsq_y = n_sumsq_spd = n_sumsq_acc = n_sumsq_yawr = 0.0
    n_n_x = n_n_y = n_n_spd = n_n_acc = n_n_yawr = 0
    n_sum_sin = n_sum_cos = n_n_yaw = 0.0
    
    # 新增：Extent 统计变量
    c_sum_extent_l = c_sum_extent_w = 0.0
    c_sumsq_extent_l = c_sumsq_extent_w = 0.0
    c_n_extent = 0
    
    n_sum_extent_l = n_sum_extent_w = 0.0
    n_sumsq_extent_l = n_sumsq_extent_w = 0.0
    n_n_extent = 0

    dataset = FilteredUnifiedDataset(**KWARGS)
    
    for i in tqdm(range(len(dataset))):
        elem = dataset[i]

        # Center agent 处理
        cen_np = np.concatenate([elem.agent_history_np, elem.agent_future_np], axis=0)  # [T,F]
        center_state = torch.from_numpy(cen_np).float().unsqueeze(0).unsqueeze(0)      # [1,1,T,F]

        pos, spd, yaw, acc, yawr, mask = trajdata2posyawspeed_acc(center_state)
        pos = pos[0, 0].cpu().numpy()       # [T,2]
        spd = spd[0, 0].cpu().numpy()       # [T]
        yaw = yaw[0, 0].cpu().numpy()       # [T]
        acc = acc[0, 0].cpu().numpy()       # [T]
        yawr = yawr[0, 0].cpu().numpy()     # [T]
        m = mask[0, 0].cpu().numpy().astype(bool)  # [T]

        if m.any():  # 确保有有效数据
            px = pos[m, 0]; py = pos[m, 1]
            s  = spd[m];    ya = yaw[m];   ac = acc[m];   yr = yawr[m]

            c_sum_x += float(px.sum()); c_sumsq_x += float((px*px).sum()); c_n_x += px.size
            c_sum_y += float(py.sum()); c_sumsq_y += float((py*py).sum()); c_n_y += py.size
            c_sum_spd += float(s.sum()); c_sumsq_spd += float((s*s).sum()); c_n_spd += s.size
            c_sum_acc += float(ac.sum()); c_sumsq_acc += float((ac*ac).sum()); c_n_acc += ac.size
            c_sum_yawr += float(yr.sum()); c_sumsq_yawr += float((yr*yr).sum()); c_n_yawr += yr.size

            c_sum_sin += float(np.sin(ya).sum())
            c_sum_cos += float(np.cos(ya).sum())
            c_n_yaw   += ya.size

        # 新增：Center agent extent 处理
        if hasattr(elem, 'agent_history_extent_np') and elem.agent_history_extent_np is not None:
            center_extent = elem.agent_history_extent_np  # 假设这是 [T, 2] 或 [2] 形状
            if center_extent.ndim == 2:  # [T, 2]
                center_extent = center_extent.mean(axis=0)  # 取时间平均，得到 [2]
            elif center_extent.ndim == 1:  # [2]
                center_extent = center_extent
            else:
                continue
                
            if len(center_extent) >= 2:
                extent_l = center_extent[0]  # 长度
                extent_w = center_extent[1]  # 宽度
                
                c_sum_extent_l += float(extent_l); c_sumsq_extent_l += float(extent_l * extent_l)
                c_sum_extent_w += float(extent_w); c_sumsq_extent_w += float(extent_w * extent_w)
                c_n_extent += 1

        # Neighbor agents 处理
        nh = elem.neighbor_histories
        nf = elem.neighbor_futures

        for h_np, f_np in zip(nh, nf):
            nb_np = np.concatenate([h_np, f_np], axis=0)  # [T,F]
            nb_state = torch.from_numpy(nb_np).float().unsqueeze(0).unsqueeze(0)  # [1,1,T,F]
            pos, spd, yaw, acc, yawr, mask = trajdata2posyawspeed_acc(nb_state, dt=dt)
            pos = pos[0, 0].cpu().numpy()
            spd = spd[0, 0].cpu().numpy()
            yaw = yaw[0, 0].cpu().numpy()
            acc = acc[0, 0].cpu().numpy()
            yawr = yawr[0, 0].cpu().numpy()
            m = mask[0, 0].cpu().numpy().astype(bool)

            if m.any():  # 确保有有效数据
                px = pos[m, 0]; py = pos[m, 1]
                s  = spd[m];    ya = yaw[m];   ac = acc[m];   yr = yawr[m]

                n_sum_x += float(px.sum()); n_sumsq_x += float((px*px).sum()); n_n_x += px.size
                n_sum_y += float(py.sum()); n_sumsq_y += float((py*py).sum()); n_n_y += py.size
                n_sum_spd += float(s.sum()); n_sumsq_spd += float((s*s).sum()); n_n_spd += s.size
                n_sum_acc += float(ac.sum()); n_sumsq_acc += float((ac*ac).sum()); n_n_acc += ac.size
                n_sum_yawr += float(yr.sum()); n_sumsq_yawr += float((yr*yr).sum()); n_n_yawr += yr.size

                n_sum_sin += float(np.sin(ya).sum())
                n_sum_cos += float(np.cos(ya).sum())
                n_n_yaw   += ya.size

        # 新增：Neighbor agents extent 处理
        if hasattr(elem, 'neighbor_history_extents') and elem.neighbor_history_extents is not None:
            neighbor_extents = elem.neighbor_history_extents  # 假设这是 [N, T, 2] 或 [N, 2] 形状
            for neighbor_extent in neighbor_extents:
                if neighbor_extent.ndim == 2:  # [T, 2]
                    neighbor_extent = neighbor_extent.mean(axis=0)  # 取时间平均，得到 [2]
                elif neighbor_extent.ndim == 1:  # [2]
                    neighbor_extent = neighbor_extent
                else:
                    continue
                    
                if len(neighbor_extent) >= 2:
                    extent_l = neighbor_extent[0]  # 长度
                    extent_w = neighbor_extent[1]  # 宽度
                    
                    n_sum_extent_l += float(extent_l); n_sumsq_extent_l += float(extent_l * extent_l)
                    n_sum_extent_w += float(extent_w); n_sumsq_extent_w += float(extent_w * extent_w)
                    n_n_extent += 1

    # 循环结束后计算统计结果
    def _ms(sum_, sqsum_, n_):
        if n_ == 0:
            return 0.0, 1.0
        mean = sum_ / n_
        var  = max(sqsum_ / n_ - mean * mean, 1e-12)
        return float(mean), float(math.sqrt(var))

    # Center agent 统计
    cx, sx = _ms(c_sum_x, c_sumsq_x, c_n_x)
    cy, sy = _ms(c_sum_y, c_sumsq_y, c_n_y)
    cs, ss = _ms(c_sum_spd, c_sumsq_spd, c_n_spd)
    ca, sa = _ms(c_sum_acc, c_sumsq_acc, c_n_acc)
    cyr, syr = _ms(c_sum_yawr, c_sumsq_yawr, c_n_yawr)
    
    # yaw（圆统计）
    if c_n_yaw > 0:
        c_avg_sin = c_sum_sin / c_n_yaw
        c_avg_cos = c_sum_cos / c_n_yaw
        cya = float(math.atan2(c_avg_sin, c_avg_cos))
        R = max(min(math.hypot(c_avg_cos, c_avg_sin), 0.999999), 1e-8)
        sya = float(math.sqrt(-2.0 * math.log(R)))
    else:
        cya, sya = 0.0, 1.0

    # 新增：Center agent extent 统计
    c_extent_l_mean, c_extent_l_std = _ms(c_sum_extent_l, c_sumsq_extent_l, c_n_extent)
    c_extent_w_mean, c_extent_w_std = _ms(c_sum_extent_w, c_sumsq_extent_w, c_n_extent)

    result = {
        "center": {
            "add": [cx, cy, cs, cya, ca, cyr, c_extent_l_mean, c_extent_w_mean],           # [x, y, speed, yaw, acc_lon, yaw_rate, length, width]
            "div": [sx, sy, ss, sya, sa, syr, c_extent_l_std, c_extent_w_std],
        }
    }

    # Neighbor agents 统计
    nx, nsx = _ms(n_sum_x, n_sumsq_x, n_n_x)
    ny, nsy = _ms(n_sum_y, n_sumsq_y, n_n_y)
    ns, nss = _ms(n_sum_spd, n_sumsq_spd, n_n_spd)
    na, nsa = _ms(n_sum_acc, n_sumsq_acc, n_n_acc)
    nyr, nsyr = _ms(n_sum_yawr, n_sumsq_yawr, n_n_yawr)
    
    if n_n_yaw > 0:
        n_avg_sin = n_sum_sin / n_n_yaw
        n_avg_cos = n_sum_cos / n_n_yaw
        nya = float(math.atan2(n_avg_sin, n_avg_cos))
        R = max(min(math.hypot(n_avg_cos, n_avg_sin), 0.999999), 1e-8)
        nsya = float(math.sqrt(-2.0 * math.log(R)))
    else:
        nya, nsya = 0.0, 1.0
        
    # 新增：Neighbor agents extent 统计
    n_extent_l_mean, n_extent_l_std = _ms(n_sum_extent_l, n_sumsq_extent_l, n_n_extent)
    n_extent_w_mean, n_extent_w_std = _ms(n_sum_extent_w, n_sumsq_extent_w, n_n_extent)
        
    result["neighbor"] = {
        "add": [nx, ny, ns, nya, na, nyr, n_extent_l_mean, n_extent_w_mean],
        "div": [nsx, nsy, nss, nsya, nsa, nsyr, n_extent_l_std, n_extent_w_std],
    }

    # --------- 打印并保存 ---------
    print("===== 归一化参数（顺序: [x, y, speed, yaw, acc_lon, yaw_rate, length, width]）=====")
    for k in result:
        print(k, "add =", [round(v, 4) for v in result[k]["add"]])
        print(k, "div =", [round(v, 4) for v in result[k]["div"]])

    with open("norm_stats.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to norm_stats.json")



if __name__ == "__main__":
    main()