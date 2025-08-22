import numpy as np
from pathlib import Path
from collections import defaultdict
from functools import partial
from trajdata.dataset import UnifiedDataset
from trajdata.caching import EnvCache, SceneCache
from trajdata.data_structures import  AgentMetadata, AgentType
from trajdata import filtering
from trajdata.utils import scene_utils
from tqdm import tqdm
from trajdata import filtering
from trajdata.utils.parallel_utils import parallel_iapply
from typing import Dict, List, Optional, Set, Tuple, Union
import copy
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from tbsim.configs.base import TrainConfig
from tbsim.utils.trajdata_utils import TRAJDATA_AGENT_TYPE_MAP
import os
import gc
from tbsim.utils.trajdata_utils import TRAJDATA_AGENT_TYPE_MAP, get_closest_lane_point_wrapper, get_full_fut_traj, get_full_fut_valid

def create_dataset(config: Dict, split: Optional[Union[str, List[str]]] = None) -> UnifiedDataset:
    
    dataset_config = copy.deepcopy(config)
    desired_data = [f"{dataset_config.name}-{s}" for s in split]
    data_dirs = {dataset_config.name: dataset_config.data_root}

    dataset = FilteredUnifiedDataset(
        desired_data=desired_data,
        data_dirs=data_dirs,
        desired_dt=dataset_config.desired_dt,
        history_sec=(dataset_config.history_sec, dataset_config.history_sec),
        future_sec=(dataset_config.future_sec, dataset_config.future_sec),
        agent_interaction_distances=defaultdict(lambda: 50.0),
        only_types=[AgentType.VEHICLE],    # 只考虑车辆
        centric="agent",
        incl_robot_future=dataset_config.incl_robot_future,  #True代表agent-centric场景central agent排除ego agent
        incl_raster_map=True,         
        cache_location=dataset_config.cache_location,
        rebuild_cache=dataset_config.rebuild_cache,
        raster_map_params=dataset_config.raster_map_params,
        standardize_data=dataset_config.standardize_data,
        # max_neighbor_num=dataset_config.max_neighbor_num,#for agent-centric batching
        # max_agent_num=dataset_config.max_agent_num,#for scene-centric batching
        ego_only=not dataset_config.incl_robot_future, #ego only=True 就代表agent-centric场景central agent一直是ego agent
        verbose=True,
        save_index=False,
        
    )
    return dataset

class CLDDataModule(pl.LightningDataModule):
    
    def __init__(self, data_config, train_config: TrainConfig):
        super().__init__()
        self._data_config = data_config
        self._train_config = train_config
        self.train_dataset = None
        self.valid_dataset = None

    @property
    def modality_shapes(self):
          return dict(
            image=(self._data_config.num_sem_layers,  # semantic map
                   self._data_config.raster_size,
                   self._data_config.raster_size)
        )

    def setup(self, stage = None):
        data_cfg = self._data_config
        future_sec = data_cfg.future_num_frames * data_cfg.step_time
        history_sec = data_cfg.history_num_frames * data_cfg.step_time
        neighbor_distance = data_cfg.max_agents_distance
        agent_only_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_only_types]
        agent_predict_types = None
        print("data_cfg.trajdata_predict_types", data_cfg.trajdata_predict_types)
        if data_cfg.trajdata_predict_types is not None:
            if data_cfg.other_agents_num is None:
                max_agent_num = None
            else:
                max_agent_num = 1+data_cfg.other_agents_num

            agent_predict_types = [TRAJDATA_AGENT_TYPE_MAP[cur_type] for cur_type in data_cfg.trajdata_predict_types]
        kwargs = dict(
            cache_location=data_cfg.trajdata_cache_location,
            desired_data=data_cfg.trajdata_source_train,
            data_dirs=data_cfg.trajdata_data_dirs,
            desired_dt=data_cfg.step_time,
            only_types=agent_only_types,
            only_predict=agent_predict_types,

            future_sec=(future_sec, future_sec),
            history_sec=(history_sec, history_sec),
            incl_robot_future=   data_cfg.incl_robot_future,
            ego_only         =not data_cfg.incl_robot_future,

            agent_interaction_distances=defaultdict(lambda: neighbor_distance),
            incl_raster_map=data_cfg.trajdata_incl_map,
            raster_map_params={
                "px_per_m": int(1 / data_cfg.pixel_size),
                "map_size_px": data_cfg.raster_size,
                "return_rgb": True,
                "offset_frac_xy": data_cfg.raster_center,
                "no_map_fill_value": data_cfg.no_map_fill_value,
            },
            centric=data_cfg.trajdata_centric,

            standardize_data=data_cfg.trajdata_standardize_data,
            verbose=True,
            max_agent_num = max_agent_num,
            num_workers=os.cpu_count(),
            rebuild_cache=data_cfg.trajdata_rebuild_cache,
            rebuild_maps=data_cfg.trajdata_rebuild_cache,
         


        )
        print(kwargs)
        self.train_dataset = UnifiedDataset(**kwargs)

        kwargs["desired_data"] = data_cfg.trajdata_source_valid
        self.valid_dataset = UnifiedDataset(**kwargs)

        gc.collect()

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self._train_config.training.batch_size,
            num_workers= os.cpu_count(), #self._train_config.training.num_data_workers,
            drop_last=True,
            collate_fn=self.train_dataset.get_collate_fn(return_dict=True),
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            shuffle=False, # since pytorch lightning only evals a subset of val on each epoch, shuffle
            batch_size=self._train_config.validation.batch_size,
            num_workers= os.cpu_count(), #self._train_config.validation.num_data_workers,
            drop_last=True,
            collate_fn=self.valid_dataset.get_collate_fn(return_dict=True),
            persistent_workers=True
        )


class FilteredUnifiedDataset(UnifiedDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def get_data_index(
        self, num_workers: int, scene_paths: List[Path]
    ):
        """几乎照搬父类，但 data_index_fn 指向我们自己的静止过滤版本"""
        desc = f"Creating {self.centric.capitalize()} Data Index"
        if self.centric != "agent":
            # 这里只示例 agent-centric，scene-centric 保持不变
            return super()._get_data_index(num_workers, scene_paths)

        
        data_index_fn = partial(
            FilteredUnifiedDataset._get_data_index_agent,
            cache_class=self.cache_class,
            cache_path=self.cache_path,
            incl_robot_future=self.incl_robot_future,
            only_types=self.only_types,
            only_predict=self.only_predict,
            no_types=self.no_types,
            history_sec=self.history_sec,
            future_sec=self.future_sec,
            desired_dt=self.desired_dt,
            ego_only=self.ego_only,
        )

        data_index: Union[
            List[Tuple[str, int, np.ndarray]],
            List[Tuple[str, int, List[Tuple[str, np.ndarray]]]],
        ] = list()
        if num_workers <= 1:
            for scene_info_path in tqdm(
                scene_paths,
                desc=desc + " (Serially)",
                disable=not self.verbose,
            ):
                _, orig_path, index_elems_len, index_elems = data_index_fn(
                    scene_info_path
                )
                if len(index_elems) > 0:
                    data_index.append((str(orig_path), index_elems_len, index_elems))
        else:
            for _, orig_path, index_elems_len, index_elems in parallel_iapply(
                data_index_fn,
                scene_paths,
                num_workers=num_workers,
                desc=desc + f" ({num_workers} CPUs)",
                disable=not self.verbose,
            ):
                if len(index_elems) > 0:
                    data_index.append((str(orig_path), index_elems_len, index_elems))

        return data_index

    @staticmethod
    def _get_data_index_agent(
        scene_info_path: Path,
        cache_class,
        cache_path: Path,
        incl_robot_future: bool,
        only_types: Optional[Set],
        only_predict: Optional[Set],
        no_types: Optional[Set],
        history_sec: Tuple[Optional[float],Optional[float]],
        future_sec: Tuple[Optional[float],Optional[float]],
        desired_dt: Optional[float],
        ego_only: bool,
        ret_scene_info: bool = False,
    ):


        index_elems_len: int = 0
        index_elems: List[Tuple[str, np.ndarray]] = list()
        scene = EnvCache.load(scene_info_path)
        scene_utils.enforce_desired_dt(scene, desired_dt)

        # 2) 拿到和 __getitem__ 一致的 SceneCache
        cache = cache_class(cache_path, scene, None)

        # 3) 过滤 agent types
        filtered_agents = filtering.agent_types(
            scene.agents,
            no_types,
            only_predict if only_predict is not None else only_types,
        )

        future_frames = int(round(future_sec[1] / desired_dt))
        for agent_info in filtered_agents:
            # 如果要给 ego 也加 future，就跳过 ego
            if incl_robot_future and agent_info.name == "ego":
                continue
            if ego_only and agent_info.name != "ego":
                continue

            # —— 新增：静止判断 —— #
            # 拿原始 first/last timestep 的状态           
            start_ts, end_ts = filtering.get_valid_ts(
                agent_info, scene.dt, history_sec, future_sec
            )
            if start_ts > end_ts:
                continue

            # 2) 在这个窗口的首尾帧做静止判断
            for t in range(start_ts, end_ts + 1):
                if is_timestep_stationary(cache,agent_info,t,future_frames):
                    # 静止就跳过，不加入索引
                    continue

                # 对每个 t，都构造一个 [t, t] 的索引
                index_elems_len += 1
                index_elems.append((agent_info.name, np.array((t, t), dtype=int)))
        return (
            (scene if ret_scene_info else None),
            scene_info_path,
            index_elems_len,
            index_elems,
        )
    
def is_timestep_stationary(cache: SceneCache, agent_info: AgentMetadata,t:int,future_frames:int,stationary_thresh:float=4.0) -> bool:
    p0 = cache.get_state(agent_info.name, t)
    p1 = cache.get_state(agent_info.name, t + future_frames)
    dist = np.linalg.norm(p1.position - p0.position)
    return dist < stationary_thresh



