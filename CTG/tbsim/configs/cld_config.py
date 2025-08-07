import math
import numpy as np
from tbsim.configs.base import AlgoConfig
from tbsim.configs.trajdata_config import TrajdataTrainConfig, TrajdataEnvConfig


class CLDTrainConfig(TrajdataTrainConfig):
    def __init__(self):
        super(CLDTrainConfig, self).__init__()

        self.trajdata_cache_location = "~/cld_cache"
        self.trajdata_source_train = ["nusc_trainval-train", "nusc_trainval-train_val"]
        self.trajdata_source_valid = ["nusc_trainval-val"]
        # dict mapping dataset IDs -> root path
        #       all datasets that will be used must be included here
        self.trajdata_data_dirs = {
            "nusc_trainval" : "../nuscenes",
            "nusc_test" : "../nuscenes",
            "nusc_mini" : "../nuscenes",
        }

        # for debug
        self.trajdata_rebuild_cache = False

        self.rollout.enabled = True
        self.rollout.save_video = True
        self.rollout.every_n_steps = 10000
        self.rollout.warm_start_n_steps = 0

        # training config
        # assuming 1 sec (10 steps) past, 2 sec (20 steps) future
        self.training.batch_size = 64 # 4 # 100
        self.training.num_steps = 100
        self.training.num_data_workers = 0

        self.save.every_n_steps = 10
        self.save.best_k = 1

        # validation config
        self.validation.enabled = True
        self.validation.batch_size = 32 # 4 # 32
        self.validation.num_data_workers = 6
        self.validation.every_n_steps = 500
        self.validation.num_steps_per_epoch = 5 # 50

        self.logging.terminal_output_to_txt = True  # whether to log stdout to txt file
        self.logging.log_wandb = True  # enable wandb logging
        self.logging.wandb_project_name = "tbsim"
        self.logging.log_every_n_steps = 10
        self.logging.flush_every_n_steps = 100

class CLDEnvConfig(TrajdataEnvConfig):
    def __init__(self):
        super(CLDEnvConfig, self).__init__()

        self.data_generation_params.trajdata_centric = "agent" # "agent", "scene"
        # which types of agents to include from ['unknown', 'vehicle', 'pedestrian', 'bicycle', 'motorcycle']
        self.data_generation_params.trajdata_only_types = ["vehicle"]
        # which types of agents to predict
        self.data_generation_params.trajdata_predict_types = ["vehicle"]
        self.data_generation_params.trajdata_scene_desc_contains = None
        self.data_generation_params.trajdata_incl_map = True
        self.data_generation_params.trajdata_max_agents_distance = 50
        self.data_generation_params.trajdata_standardize_data = True

    
   
        # raster image size [pixels]
        self.rasterizer.raster_size = 224
        # raster's spatial resolution [meters per pixel]: the size in the real world one pixel corresponds to.
        self.rasterizer.pixel_size = 1.0 / 1.0 # 1 px/m
        self.rasterizer.num_sem_layers = 3 
        # how to group layers together to viz RGB image
        self.rasterizer.rgb_idx_groups = ([0], [1], [2])
        # where the agent is on the map, (0.0, 0.0) is the center
        self.rasterizer.ego_center = (-0.5, 0.0)
        self.data_generation_params.other_agents_num = None



class CLDAlgoConfig(AlgoConfig):
    def __init__(self):
        super(CLDAlgoConfig, self).__init__()


        self.name = "diffuser_ppo"

        self.scene_agent_max_neighbor_dist = 30 # used only when data_centric == 'scene' and coordinate == 'agent'
        self.map_encoder_model_arch = "resnet18"
        self.map_feature_dim = 128
        self.map_grid_feature_dim = 32

        self.history_num_frames = 30
        self.history_feature_dim = 64

        self.state_in_dim = 4
        self.state_out_feat_dim = 64

        self.cond_feat_dim = 128
        self.time_emb_dim = 32

        self.attn_hidden_dim = 64
        self.attn_dilations = (1,2,4)
        self.attn_n_heads = 4
        self.grid_map_traj_dim = 32
        self.mix_gauss = 5


        self.horizon = 50 # param to control the number of time steps to use for future prediction
        self.n_diffusion_steps = 100
        self.ddim_steps = 20
    
        self.future_num_frames = self.horizon
        self.step_time = 0.1
  
        self.dynamics.type = "Unicycle"
        self.dynamics.max_steer = 0.5
        self.dynamics.max_yawvel = math.pi * 2.0
        self.dynamics.acce_bound = (-10, 8)
        self.dynamics.ddh_bound = (-math.pi * 2.0, math.pi * 2.0)
        self.dynamics.max_speed = 40.0  # roughly 90mph


        self.optim_params.learning_rate.initial = 0.0002
        self.optim_params.learning_rate.weight_decay = 1e-4


        self.norm_info = {
            'diffuser': [( 2.135494, 0.003704, 0.970226, 0.000573, -0.002965, 0.000309,  ), ( 5.544400, 0.524067, 2.206522, 0.049049, 0.729327, 0.023765,  )],
            'agent_hist': [( -1.198923, 0.000128, 0.953161, 4.698113, 2.051664,  ), ( 3.180241, 0.159182, 2.129779, 2.116855, 0.388149,  )],
            'neighbor_hist': [( -0.237441, 1.118636, 0.489575, 0.868664, 0.222984,  ), ( 7.587311, 7.444489, 1.680952, 2.578202, 0.832563,  )],
        }


