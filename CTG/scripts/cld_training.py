import argparse
import sys
import os
import socket

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import  WandbLogger

from tbsim.utils.log_utils import PrintLogger
import tbsim.utils.sc_train_utils as TrainUtils
from tbsim.utils.env_utils import RolloutCallback
from tbsim.configs.registry import get_registered_experiment_config
from tbsim.datasets.factory import datamodule_factory
from tbsim.utils.config_utils import get_experiment_config_from_file
from tbsim.utils.batch_utils import set_global_batch_type
from tbsim.utils.trajdata_utils import set_global_trajdata_batch_env, set_global_trajdata_batch_raster_cfg
from tbsim.algos.factory import algo_factory

os.environ["WANDB_DISABLE_CODE"] = "True"
def main(cfg, auto_remove_exp_dir=False, debug=False):
    pl.seed_everything(cfg.seed)

    if cfg.env.name == "l5kit":
        set_global_batch_type("l5kit")
    elif cfg.env.name in ["nusc", "trajdata"]:
        set_global_batch_type("trajdata")
        if cfg.env.name == "nusc":
            set_global_trajdata_batch_env("nusc_trainval")
        elif cfg.env.name == "trajdata":
            # assumes all used trajdata datasets use share same map layers
            set_global_trajdata_batch_env(cfg.train.trajdata_source_train[0])
        set_global_trajdata_batch_raster_cfg(cfg.env.rasterizer) # determines cfg for rasterizing agents
    else:
        raise NotImplementedError("Env {} is not supported".format(cfg.env.name))

    print("\n============= New Training Run with Config =============")
    # print(cfg)
    # print("")
    root_dir, log_dir, ckpt_dir, video_dir, version_key = TrainUtils.get_exp_dir(
        exp_name=cfg.name,
        output_dir=cfg.root_dir,
        save_checkpoints=cfg.train.save.enabled,
        auto_remove_exp_dir=auto_remove_exp_dir
    )

    # Save experiment config to the training dir
    cfg.dump(os.path.join(root_dir, version_key, "config.json"))

    if cfg.train.logging.terminal_output_to_txt and not debug:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, "log.txt"))
        sys.stdout = logger
        sys.stderr = logger

    train_callbacks = []

    # Training Parallelism
    assert cfg.train.parallel_strategy in [
        "dp",
        "ddp_spawn",
        None,
    ]  # TODO: look into other strategies
    if not cfg.devices.num_gpus > 1:
        # Override strategy when training on a single GPU
        with cfg.train.unlocked():
            cfg.train.parallel_strategy = None


    # Dataset
    datamodule = datamodule_factory(
        cls_name=cfg.train.datamodule_class, config=cfg
    )
    # datamodule.setup()

    # Environment for close-loop evaluation


    # Model
    model = algo_factory(
        config=cfg,
        modality_shapes=datamodule.modality_shapes
    )


    ckpt_fixed_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="iter{step}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        monitor=None,
        every_n_train_steps=cfg.train.save.every_n_steps,
        verbose=True,
    )
    train_callbacks.append(ckpt_fixed_callback)


    logger = None
    if debug:
        print("Debugging mode, suppress logging.")
    elif cfg.train.logging.log_wandb:
        assert (
            "WANDB_APIKEY" in os.environ
        ), "Set api key by `export WANDB_APIKEY=<your-apikey>`"
        apikey = os.environ["WANDB_APIKEY"]
        wandb.login(key=apikey)
        logger = WandbLogger(
            name=cfg.name, 
            project=cfg.train.logging.wandb_project_name,
            log_model=True,
        )
        # record the entire config on wandb
        logger.experiment.config.update(cfg.to_dict())
        logger.watch(model=model)
    else:
        print("WARNING: not logging training stats")

    # Train
    trainer = pl.Trainer(
        default_root_dir=root_dir,
        # checkpointing
        enable_checkpointing=cfg.train.save.enabled,
        # logging
        logger=logger,
        # flush_logs_every_n_steps=cfg.train.logging.flush_every_n_steps,
        log_every_n_steps=cfg.train.logging.log_every_n_steps,
        # training
        max_steps=cfg.train.training.num_steps,
        # validation
        val_check_interval=cfg.train.validation.every_n_steps,
        limit_val_batches=cfg.train.validation.num_steps_per_epoch,
        # all callbacks
        callbacks=train_callbacks,


        num_sanity_val_steps=0,

    )

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="(optional) create experiment config from a preregistered name (see configs/registry.py)",
    )
    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    parser.add_argument(
        "--wandb_project_name",
        type=str,
        default=None,
        help="(optional) if provided, override the wandb project name defined in the config",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Root directory of training output (checkpoints, visualization, tensorboard log, etc.)",
    )

    parser.add_argument(
        "--remove_exp_dir",
        action="store_true",
        help="Whether to automatically remove existing experiment directory of the same name (remember to set this to "
        "True to avoid unexpected stall when launching cloud experiments).",
    )

 

    parser.add_argument(
        "--debug", action="store_true", help="Debug mode, suppress wandb logging, etc."
    )
    parser.add_argument(
        "--source_train", type=str, default=None, help="Trajdata source train"
    )
    parser.add_argument(
        "--source_valid", type=str, default=None, help="Trajdata source valid"
    )
    parser.add_argument(
        "--data_dirs", type=str, default=None, help="Trajdata data dirs"
    )
    parser.add_argument(
        "--rebuild_cache", type=str, default=None, help="Trajdata rebuild cache"
    )
    parser.add_argument(
        "--on_ngc", action="store_true", help="Running on NGC"
    )
    parser.add_argument(
    "--nusc_trainval_dir",type=str,default=None
    )
    parser.add_argument(
    "--nusc_mini_dir",type=str,default=None
    )
    parser.add_argument(
        "--datamodule_class", type=str, default="CLDDataModule", help="Trajdata datamodule class"
    )
    parser.add_argument(
        "--trajdata_cache_location", type=str, default=None, help="Wandb run name"
    )

    parser.add_argument(
        "--mix_gauss", type=int, default=None, help="mix_gauss"
    )
    parser.add_argument(
        "--training_num_steps", type=int, default=None, help="training_num_steps"
    )

    args = parser.parse_args()

    if args.config_name is not None:
        default_config = get_registered_experiment_config(args.config_name)
        print('args.config_name', args.config_name)
        print('default_config', default_config)
    elif args.config_file is not None:
        # Update default config with external json file
        default_config = get_experiment_config_from_file(args.config_file, locked=False)
    else:
        raise Exception(
            "Need either a config name or a json file to create experiment config"
        )

    if args.name is not None:
        default_config.name = args.name # wandb run name



    if args.output_dir is not None:
        default_config.root_dir = os.path.abspath(args.output_dir)

    if args.wandb_project_name is not None:
        default_config.train.logging.wandb_project_name = args.wandb_project_name
 

    if args.source_train is not None:
        default_config.train.trajdata_source_train = [args.source_train]
    if args.source_valid is not None:
        default_config.train.trajdata_source_valid = [args.source_valid]
    
    if args.training_num_steps is not None:
        default_config.train.training.num_steps = args.training_num_steps

    if args.rebuild_cache is not None:
        default_config.train.rebuild_cache = args.rebuild_cache
    if args.trajdata_cache_location is not None:
        default_config.train.trajdata_cache_location = args.trajdata_cache_location
  
    if args.nusc_trainval_dir is not None:
        default_config.train.trajdata_data_dirs["nusc_trainval"] = args.nusc_trainval_dir
    if args.nusc_mini_dir is not None:
        default_config.train.trajdata_data_dirs["nusc_mini"] = args.nusc_mini_dir
    if args.mix_gauss is not None:
        default_config.algo.mix_gauss = args.mix_gauss
    if args.datamodule_class is not None:
        default_config.train.datamodule_class = args.datamodule_class
    if args.debug:
        # Test policy rollout
        default_config.train.validation.every_n_steps = 5
        default_config.train.save.every_n_steps = 10
        default_config.train.rollout.every_n_steps = 10
        default_config.train.rollout.num_episodes = 1




    default_config.lock()  # Make config read-only
    main(default_config, auto_remove_exp_dir=args.remove_exp_dir, debug=args.debug)
