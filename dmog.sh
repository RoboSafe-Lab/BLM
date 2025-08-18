#!/bin/bash
#SBATCH --job-name=xiao
#SBATCH --array=0             # 提交2个任务 (任务索引0、1)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3-24:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH -p gpu

source ~/anaconda3/etc/profile.d/conda.sh
conda activate sc

export PYTHONPATH=$HOME/safety_critical/
export WANDB_APIKEY=1e2e2040243d63e8577d02a48e3881807e1944d5
# 定义checkpoint文件路径（如果需要从checkpoint继续训练，请取消注释并填写路径）



python ~/safety_critical/CTG/scripts/ppo.py \
    --config_file "simulation_config.json" \
    # --rebuild_cache "false" \
    # --num_Gaussian "5" \
    # --training_num_steps 50000 \
    # --name "dmog-k5"

