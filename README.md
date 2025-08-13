# Adversarial Diffusion via Guided Training by Flexible Reward Mechanism


## Installation


#### 步骤 1:下载项目 

```bash
mkdir ~/safety-critical
cd safety-critical
git clone --single-branch --branch sc https://github.com/RoboSafe-Lab/CLDPlus.git .
```
#### 步骤 2:安装 创建conda环境
```bash
conda create -n sc python=3.9 -y
conda activate sc
# conda remove pytorch pytorch-lightning --force -y
```
#### 步骤 3: 安装 CTG 
```bash
# git clone https://github.com/NVlabs/CTG.git
cd CTG
pip install -e .
```

#### 步骤 3: 安装定制版本的 trajdata
```bash
cd ..
# git clone https://github.com/AIasd/trajdata.git
cd trajdata
pip install -e .
```

#### 步骤 4: 安装 Pplan
```bash
cd ..
# git clone https://github.com/NVlabs/spline-planner.git Pplan
cd Pplan
pip install -e .
```


#### 步骤 5: 解决潜在问题
如果遇到问题，可能需要运行以下命令:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 torchmetrics==0.11.1 torchtext --extra-index-url https://download.pytorch.org/whl/cu113
pip uninstall numpy torch
pip install numpy==1.21.5
```







## Quick start
### 1. Obtain dataset(s)
We currently support the nuScenes [dataset](https://www.nuscenes.org/nuscenes).


#### nuScenes
* Download the nuScenes dataset (with the v1.3 map extension pack) and organize the dataset directory as follows:
    ```
    nuscenes/
    │   maps/
    │   v1.0-mini/
    │   v1.0-trainval/
    ```


### 2. Train a diffuser model
#### 步骤 1: 训练基本模型
nuScenes dataset (Note: remove `--debug` flag when doing the actual training and to support wandb logging):
```bash
python scripts/train.py --dataset_path <path-to-nuscenes-data-directory> --config_name trajdata_nusc_diff --debug
```

#### 步骤 2a: 训练CTG模型 (具体示例)
```bash
python scripts/train.py --dataset_path ../behavior-generation-dataset/nuscenes --config_name trajdata_nusc_diff --debug
```

#### 步骤 2b: 训练CTG++模型 (具体示例)
```bash
python scripts/train.py --dataset_path ../behavior-generation-dataset/nuscenes --config_name trajdata_nusc_scene_diff --debug
```

### 3. Run rollout of a trained model (closed-loop simulation)
#### 步骤 1: 运行基本Rollout
```bash
python scripts/scene_editor.py \
  --results_root_dir nusc_results/ \
  --num_scenes_per_batch 1 \
  --dataset_path <path-to-nuscenes-data-directory> \
  --env trajdata \
  --policy_ckpt_dir <path-to-checkpoint-dir> \
  --policy_ckpt_key <ckpt-file-identifier> \
  --eval_class <class-of-model-to-rollout> \
  --editing_source 'config' 'heuristic' \
  --registered_name 'trajdata_nusc_diff' \
  --render
```

#### 步骤 2a: 运行CTG模型 (使用预训练模型的具体示例)
```bash
python scripts/scene_editor.py \
  --results_root_dir nusc_results/ \
  --num_scenes_per_batch 1 \
  --dataset_path ../behavior-generation-dataset/nuscenes \
  --env trajdata \
  --policy_ckpt_dir ../../summer_project/behavior-generation/trained_models_only_new/trajdata_nusc/ctg_original \
  --policy_ckpt_key iter70000.ckpt \
  --eval_class Diffuser \
  --editing_source 'config' 'heuristic' \
  --registered_name 'trajdata_nusc_diff' \
  --render
```

#### 步骤 2b: 运行CTG++模型 (使用预训练模型的具体示例)
```bash
python scripts/scene_editor.py \
  --results_root_dir nusc_results/ \
  --num_scenes_per_batch 1 \
  --dataset_path ../behavior-generation-dataset/nuscenes \
  --env trajdata \
  --policy_ckpt_dir ../../summer_project/behavior-generation/trained_models_only_new/trajdata_nusc/ctg++8_9,10edge \
  --policy_ckpt_key iter50000.ckpt \
  --eval_class SceneDiffuser \
  --editing_source 'config' 'heuristic' \
  --registered_name 'trajdata_nusc_scene_diff' \
  --render
```

### 4. Parse Results for rollout
#### 步骤 3: 解析rollout结果
```bash
python scripts/parse_scene_edit_results.py --results_dir <rollout_results_dir> --estimate_dist
```

## Pre-trained models
We have provided checkpoints for models of CTG and CTG++ [here](https://drive.google.com/drive/folders/17oYCNGTzBPWjKqvvA8JO67WswyI0j5vw?usp=sharing). 
Note that the provided CTG model slightly differ from that in the original CTG paper. The main difference is that the prediction horizon is 52 rather than 20. The pre-trained models are provided under the **CC-BY-NC-SA-4.0 license**.

## 连接ssh gateway
#### 步骤 1: 连接到SSH网关
```bash
ssh -p 44788 yx3006@sshgw.hw.ac.uk
```
输入：学校邮箱密码
输入：Authenticator密码

#### 步骤 2: 连接到目标服务器
```bash
ssh yx3006@dmog.hw.ac.uk
```



