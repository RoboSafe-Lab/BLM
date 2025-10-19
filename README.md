# Bi-level Training of Latent Diffusion Model for Traffic Simulation




#### Download Project

```bash
mkdir ~/safety_critical
cd safety_critical
git clone --single-branch --branch sc https://github.com/RoboSafe-Lab/CLDPlus.git .
```
#### create conda
```bash
conda create -n sc python=3.9 -y
conda activate sc
```
#### Install CTG
```bash
cd CTG
pip install -e .
```

#### Install trajdata
```bash
cd ../trajdata
pip install -e .
```

#### Install Pplan
```bash
cd ../Pplan
pip install -e .
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

### 2. Training Configuration

#### Debug Configuration (VS Code)
For debugging purposes, use the following VS Code debug configuration:



#### Training Command Line
For actual training, use the following command:

```bash
python CTG/scripts/cld_training.py \
    --config_file cld_config.json \
    --source_train nusc_mini-mini_train \

python CTG/scripts/vae_diffusion_train.py \
    --config_file vae_diffusion.json \
   
```



**Starting Training:**
1. Ensure all dependencies are installed
2. Run the training command above from the project root directory
3. Monitor training progress in the terminal

The training process will display loss values, validation metrics, and other training progress information in the terminal.








