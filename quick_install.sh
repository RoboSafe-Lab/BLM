#!/bin/bash

# 快速安装脚本 - 无交互模式
# 直接安装所有组件

set -e

echo "=== Safety Critical 快速安装脚本 ==="

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "错误: conda 未安装"
    exit 1
fi

# 检查目录
if [ ! -d "CTG" ] || [ ! -d "trajdata" ] || [ ! -d "Pplan" ]; then
    echo "错误: 请在项目根目录运行"
    exit 1
fi

# 创建并激活环境
echo "1. 创建conda环境..."
conda create -n sc python=3.9 -y || true
eval "$(conda shell.bash hook)"
conda activate sc

# 安装所有组件
echo "2. 安装 CTG..."
cd CTG && pip install -e . && cd ..

echo "3. 安装 trajdata..."
cd trajdata && pip install -e . && cd ..

echo "4. 安装 Pplan..."
cd Pplan && pip install -e . && cd ..

echo "5. 安装PyTorch..."
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 torchmetrics==0.11.1 torchtext --extra-index-url https://download.pytorch.org/whl/cu113

echo "6. 修复numpy..."
pip uninstall numpy torch -y || true
pip install numpy==1.21.5

echo "7. 安装其他依赖..."
pip install tianshou numba==0.56.4

echo "安装完成！"
echo "使用方法: conda activate sc" 