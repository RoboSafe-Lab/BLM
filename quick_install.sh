#!/bin/bash

# Quick Install Script - Non-interactive Mode
# Installs all components automatically

set -e

echo "=== Safety Critical Quick Install Script ==="

# Check conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed"
    exit 1
fi

# Check directory
if [ ! -d "CTG" ] || [ ! -d "trajdata" ] || [ ! -d "Pplan" ]; then
    echo "Error: Please run from the project root directory"
    exit 1
fi

# Create and activate environment
echo "1. Creating conda environment..."
conda create -n sc python=3.9 -y || true
eval "$(conda shell.bash hook)"
conda activate sc

# Install all components
echo "2. Installing CTG..."
cd CTG && pip install -e . && cd ..

echo "3. Installing trajdata..."
cd trajdata && pip install -e . && cd ..

echo "4. Installing Pplan..."
cd Pplan && pip install -e . && cd ..

echo "5. Installing PyTorch..."
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 torchmetrics==0.11.1 torchtext --extra-index-url https://download.pytorch.org/whl/cu113

echo "6. Fixing numpy..."
pip uninstall numpy torch -y || true
pip install numpy==1.21.5

echo "7. Installing other dependencies..."
pip install tianshou numba==0.56.4

echo "Installation complete!"
echo "Usage: conda activate sc" 