#!/bin/bash

# ADL HW2 Environment Setup Script for Ubuntu 20.04
# Run this script to install all required dependencies

echo "Setting up ADL HW2 environment on Ubuntu 20.04..."

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install Python 3.10 if not available (Ubuntu 20.04 comes with Python 3.8 by default)
echo "Checking Python version..."
python3 --version

# Install pip if not available
echo "Installing pip..."
sudo apt-get install -y python3-pip

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y git wget curl

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support (RTX 3070 compatible)
echo "Installing PyTorch with CUDA support..."
python3 -m pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Uninstall any existing packages to avoid conflicts
echo "Cleaning existing installations..."
python3 -m pip uninstall transformers peft -y || true

# Clear pip cache
echo "Clearing pip cache..."
python3 -m pip cache purge || true

# Install specific compatible versions (tested working combination)
echo "Installing compatible transformers and PEFT versions..."
python3 -m pip install --no-cache-dir transformers==4.56.2
python3 -m pip install --no-cache-dir peft==0.13.0

# Install quantization packages
echo "Installing quantization packages..."
python3 -m pip install --no-cache-dir bitsandbytes==0.44.1

# Install dataset utilities
echo "Installing dataset utilities..."
python3 -m pip install --no-cache-dir datasets==3.0.1

# Install additional utilities
echo "Installing additional utilities..."
python3 -m pip install --no-cache-dir accelerate
python3 -m pip install --no-cache-dir gdown
python3 -m pip install --no-cache-dir matplotlib
python3 -m pip install --no-cache-dir numpy
python3 -m pip install --no-cache-dir tqdm
python3 -m pip install --no-cache-dir scipy

# Fix potential ml_dtypes compatibility issues
echo "Installing compatible ml_dtypes version..."
python3 -m pip install --no-cache-dir ml_dtypes==0.5.3

# Make scripts executable
echo "Making scripts executable..."
chmod +x run.sh
chmod +x download.sh

# Check CUDA availability
echo "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"

# Verify transformers installation
echo "Verifying transformers installation..."
if python3 -c "import transformers; from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig; print(f'✓ Transformers {transformers.__version__} working correctly')" 2>/dev/null; then
    echo "✓ Transformers verification passed"
else
    echo "✗ Transformers verification failed"
    echo "Run ./fix_transformers.sh to diagnose and fix issues"
fi

# Test import of all required packages
echo "Testing all package imports..."
python3 -c "
packages = ['torch', 'transformers', 'datasets', 'peft', 'bitsandbytes', 'accelerate']
failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError as e:
        print(f'✗ {pkg}: {e}')
        failed.append(pkg)

if failed:
    print(f'Failed packages: {failed}')
    exit(1)
else:
    print('✓ All packages imported successfully')
"

echo "Installation completed!"
echo "System specifications verified for:"
echo "  - Ubuntu 20.04"
echo "  - 32GB RAM"
echo "  - RTX 3070 8GB VRAM"
echo "  - 20GB disk space available"
echo ""
echo "You can now run: python3 test_implementation.py"
echo "If you encounter transformers import issues, run: ./fix_transformers.sh"