#!/bin/bash

# Complete environment fix for all known compatibility issues
echo "=== Complete ADL HW2 Environment Fix ==="
echo

echo "This script will fix all known compatibility issues:"
echo "1. PEFT-Transformers version conflicts"
echo "2. ml_dtypes compatibility issues"  
echo "3. JAX/TensorFlow conflicts"
echo "4. CUDA-related warnings"
echo

read -p "Continue with complete fix? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo "=== Step 1: Clean Environment ==="
echo "Removing potentially conflicting packages..."
packages_to_remove="transformers peft ml_dtypes jax jaxlib tensorflow tensorrt"
for pkg in $packages_to_remove; do
    echo "Removing $pkg..."
    pip uninstall $pkg -y 2>/dev/null || true
done

echo "Clearing pip cache..."
pip cache purge

echo "=== Step 2: Install Core Packages ==="
echo "Installing PyTorch (CUDA 11.8)..."
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing compatible transformers and PEFT..."
pip install transformers==4.56.2  # Tested working version for Qwen3 support
pip install peft==0.13.0          # Compatible with newer transformers

echo "Installing quantization support..."
pip install bitsandbytes==0.44.1

echo "=== Step 3: Install Supporting Packages ==="
echo "Installing datasets and utilities..."
pip install datasets==3.0.1
pip install accelerate
pip install numpy
pip install scipy
pip install matplotlib
pip install tqdm
pip install gdown

echo "Installing compatible ml_dtypes (if needed)..."
pip install ml_dtypes==0.3.1 || echo "ml_dtypes not needed"

echo "=== Step 4: Test Installation ==="
echo "Testing core imports..."
python3 -c "
import sys
print(f'Python: {sys.version}')

# Test PyTorch
import torch
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')

# Test Transformers
import transformers
print(f'✓ Transformers: {transformers.__version__}')

# Test PEFT
import peft
print(f'✓ PEFT: {peft.__version__}')

# Test key classes
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
print('✓ All key classes imported successfully')

# Test utils functions
import sys
import os
sys.path.append('.')
try:
    from utils import get_prompt, get_bnb_config
    print('✓ Utils functions work')
except Exception as e:
    print(f'⚠ Utils test: {e}')

print('\n🎉 Environment setup completed successfully!')
"

if [ $? -eq 0 ]; then
    echo
    echo "=== SUCCESS! ==="
    echo "Your environment is now ready. You can run:"
    echo "  python3 test_implementation.py"
    echo "  python3 quick_test.py"
    echo "  python3 train.py  # to start training"
else
    echo
    echo "=== FAILED ==="
    echo "There are still issues. Please check the error messages above."
    echo "You may need to:"
    echo "1. Check your CUDA installation"
    echo "2. Try different package versions"
    echo "3. Use a fresh conda environment"
fi