#!/bin/bash

echo "Setting up ADL HW1 environment with compatibility fixes..."

# 檢查是否已有環境
if conda env list | grep -q "adl-hw1"; then
    echo "Removing existing adl-hw1 environment..."
    conda deactivate 2>/dev/null || true
    conda env remove -n adl-hw1 -y
fi

# 創建新環境
echo "Creating new Python 3.10 environment..."
conda create -n adl-hw1 python=3.10 -y
conda activate adl-hw1

# 確認 Python 版本
python --version

# 清理可能的衝突
export PYTHONPATH=""
pip cache purge

# 安裝 PyTorch (使用 conda 來確保相容性)
echo "Installing PyTorch 2.1.0..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing CUDA version..."
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
else
    echo "No CUDA detected, installing CPU version..."
    conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch -y
fi

# 驗證 PyTorch 安裝
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 安裝其他套件
echo "Installing other required packages..."
pip install tokenizers==0.15.2
pip install transformers==4.50.0
pip install datasets==2.21.0
pip install accelerate==0.34.2
pip install scikit-learn==1.5.1
pip install nltk==3.9.1
pip install tqdm numpy pandas evaluate matplotlib gdown

# 最終驗證
echo "Final verification..."
python -c "
try:
    import torch
    import transformers
    import datasets
    import accelerate
    import sklearn
    import nltk
    print('✅ All packages imported successfully!')
    print(f'PyTorch: {torch.__version__}')
    print(f'Transformers: {transformers.__version__}')
    print(f'Datasets: {datasets.__version__}')
    print(f'Accelerate: {accelerate.__version__}')
    print(f'Scikit-learn: {sklearn.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

echo "Environment setup completed successfully!"
echo "Activate with: conda activate adl-hw1"