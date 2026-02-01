# ADL 2025 HW2 - Classical Chinese Instruction Tuning

This repository contains the implementation for instruction tuning for Classical Chinese translation tasks using QLoRA (4-bit quantized Low-Rank Adaptation) with the Qwen3-4B model.  

### Run Training

**Tested Working Command:**
```bash
python3 train.py \
    --model_name "Qwen/Qwen3-4B" \
    --train_data "data/train.json" \
    --output_dir "./adapter_checkpoint" \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --gradient_accumulation_steps 16 \
    --max_length 256
```


## Task Overview

The objective is to fine-tune the Qwen3-4B language model to perform Classical Chinese translation tasks, including:
- Modern Chinese to Classical Chinese translation
- Classical Chinese to Modern Chinese translation

## Environment Setup

### Required Packages

**Tested Working Configuration:**
- torch==2.4.1
- transformers==4.56.2 (required for Qwen3-4B support)
- ml_dtypes==0.5.3 (required for transformers 4.56.2)
- bitsandbytes==0.44.1
- peft==0.13.0
- datasets==3.0.1
- gdown (for downloading models)

### Installation

<!-- #### Option 1: Automated Setup (Recommended)
```bash
# First, check system requirements
chmod +x system_check.sh
./system_check.sh

# Then setup environment
chmod +x setup_environment.sh
./setup_environment.sh
```

#### Option 2: Manual Installation -->

1. Update system packages:
```bash
sudo apt-get update
sudo apt-get install -y python3-pip git wget curl
```

2. Install required packages:
```bash
python3 -m pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install transformers>=4.51.0 bitsandbytes==0.44.1 peft==0.13.0 datasets==3.0.1
python3 -m pip install gdown accelerate matplotlib numpy tqdm
```

3. Make scripts executable:
```bash
chmod +x run.sh download.sh
```

## File Structure

```
adl-hw2/
├── train.py               # Training script for QLoRA fine-tuning
├── predict.py             # Prediction script for generating responses
├── utils.py               # Utility functions (prompt formatting, quantization config)
├── ppl.py                 # Perplexity evaluation script (provided by TAs)
├── run.sh                 # Inference script for testing
├── download.sh            # Script to download trained adapter weights
├── README.md              # This file
├── data/
│   ├── train.json         # Training data (10,000 samples)
│   ├── public_test.json   # Public testing data (250 samples)
│   └── private_test.json  # Private testing data (250 samples)
└── adapter_checkpoint/    # Directory containing trained LoRA weights (created after training)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Training Process

### 1. Data Format

The training data is in JSON format with the following structure:
```json
[
  {
    "id": "unique_id",
    "instruction": "翻譯成文言文：\n雅裏惱怒地說： 從前在福山田獵時，你誣陷獵官，現在又說這種話。",
    "output": "雅裏怒曰： 昔畋於福山，卿誣獵官，今復有此言。"
  }
]
```

### 2. Model Configuration

- **Base Model**: Qwen3-4B
- **Quantization**: 4-bit quantization using BitsAndBytesConfig
- **LoRA Configuration**:
  - r (rank): 16
  - alpha: 32
  - dropout: 0.1
  - target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

### 3. Training Hyperparameters

- **Learning Rate**: 2e-4
- **Batch Size**: 1 per device
- **Gradient Accumulation Steps**: 16 (effective batch size: 16)
- **Epochs**: 3
- **Max Sequence Length**: 256
- **Warmup Ratio**: 0.1
- **Weight Decay**: 0.01

### 4. Run Training

```bash
python train.py \
    --model_name "Qwen/Qwen3-4B" \
    --train_data "data/train.json" \
    --output_dir "./adapter_checkpoint" \
    --num_epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --gradient_accumulation_steps 16 \
    --max_length 256
```

## Inference

### Using run.sh

```bash
bash run.sh \
    "Qwen/Qwen3-4B" \
    "./adapter_checkpoint" \
    "data/public_test.json" \
    "output.json"
```

### Direct Python Call

```bash
python3 predict.py \
    --base_model_path "Qwen/Qwen3-4B" \
    --adapter_path "./adapter_checkpoint" \
    --input_file "data/public_test.json" \
    --output_file "output.json" \
```

## Evaluation

### Perplexity Evaluation

```bash
python3 ppl.py \
    --base_model_path "Qwen/Qwen3-4B" \
    --peft_path "./adapter_checkpoint" \
    --test_data_path "data/public_test.json"
```

## Implementation Details

### Prompt Template

The prompt template used is:
```
你是古文的專家，負責轉換古代文言文與現代白話文。接下來是你跟用戶的對話，你要對用戶的問題提供簡潔、有用、精確的轉換。USER:{instruction} ASSISTANT:
```


<!-- ## Troubleshooting

### Environment Issues

1. **ml_dtypes Compatibility Error (`float4_e2m1fn` missing)**:
   ```bash
   # Fix ml_dtypes issues
   chmod +x fix_ml_dtypes.sh
   ./fix_ml_dtypes.sh
   ```

2. **Complete Environment Issues**:
   ```bash
   # Nuclear option - fix everything
   chmod +x complete_fix.sh
   ./complete_fix.sh
   ```

3. **Qwen3 Model Not Supported Error**:
   ```bash
   # Fix Qwen3 support or use alternative
   chmod +x fix_qwen3_support.sh
   ./fix_qwen3_support.sh
   
   # Or test which models work
   python3 test_qwen_models.py
   
   # Manual alternative - use Qwen2.5 instead
   python3 train.py --model_name "Qwen/Qwen2.5-4B"
   ```

4. **PEFT-Transformers Compatibility Error**:
   ```bash
   # Quick fix for version compatibility
   chmod +x quick_fix.sh
   ./quick_fix.sh
   
   # Or manual fix
   pip uninstall transformers peft -y
   pip install transformers==4.46.0 peft==0.13.0
   ```

2. **Transformers Import Error**:
   ```bash
   # Run diagnostics
   python3 diagnose_environment.py
   
   # Try automatic fix
   chmod +x fix_transformers.sh
   ./fix_transformers.sh
   ```

2. **Working in Transformers Source Directory**:
   If you're in the transformers repository directory, this can cause import conflicts:
   ```bash
   cd /tmp  # or any other directory
   # Clone your homework repository there
   ```

3. **Package Conflicts**:
   ```bash
   # Clean reinstall
   pip3 uninstall transformers torch peft bitsandbytes -y
   pip3 cache purge
   ./setup_environment.sh
   ```

### Training Issues

1. **CUDA Out of Memory (RTX 3070 8GB)**:
   - Reduce batch_size to 1
   - Reduce max_length to 1024
   - Increase gradient_accumulation_steps

2. **Slow Training**:
   - Check GPU utilization: `nvidia-smi`
   - Ensure CUDA drivers are properly installed
   - Monitor temperature: `nvidia-smi -l 1`

3. **Import Errors During Training**:
   - Verify packages: `python3 test_implementation.py`
   - Check CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`

### Model Loading Issues

1. **Base Model Download**:
   - Ensure internet connection for first download
   - Model cached in `~/.cache/huggingface/`
   - Check disk space (model is ~7GB)

2. **Adapter Loading**:
   - Verify adapter files exist: `ls adapter_checkpoint/`
   - Check file sizes are reasonable
   - Ensure download.sh completed successfully -->

### Quick Diagnostics

Run these commands to check your setup:
```bash
# System check
chmod +x system_check.sh
./system_check.sh

# Environment diagnostics  
python3 diagnose_environment.py

# Test implementation
python3 test_implementation.py
```
