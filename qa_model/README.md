# ADL 2025 HW1 - Extractive Question Answering

This repository contains the implementation for ADL 2025 Homework 1: Extractive Question Answering on Traditional Chinese dataset.

## Environment Setup

### Requirements
- Python 3.10
- PyTorch 2.1.0
- Transformers 4.50.0
- Datasets 2.21.0
- Accelerate 0.34.2
- Other dependencies as specified in requirements.txt

### Installation

1. **Create and activate conda environment:**
```bash
chmod +x setup_environment.sh
./setup_environment.sh
conda activate adl-hw1
```

2. **Or install manually:**
```bash
conda create -n adl-hw1 python=3.10 -y
conda activate adl-hw1
pip install -r requirements.txt
```

## Data Preparation

### Download Dataset
```bash
chmod +x download.sh
./download.sh
```

This will download the required files zipped in a folder and then unzip it.

## Usage

### Quick Start
```bash
chmod +x run.sh
./run.sh [context_file] [test_file] [prediction_file]
```

**Example:**
```bash
./run.sh ./context.json ./test.json ./prediction.json
```

### Manual Training and Inference

#### 1. Train Paragraph Selection Model
```bash
# Convert data to multiple choice format for paragraph selection
python convert_kaggle_to_mc.py ./train.json ./context.json ./mc_train.json 
python convert_kaggle_to_mc.py ./valid.json ./context.json ./mc_valid.json

# Train model
python run_swag_no_trainer.py \
    --model_name_or_path hfl/chinese-lert-base \
    --train_file mc_train.json \
    --validation_file mc_valid.json \
    --max_seq_length 512 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --output_dir ./mc_output \
    --with_tracking \
    --report_to tensorboard

```

#### 2. Train Span Selection (QA) Model  
```bash
accelerate launch --mixed_precision=fp16 run_qa_no_trainer.py \
    --train_file ./train.json \
    --validation_file ./valid.json \
    --context_file ./context.json \
    --model_name_or_path hfl/chinese-lert-base \
    --tokenizer_name hfl/chinese-lert-base \
    --output_dir ./qa_output \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 3 \
    --max_seq_length 512 \
    --doc_stride 128 \
    --pad_to_max_length \
    --with_tracking \
    --report_to tensorboard

```

#### 3. Run Inference
```bash
# Step 1: Paragraph selection
python inference_paragraph_selection.py \
    --model_path ./mc_output \
    --test_file ./test.json \
    --context_file ./context.json \
    --output_file ./selected_test.json \
    --max_seq_length 512

# Step 2: Span selection
accelerate launch --mixed_precision=fp16 run_qa_no_trainer_test.py \
    --model_name_or_path ./qa_output \
    --test_file ./selected_test.json \
    --context_file context.json \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_device_eval_batch_size 4 \
    --output_dir ./qa_output/predict \
    --do_predict

# Step 3: Format final predictions
python json_to_csv.py \
    --predictions_file ./qa_output/predict/predict_predictions.json \
    --output_file prediction.csv

```

## File Structure

```
adl_hw1
├── README.md
├── requirements.txt
├── setup_environment.sh
├── setup_environment_fixed.sh    # In case setup_environment.sh doesn't work, use this instead
├── download.sh
├── run.sh
├── convert_kaggle_to_mc.py       # Convert original data to mc model format
├── convert_to_qa_format.py       # Convert data to QA model format
├── run_swag_no_trainer.py        # Paragraph selection training
├── run_qa_no_trainer.py          # QA training script
├── inference_paragraph_selection.py # Paragraph selection inference
├── run_qa_no_trainer_test.py     # QA inference script
├── train.json
├── valid.json
├── test.json
├── context.json
├── mc_output
    └── model, config, tokenizer
├── qa_output
    └── model, config, tokenizer
    └── predict/
        └── predict_predictions.json
```
After paragraph selection inference, the 'selected_test.json' file will be generated under the main directory. It will then be used for the QA inference. After the QA inference and format change, the 'prediction.csv' file will also be generated under the main directory.

## Memory Optimization

For systems with limited GPU memory:

1. **Reduce batch size**: Use `--per_device_train_batch_size 1`
2. **Increase gradient accumulation**: Use `--gradient_accumulation_steps 16`
3. **Reduce sequence length**: Use `--max_seq_length 256`
4. **Use gradient checkpointing**: Automatically enabled
5. **Set memory environment**: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Use smaller batch size and more gradient accumulation
```

### Slow Training
```bash
# Use mixed precision training
--mixed_precision=fp16
# Reduce sequence length
--max_seq_length 256
```

### Package Version Conflicts
```bash
# Reinstall with exact versions
pip install -r requirements.txt --force-reinstall
```

## References

- [Chinese LERT Model](https://huggingface.co/hfl/chinese-lert-base)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Accelerate Library](https://huggingface.co/docs/accelerate/)
- Github Copilot
- ChatGPT
