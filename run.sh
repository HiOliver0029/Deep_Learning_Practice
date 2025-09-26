#!/bin/bash

# ADL HW1 - Question Answering System
# Usage: ./run.sh [context_file] [test_file] [prediction_file]

# Default parameters
CONTEXT_FILE=${1:-"./context.json"}
TEST_FILE=${2:-"./test.json"}
PRED_FILE=${3:-"./prediction.csv"}

echo "Running ADL HW1 Question Answering System"
echo "Context file: $CONTEXT_FILE"
echo "Test file: $TEST_FILE"
echo "Output prediction file: $PRED_FILE"

# Check if required files exist
if [ ! -f "$CONTEXT_FILE" ]; then
    echo "Error: Context file $CONTEXT_FILE not found!"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file $TEST_FILE not found!"
    exit 1
fi

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_USE_CHECKPOINTING=1

# Check if models exist, if not train them
if [ ! -d "./mc_output" ]; then
    echo "Training paragraph selection model..."
    
    # Convert data to multiple choice format for paragraph selection
    python convert_kaggle_to_mc.py ./train.json $CONTEXT_FILE ./mc_train.json 
    python convert_kaggle_to_mc.py ./valid.json $CONTEXT_FILE ./mc_valid.json
    
    # Train paragraph selection model
    accelerate launch --mixed_precision=fp16 run_swag_no_trainer.py \
        --model_name_or_path hfl/chinese-lert-base \
        --train_file ./mc_train.json \
        --validation_file ./mc_valid.json \
        --max_seq_length 512 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --output_dir ./mc_output \
        --with_tracking \
        --report_to tensorboard
fi

if [ ! -d "./qa_output" ]; then
    echo "Training span selection model..."
    
    # Train span selection (QA) model
    accelerate launch --mixed_precision=fp16 run_qa_no_trainer.py \
        --model_name_or_path hfl/chinese-lert-base \
        --train_file ./train.json \
        --validation_file ./valid.json \
        --context_file $CONTEXT_FILE \
        --max_seq_length 512 \
        --doc_stride 128 \
        --per_device_train_batch_size 4 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 3e-5 \
        --num_train_epochs 2 \
        --output_dir ./qa_output \
        --with_tracking \
        --report_to tensorboard
fi

echo "Starting inference..."

# Step 1: Paragraph Selection - 輸出篩選後的test格式
echo "Step 1: Selecting relevant paragraphs..."
python inference_paragraph_selection.py \
    --model_path ./mc_output \
    --test_file $TEST_FILE \
    --context_file $CONTEXT_FILE \
    --output_file ./selected_test.json \
    --max_seq_length 512

# Step 2: Span Selection (Final QA) - 直接使用篩選後的test格式
echo "Step 2: Extracting answer spans..."
accelerate launch --mixed_precision=fp16 run_qa_no_trainer_test.py \
    --model_name_or_path ./qa_output \
    --test_file ./selected_test.json \
    --context_file $CONTEXT_FILE \
    --max_seq_length 512 \
    --doc_stride 128 \
    --per_device_eval_batch_size 4 \
    --output_dir ./qa_output/predict \
    --do_predict

# Step 3: Format final predictions
echo "Step 3: Formatting final predictions..."
python json_to_csv.py \
    --predictions_file ./qa_output/predict/predict_predictions.json \
    --output_file $PRED_FILE

echo "Prediction completed! Results saved to $PRED_FILE"