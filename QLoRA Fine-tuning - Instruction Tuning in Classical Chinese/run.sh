#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <model_path> <adapter_path> <input_file> <output_file>"
    exit 1
fi

# Set arguments
MODEL_PATH=$1
ADAPTER_PATH=$2
INPUT_FILE=$3
OUTPUT_FILE=$4

# Run prediction using python3 
python3 predict.py \
    --base_model_path "$MODEL_PATH" \
    --adapter_path "$ADAPTER_PATH" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" 
    # --max_new_tokens 512 \
    # --temperature 0.7 \
    # --top_p 0.9

echo "Prediction completed. Results saved to $OUTPUT_FILE"