#!/bin/bash
# download.sh - Download fine-tuned retriever and reranker models
# This script must complete within 1 hour and download at most 2GB
# 
# Models are hosted on Google Drive (publicly accessible)
# - Retriever: https://drive.google.com/drive/folders/10ICgzsz_ojfieJAaQeuyiOGFJVFekInk
# - Reranker: https://drive.google.com/drive/folders/1T7CWqywBZmHMRxy1LE19OmnvAK6IWMQF

set -e

echo "======================================================================"
echo "Downloading fine-tuned models from Google Drive..."
echo "======================================================================"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install -q gdown
fi

# Create models directory
mkdir -p models

# Download Retriever model from Google Drive folder
echo ""
echo "[1/2] Downloading retriever model..."
echo "      Source: Google Drive folder (10ICgzsz_ojfieJAaQeuyiOGFJVFekInk)"
echo "----------------------------------------------------------------------"
if gdown --folder https://drive.google.com/drive/folders/10ICgzsz_ojfieJAaQeuyiOGFJVFekInk -O models/retriever --remaining-ok 2>&1 | grep -q "Download completed"; then
    echo "✓ Retriever model downloaded successfully"
elif gdown --folder 10ICgzsz_ojfieJAaQeuyiOGFJVFekInk -O models/retriever --remaining-ok 2>&1; then
    echo "✓ Retriever model downloaded (using folder ID)"
else
    echo "✓ Retriever download attempted (check verification below)"
fi

# Download Reranker model from Google Drive folder
echo ""
echo "[2/2] Downloading reranker model..."
echo "      Source: Google Drive folder (1T7CWqywBZmHMRxy1LE19OmnvAK6IWMQF)"
echo "----------------------------------------------------------------------"
if gdown --folder https://drive.google.com/drive/folders/1T7CWqywBZmHMRxy1LE19OmnvAK6IWMQF -O models/reranker --remaining-ok 2>&1 | grep -q "Download completed"; then
    echo "✓ Reranker model downloaded successfully"
elif gdown --folder 1T7CWqywBZmHMRxy1LE19OmnvAK6IWMQF -O models/reranker --remaining-ok 2>&1; then
    echo "✓ Reranker model downloaded (using folder ID)"
else
    echo "✓ Reranker download attempted (check verification below)"
fi

echo ""
echo "======================================================================"
echo "Verifying downloads..."
echo "======================================================================"
echo ""

# Verification function
verify_model() {
    local model_name=$1
    local model_path=$2
    
    if [ -d "$model_path" ] && [ "$(ls -A $model_path 2>/dev/null)" ]; then
        local file_count=$(ls -1 $model_path 2>/dev/null | wc -l)
        echo "✓ $model_name: OK ($file_count files)"
        
        # Check for essential files
        if [ -f "$model_path/model.safetensors" ]; then
            local size=$(du -sh "$model_path/model.safetensors" 2>/dev/null | cut -f1)
            echo "  └─ model.safetensors: $size"
        fi
        if [ -f "$model_path/config.json" ]; then
            echo "  └─ config.json: present"
        fi
        return 0
    else
        echo "✗ $model_name: FAILED or empty"
        echo "  └─ Path: $model_path"
        return 1
    fi
}

# Verify both models
retriever_ok=0
reranker_ok=0

verify_model "Retriever" "models/retriever" && retriever_ok=1
echo ""
verify_model "Reranker" "models/reranker" && reranker_ok=1

echo ""
echo "======================================================================"

if [ $retriever_ok -eq 1 ] && [ $reranker_ok -eq 1 ]; then
    echo "✓ All models downloaded successfully!"
    echo "✓ Models are ready for inference!"
    echo ""
    echo "Next step: bash run.sh ./data/test_open.txt"
    echo "======================================================================"
    exit 0
else
    echo "⚠ Some downloads may have failed."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Ensure Google Drive folders are public (Anyone with link)"
    echo "  2. Check your internet connection"
    echo "  3. Try running the script again"
    echo ""
    echo "If issues persist, please check DOWNLOAD_SETUP_GUIDE.md"
    echo "======================================================================"
    exit 1
fi
