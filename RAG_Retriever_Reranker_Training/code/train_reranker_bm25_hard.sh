#!/bin/bash
# Train Reranker with BM25 Hard Negative Mining
# 使用 BM25 從整個資料庫挖掘 hard negatives

echo "============================================================"
echo "Training Reranker with BM25 Hard Negatives"
echo "============================================================"
echo ""
echo "Key improvements:"
echo "  - BM25 mines hard negatives from ENTIRE corpus"
echo "  - Not limited to evidences in training data"
echo "  - 90% hard negatives, 1:8 ratio"
echo "  - Query-level split, LR=1e-5, warmup=10%"
echo ""

# Check if rank-bm25 is installed
echo "Checking dependencies..."
python -c "import rank_bm25; print('✓ rank-bm25 installed')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "✗ rank-bm25 not installed"
    echo "Installing rank-bm25..."
    pip install rank-bm25
    if [ $? -ne 0 ]; then
        echo "Failed to install rank-bm25"
        exit 1
    fi
fi

echo ""
echo "Starting training..."
echo ""

python code/train_reranker_bm25_hard.py \
    --train_file ./data/train.txt \
    --corpus_file ./data/corpus.txt \
    --qrels_file ./data/qrels.txt \
    --model_name cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --output_dir ./models/reranker_bm25_hard \
    --batch_size 32 \
    --epochs 6 \
    --learning_rate 1e-5 \
    --num_negatives 8 \
    --hard_negative_ratio 0.9 \
    --bm25_top_k 100 \
    --dev_ratio 0.05 \
    --eval_steps 500 \
    --seed 42 \
    --max_length 512

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Training completed successfully!"
    echo "============================================================"
    echo ""
    echo "Model saved to: ./models/reranker_bm25_hard"
    echo "Training log: ./results/logs/training/reranker_bm25_training_log.json"
    echo ""
    echo "Next steps:"
    echo "  1. Run inference: ./run_inference_bm25_hard.sh"
    echo "  2. Compare results with ultra_v2"
else
    echo ""
    echo "Training failed!"
    exit 1
fi
