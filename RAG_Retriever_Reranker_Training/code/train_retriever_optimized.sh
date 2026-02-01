#!/bin/bash
# Optimized Retriever Training Script
# CRITICAL: Since TOP_K=10 is fixed, retriever must excel at top-10 retrieval
# Improvements over baseline:
# - Larger batch size (64 vs 32) for better contrastive learning
# - More epochs (6 vs 3) for better convergence
# - Lower learning rate (8e-6 vs 2e-5) for stable, fine-grained training
# - More negatives (8 vs 4) for harder discrimination task

echo "=========================================="
echo "Starting Optimized Retriever Training"
echo "=========================================="
echo ""
echo "Optimizations (focused on TOP-10 precision):"
echo "  • Batch size: 32 → 64 (better contrastive learning)"
echo "  • Epochs: 3 → 6 (better convergence)"
echo "  • Learning rate: 2e-5 → 8e-6 (very stable fine-tuning)"
echo "  • Num negatives: 4 → 8 (harder discrimination)"
echo ""
echo "Expected training time: 4-5 hours on A100"
echo "Expected improvements:"
echo "  • Higher Recall@10 (target: >0.84)"
echo "  • Better top-10 precision (critical for TOP_K=10 limit)"
echo "  • More robust embeddings"
echo ""
echo "=========================================="
echo ""

python code/train_retriever.py \
    --train_file ./data/train.txt \
    --corpus_file ./data/corpus.txt \
    --qrels_file ./data/qrels.txt \
    --model_name intfloat/multilingual-e5-small \
    --output_dir ./models/retriever_optimized_1019-2 \
    --batch_size 32 \
    --epochs 6 \
    --learning_rate 8e-6 \
    --num_negatives 8 \
    --use_hard_negatives

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Model saved to: ./models/retriever_optimized"
echo ""
echo "Next steps:"
echo "  1. Rebuild vector database:"
echo "     python save_embeddings.py --retriever_model_path ./models/retriever_optimized --build_db"
echo ""
echo "  2. Run inference with optimized retriever:"
echo "     python inference_batch.py --retriever_model_path ./models/retriever_optimized --reranker_model_path ./models/reranker_ultra"
echo ""
