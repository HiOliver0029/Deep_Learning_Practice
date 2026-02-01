# ADL HW3: Retrieval-Augmented Generation (RAG)

This project implements a complete RAG pipeline with fine-tuned retriever and reranker models for question answering.

## Table of Contents
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Training Models](#training-models)
- [Running Inference](#running-inference)
- [Evaluation and Analysis](#evaluation-and-analysis)
- [Project Structure](#project-structure)

---

## Environment Setup

### Prerequisites
- Python 3.12
- CUDA-compatible GPU (recommended: at least 8GB VRAM)
- Ubuntu 20.04 or compatible system

### Installation

1. **Clone the repository and navigate to the project directory**
   ```bash
   cd adl-hw3
   ```

2. **Install required packages**
   
   **For x86_64 systems:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **For ARM64 (aarch64) systems with CUDA (e.g., AWS Graviton + A100):**
   ```bash
   # Use conda for better ARM64 + CUDA support
   conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```
   
   <!-- > See [INSTALL_ARM64_A100.md](INSTALL_ARM64_A100.md) for detailed ARM64 installation guide. -->

3. **Set up Hugging Face token** (required for LLM access)
   
   Create a `.env` file in the project root:
   ```bash
   echo "hf_token=YOUR_HUGGINGFACE_TOKEN" > .env
   ```
   
   Get your token from: https://huggingface.co/docs/hub/security-tokens

---

## Quick Start

### For Evaluation Only (Using Pre-trained Models)

1. **Download pre-trained models**
   ```bash
   bash download.sh
   ```
   This will download the fine-tuned retriever and reranker models to `./models/` directory.

2. **Run inference**
   ```bash
   bash run.sh ./data/test_open.txt
   ```
   Results will be saved to `./results/result.json`

---

## Training Models

### Step 1: Build Vector Database

Before training or inference, you need to build the vector database from the corpus:

```bash
python save_embeddings.py \
    --retriever_model_path intfloat/multilingual-e5-small \
    --build_db
```

This creates:
- `./vector_database/passage_index.faiss` - FAISS index for fast retrieval
- `./vector_database/passage_store.db` - SQLite database for passage storage

### Step 2: Train Retriever Model

Train the bi-encoder retriever using contrastive learning:

**Standard Training:**
```bash
python code/train_retriever.py \
    --train_file ./data/train.txt \
    --corpus_file ./data/corpus.txt \
    --qrels_file ./data/qrels.txt \
    --model_name intfloat/multilingual-e5-small \
    --output_dir ./models/retriever \
    --batch_size 32 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --num_negatives 4 \
    --use_hard_negatives
```

**Training details:**
- Loss function: MultipleNegativesRankingLoss (contrastive learning)
- Positive samples: From qrels.txt
- Negative samples: Combination of hard negatives (from evidences) and random corpus samples
- Training time: ~2-4 hours on RTX 3070

### Step 3: Train Reranker Model

#### Option A: Standard Training (Evidence-based Hard Negatives)

Train the cross-encoder reranker using evidences from training data:

```bash
python code/train_reranker.py \
    --train_file ./data/train.txt \
    --corpus_file ./data/corpus.txt \
    --qrels_file ./data/qrels.txt \
    --model_name cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --output_dir ./models/reranker \
    --batch_size 16 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --num_negatives 4 \
    --use_hard_negatives
```

**Training details:**
- Loss function: Binary Cross-Entropy
- Hard negatives: From evidences in training data
- Training time: ~2-3 hours on A100

#### Option B: BM25 Hard Negatives Mining (Recommended for Best Performance) 

Train reranker using BM25 to mine hard negatives from the **entire corpus**:

```bash
python code/train_reranker_bm25_hard_v2.py \
    --train_file ./data/train.txt \
    --corpus_file ./data/corpus.txt \
    --qrels_file ./data/qrels.txt \
    --model_name cross-encoder/ms-marco-MiniLM-L-12-v2 \
    --output_dir ./models/reranker \
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
```

**Key advantages of BM25 mining:**
- **Corpus-wide mining**: Hard negatives from entire corpus, not just training evidences
- **More diverse negatives**: BM25 finds lexically similar but semantically different passages
- **Better discrimination**: Model learns to distinguish harder cases
- **Query-level split**: Prevents data leakage between train/dev sets
- **Automatic evaluation**: Dev accuracy tracked during training

**Key parameters explained:**
- `hard_negative_ratio`: 0.9 = 90% of negatives from BM25, 10% random (prevents overfitting)
- `bm25_top_k`: 100 = Consider top 100 BM25 results for mining hard negatives
- `dev_ratio`: 0.05 = 5% queries for validation
- `eval_steps`: 500 = Evaluate every 500 steps

<!-- **Expected performance improvement:**
- **Baseline (evidence-only)**: MRR@10 ≈ 0.58-0.65
- **BM25 hard negatives**: MRR@10 ≈ 0.72+ 🎯
- Training time: ~4-6 hours on A100 -->

**Training details:**
- Loss function: Binary Cross-Entropy
- Training pairs: (query, passage, label) where label ∈ {0, 1}
- Negative sampling: 90% from BM25 top-100, 10% random corpus
- Automatic dev evaluation with CEBinaryClassificationEvaluator
- Saves best model based on dev accuracy

<!-- > 📖 See [BM25_HARD_NEGATIVES_GUIDE.md](BM25_HARD_NEGATIVES_GUIDE.md) for detailed explanation of BM25 mining strategy. -->

### Step 4: Rebuild Vector Database with Fine-tuned Retriever

After training the retriever, rebuild the vector database with the fine-tuned model:

```bash
python save_embeddings.py \
    --retriever_model_path ./models/retriever \
    --build_db
```

---

## Running Inference

### Method 1: Using run.sh

```bash
bash run.sh ./data/test_open.txt
```

### Method 2: Direct Python Command

```bash
python inference_batch.py \
    --test_data_path ./data/test_open.txt \
    --retriever_model_path ./models/retriever \
    --reranker_model_path ./models/reranker \
    --result_file_name result.json
```

**Output:**
- Results are saved to `./results/result.json`
- Console output shows: Recall@10, MRR@10, and Bi-Encoder CosSim

---

## Evaluation and Analysis

<!-- ### Analyze Results

Run the evaluation script to generate detailed analysis and visualizations:

```bash
python code/evaluate_results.py \
    --result_file ./results/result.json \
    --output_dir ./results/analysis
```

**Outputs:**
- `analysis.json`: Detailed metrics and statistics
- `retrieval_impact.png`: Impact of retrieval quality on answer accuracy
- `reranker_scores.png`: Distribution of reranker scores
- `mrr_distribution.png`: MRR distribution across queries -->

### Prompt Experimentation

To test different prompt variations for the report:

1. See `code/prompt_variations.py` for 6 different prompt templates
2. Copy desired prompt functions to `utils.py`
3. Run inference with different prompts
4. Compare results to document in report

---

## Project Structure

```
adl-hw3/
├── data/
│   ├── corpus.txt              # Passage corpus
│   ├── train.txt               # Training data
│   ├── test_open.txt           # Test data
│   └── qrels.txt               # Query-passage relevance labels
├── code/
│   ├── train_retriever.py      # Retriever training script
│   ├── train_reranker.py       # Reranker training script
│   ├── evaluate_results.py     # Result analysis script
│   └── prompt_variations.py    # Different prompt templates
├── models/                      # Trained models (after training)
│   ├── retriever/
│   └── reranker/
├── vector_database/             # FAISS index and SQLite DB
│   ├── passage_index.faiss
│   └── passage_store.db
├── results/                     # Inference results
│   └── result.json
├── save_embeddings.py          # Build vector database
├── inference_batch.py          # Inference pipeline
├── utils.py                    # Prompt functions
├── download.sh                 # Download trained models
├── run.sh                      # Run inference
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` in training scripts
- Reduce `BATCH_Q` and `BATCH_GEN` in `inference_batch.py`

### Model Download Issues
- Ensure `download.sh` has correct URLs
- Check internet connection
- Verify file hosting service is accessible

### CUDA Not Available
- Ensure PyTorch is installed with CUDA support
- Check GPU drivers: `nvidia-smi`

---

## References

- Retriever base model: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)
- Reranker base model: [cross-encoder/ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)
- LLM: [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
