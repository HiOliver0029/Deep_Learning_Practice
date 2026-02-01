"""
Enhanced training script for reranker with BM25 hard negatives mining and reliable loss logging.
Version 2: Improved loss capture mechanism.
"""

import json
import argparse
import random
from datetime import datetime
import os
from typing import List, Dict, Any
from tqdm import tqdm
import torch
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample
try:
    # Try new API first (sentence-transformers >= 3.0)
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator as CEClassificationEvaluator
    USE_NEW_API = True
except ImportError:
    # Fall back to old API
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    USE_NEW_API = False
from torch.utils.data import DataLoader
from rank_bm25 import BM25Okapi
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train reranker with BM25 hard negatives mining")
    parser.add_argument("--train_file", type=str, default="./data/train.txt", help="Path to training data")
    parser.add_argument("--corpus_file", type=str, default="./data/corpus.txt", help="Path to corpus")
    parser.add_argument("--qrels_file", type=str, default="./data/qrels.txt", help="Path to qrels")
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2", help="Base model")
    parser.add_argument("--output_dir", type=str, default="./models/reranker_bm25_hard_v2", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_negatives", type=int, default=8, help="Number of negative samples per query")
    parser.add_argument("--hard_negative_ratio", type=float, default=0.9, help="Ratio of BM25 hard negatives")
    parser.add_argument("--bm25_top_k", type=int, default=100, help="Top-K passages for BM25 mining")
    parser.add_argument("--dev_ratio", type=float, default=0.05, help="Development set ratio")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory for examples")
    parser.add_argument("--use_cache", action="store_true", help="Use cached examples if available")
    return parser.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_corpus(corpus_file: str) -> Dict[str, str]:
    print(f"Loading corpus from {corpus_file}...")
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading corpus"):
            if not line.strip():
                continue
            obj = json.loads(line)
            corpus[obj['id']] = obj['text']
    print(f"Loaded {len(corpus)} passages")
    return corpus

def load_qrels(qrels_file: str) -> Dict[str, List[str]]:
    print(f"Loading qrels from {qrels_file}...")
    with open(qrels_file, 'r', encoding='utf-8') as f:
        qrels = json.load(f)
    
    qid2positive = {}
    for qid, pid_dict in qrels.items():
        positive_pids = [pid for pid, label in pid_dict.items() if label == 1]
        if positive_pids:
            qid2positive[qid] = positive_pids
    
    print(f"Loaded qrels for {len(qid2positive)} queries")
    return qid2positive

def build_bm25_index(corpus: Dict[str, str]):
    print("Building BM25 index...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[pid] for pid in corpus_ids]
    
    tokenized_corpus = [text.lower().split() for text in tqdm(corpus_texts, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("BM25 index built successfully")
    return bm25, corpus_ids

def mine_hard_negatives_bm25(query: str, bm25, corpus_ids: List[str], corpus: Dict[str, str],
                              positive_pids: List[str], top_k: int = 100) -> List[str]:
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    hard_negatives = []
    for idx in top_indices:
        pid = corpus_ids[idx]
        if pid not in positive_pids:
            hard_negatives.append(pid)
    
    return hard_negatives

def prepare_training_data(train_file: str, qrels: Dict[str, List[str]], corpus: Dict[str, str],
                         bm25, corpus_ids: List[str], args, dev_ratio: float = 0.05):
    print(f"Preparing training data with BM25 hard negatives...")
    
    # Load queries
    all_queries = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get('qid') in qrels and obj.get('rewrite'):
                all_queries.append(obj)
    
    # Split train/dev by queries
    random.shuffle(all_queries)
    split_idx = int(len(all_queries) * (1 - dev_ratio))
    train_queries = all_queries[:split_idx]
    dev_queries = all_queries[split_idx:]
    
    print(f"Split: {len(train_queries)} train queries, {len(dev_queries)} dev queries")
    
    def create_examples(queries, desc):
        examples = []
        for obj in tqdm(queries, desc=desc):
            qid = obj['qid']
            query = obj['rewrite']
            positive_pids = qrels[qid]
            
            # Mine hard negatives using BM25
            hard_negatives = mine_hard_negatives_bm25(
                query, bm25, corpus_ids, corpus, positive_pids, args.bm25_top_k
            )
            
            for pos_pid in positive_pids:
                if pos_pid not in corpus:
                    continue
                
                # Sample negatives (mix of BM25 hard negatives and random)
                num_hard = int(args.num_negatives * args.hard_negative_ratio)
                num_random = args.num_negatives - num_hard
                
                sampled_negatives = []
                
                # Sample from BM25 hard negatives
                if hard_negatives and num_hard > 0:
                    sampled_negatives.extend(random.sample(
                        hard_negatives, min(num_hard, len(hard_negatives))
                    ))
                
                # Fill with random negatives
                while len(sampled_negatives) < args.num_negatives:
                    neg_pid = random.choice(corpus_ids)
                    if neg_pid not in positive_pids and neg_pid not in sampled_negatives:
                        sampled_negatives.append(neg_pid)
                
                # Create positive example
                examples.append(InputExample(
                    texts=[query, corpus[pos_pid]],
                    label=1.0
                ))
                
                # Create negative examples
                for neg_pid in sampled_negatives[:args.num_negatives]:
                    examples.append(InputExample(
                        texts=[query, corpus[neg_pid]],
                        label=0.0
                    ))
        
        return examples
    
    train_examples = create_examples(train_queries, "Creating train examples")
    dev_examples = create_examples(dev_queries, "Creating dev examples")
    
    print(f"Created {len(train_examples)} train examples, {len(dev_examples)} dev examples")
    
    return train_examples, dev_examples, train_queries, dev_queries


def save_examples_cache(train_examples, dev_examples, cache_dir):
    """Save examples to cache for faster reloading."""
    import pickle
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, 'reranker_examples_cache.pkl')
    print(f"\nSaving examples cache to {cache_file}...")
    
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'train_examples': train_examples,
            'dev_examples': dev_examples
        }, f)
    
    print(f"✓ Cached {len(train_examples)} train + {len(dev_examples)} dev examples")


def load_examples_cache(cache_dir):
    """Load examples from cache if available."""
    import pickle
    cache_file = os.path.join(cache_dir, 'reranker_examples_cache.pkl')
    
    if not os.path.exists(cache_file):
        return None, None
    
    print(f"\nLoading examples from cache: {cache_file}...")
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        train_examples = data['train_examples']
        dev_examples = data['dev_examples']
        
        print(f"✓ Loaded {len(train_examples)} train + {len(dev_examples)} dev examples from cache")
        return train_examples, dev_examples
    except Exception as e:
        print(f"Warning: Failed to load cache: {e}")
        return None, None

# Custom DataLoader that tracks loss
class LossTrackingDataLoader:
    def __init__(self, examples, batch_size, logger):
        self.examples = examples
        self.batch_size = batch_size
        self.logger = logger
        
    def __len__(self):
        return (len(self.examples) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        indices = list(range(len(self.examples)))
        random.shuffle(indices)
        
        for i in range(0, len(self.examples), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.examples[idx] for idx in batch_indices]
            yield batch

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 80)
    print("Training Reranker with BM25 Hard Negatives (Version 2 - Enhanced Loss Logging)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Hard negative ratio: {args.hard_negative_ratio}")
    print(f"BM25 top-k: {args.bm25_top_k}")
    print("=" * 80)
    
    # Load data
    corpus = load_corpus(args.corpus_file)
    qrels = load_qrels(args.qrels_file)
    
    # Try to load from cache if requested
    train_examples = None
    dev_examples = None
    train_queries = []
    dev_queries = []
    
    if args.use_cache:
        train_examples, dev_examples = load_examples_cache(args.cache_dir)
    
    # If cache not available or not requested, generate examples
    if train_examples is None or dev_examples is None:
        print("\n" + "="*80)
        print("Generating training examples (this may take a while)...")
        print("="*80)
        
        bm25, corpus_ids = build_bm25_index(corpus)
        
        # Prepare training data
        train_examples, dev_examples, train_queries, dev_queries = prepare_training_data(
            args.train_file, qrels, corpus, bm25, corpus_ids, args, args.dev_ratio
        )
        
        # Save to cache for next time
        save_examples_cache(train_examples, dev_examples, args.cache_dir)
    else:
        print("\n✓ Using cached examples - skipped BM25 index building and example generation")
        # Note: train_queries and dev_queries not available from cache
    
    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_length)
    
    # Prepare evaluator - extract data from InputExample objects
    dev_sentences1 = [ex.texts[0] for ex in dev_examples]
    dev_sentences2 = [ex.texts[1] for ex in dev_examples]
    dev_labels = [ex.label for ex in dev_examples]
    
    if USE_NEW_API:
        # New API: CrossEncoderClassificationEvaluator
        # Combine sentence pairs into tuples
        dev_sentence_pairs = list(zip(dev_sentences1, dev_sentences2))
        evaluator = CEClassificationEvaluator(
            dev_sentence_pairs, 
            dev_labels, 
            name="dev"
        )
    else:
        # Old API: CEBinaryClassificationEvaluator
        evaluator = CEBinaryClassificationEvaluator(
            dev_sentences1, dev_sentences2, dev_labels, name="dev"
        )
    
    # Training configuration
    num_steps_per_epoch = (len(train_examples) + args.batch_size - 1) // args.batch_size
    total_steps = num_steps_per_epoch * args.epochs
    warmup_steps = min(5000, int(0.1 * total_steps))
    
    print(f"\nTraining configuration:")
    print(f"  Train examples: {len(train_examples)}")
    print(f"  Dev examples: {len(dev_examples)}")
    print(f"  Steps per epoch: {num_steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Eval steps: {args.eval_steps}")
    
    # Training metadata
    training_metadata = {
        'type': 'reranker_bm25_hard_v2',
        'model': args.model_name,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'counts': {
            'train_examples': len(train_examples),
            'dev_examples': len(dev_examples),
            'train_queries': len(train_queries),
            'dev_queries': len(dev_queries),
            'steps_per_epoch': num_steps_per_epoch,
            'total_steps': total_steps,
        },
        'evaluations': []
    }
    
    # Enhanced logging callback
    class EnhancedLogger:
        def __init__(self, metadata: Dict[str, Any]):
            self.metadata = metadata
            self.global_step = 0
            self.recent_losses = []  # Store recent losses from stdout parsing
            
        def __call__(self, score, epoch, steps):
            """Called after evaluation"""
            # Calculate average of recent losses
            avg_loss = sum(self.recent_losses) / len(self.recent_losses) if self.recent_losses else 0.0
            
            eval_data = {
                'epoch': float(epoch),
                'steps': steps,
                'avg_loss': float(avg_loss),
                'dev_accuracy': float(score) if score is not None else None,
            }
            
            self.metadata['evaluations'].append(eval_data)
            
            acc_val = float(score) if score is not None else 0.0
            print(f"\n  Step {steps} (Epoch {epoch:.2f}): Loss={avg_loss:.4f}, Dev Acc={acc_val:.4f}")
            
            self._save_log()
            self.recent_losses = []  # Clear after logging
        
        def record_loss(self, loss_value):
            """Record loss value"""
            self.recent_losses.append(float(loss_value))
            self.global_step += 1
        
        def _save_log(self):
            try:
                os.makedirs('./results/logs/training', exist_ok=True)
                with open('./results/logs/training/reranker_bm25_v2_training_log.json', 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[warn] Failed to save log: {e}")
    
    logger = EnhancedLogger(training_metadata)
    
    # Wrap stdout to capture loss from transformers logging
    import sys
    import re
    
    class LossCaptureWrapper:
        def __init__(self, original_stdout, logger):
            self.original_stdout = original_stdout
            self.logger = logger
            self.loss_pattern = re.compile(r"'loss':\s*([\d.]+)")
        
        def write(self, text):
            self.original_stdout.write(text)
            # Try to extract loss from transformers log output
            match = self.loss_pattern.search(text)
            if match:
                try:
                    loss_value = float(match.group(1))
                    self.logger.record_loss(loss_value)
                except:
                    pass
        
        def flush(self):
            self.original_stdout.flush()
    
    # Install stdout wrapper
    original_stdout = sys.stdout
    sys.stdout = LossCaptureWrapper(original_stdout, logger)
    
    # Create DataLoader from InputExample objects
    from torch.utils.data import DataLoader as TorchDataLoader
    
    train_dataloader = TorchDataLoader(
        train_examples,
        shuffle=True,
        batch_size=args.batch_size
    )
    
    # Train with transformers callback
    print("\nStarting training...")
    try:
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=args.epochs,
            evaluation_steps=args.eval_steps,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': args.learning_rate},
            output_path=args.output_dir,
            save_best_model=True,
            show_progress_bar=True,
            callback=logger
        )
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
    
    # Save final model
    model.save(args.output_dir)
    print(f"\nTraining completed! Model saved to {args.output_dir}")
    
    # Save final log
    training_metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        os.makedirs('./results/logs/training', exist_ok=True)
        log_path = './results/logs/training/reranker_bm25_v2_training_log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\nTraining log saved to: {log_path}")
        print("\n" + "=" * 80)
        print("Training Summary:")
        print("=" * 80)
        for eval_data in training_metadata['evaluations']:
            print(f"Step {eval_data['steps']} (Epoch {eval_data['epoch']:.2f}): "
                  f"Loss={eval_data['avg_loss']:.4f}, "
                  f"Dev Acc={eval_data.get('dev_accuracy', 0):.4f}")
        print("=" * 80)
    except Exception as e:
        print(f"[warn] Failed to save final log: {e}")

if __name__ == "__main__":
    main()
