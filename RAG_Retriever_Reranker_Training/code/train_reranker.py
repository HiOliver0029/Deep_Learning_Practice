"""
Training script for fine-tuning the reranker model (cross-encoder/ms-marco-MiniLM-L-12-v2).
Uses binary classification with cross-encoder architecture.
Enhanced with epoch-level loss and accuracy logging.
"""

import json
import argparse
import random
from datetime import datetime
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import torch
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import numpy as np
import math
import os
import gc


def report_gpu():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # print(torch.cuda.list_gpu_processes())
    gc.collect()
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(description="Train reranker model")
    parser.add_argument("--train_file", type=str, default="./data/train.txt", help="Path to training data")
    parser.add_argument("--corpus_file", type=str, default="./data/corpus.txt", help="Path to corpus")
    parser.add_argument("--qrels_file", type=str, default="./data/qrels.txt", help="Path to qrels")
    parser.add_argument("--model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2", help="Base model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./models/reranker", help="Output directory for trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num_negatives", type=int, default=4, help="Number of negative samples per positive")
    parser.add_argument("--use_hard_negatives", action="store_true", help="Use hard negatives from evidences")
    parser.add_argument("--seed", type=int, default=24, help="Random seed")
    return parser.parse_args()

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_corpus(corpus_file: str) -> dict:
    """Load corpus passages into a dictionary"""
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

def load_qrels(qrels_file: str) -> dict:
    """Load qrels mapping"""
    print(f"Loading qrels from {qrels_file}...")
    with open(qrels_file, 'r', encoding='utf-8') as f:
        qrels = json.load(f)
    
    # Convert to qid -> positive_pids mapping
    qid2positive = {}
    for qid, pid_dict in qrels.items():
        positive_pids = [pid for pid, label in pid_dict.items() if label == 1]
        if positive_pids:
            qid2positive[qid] = positive_pids
    
    print(f"Loaded qrels for {len(qid2positive)} queries")
    return qid2positive

def load_training_data(train_file: str, qrels: dict, corpus: dict, 
                       num_negatives: int, use_hard_negatives: bool) -> List[InputExample]:
    """
    Load and construct training examples for cross-encoder.
    Creates (query, passage, label) tuples where label=1 for positive, 0 for negative.
    
    Args:
        train_file: Path to training data
        qrels: Query to positive passage mapping
        corpus: Full corpus dictionary
        num_negatives: Number of negative samples per positive
        use_hard_negatives: Whether to use hard negatives from evidences
    
    Returns:
        List of InputExample with (query, passage) texts and label
    """
    print(f"Loading training data from {train_file}...")
    training_examples = []
    corpus_ids = list(corpus.keys())
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Constructing training examples"):
            if not line.strip():
                continue
            
            obj = json.loads(line)
            qid = obj.get('qid')
            query = obj.get('rewrite')
            
            if not query or qid not in qrels:
                continue
            
            # Get positive passages
            positive_pids = qrels[qid]
            
            # For each positive passage, create training examples
            for pos_pid in positive_pids:
                if pos_pid not in corpus:
                    continue
                
                positive_text = corpus[pos_pid]
                
                # Add positive example (label = 1)
                training_examples.append(
                    InputExample(texts=[query, positive_text], label=1.0)
                )
                
                # Sample negative passages
                negatives = []
                
                # 1. Use hard negatives from evidences if available and enabled
                if use_hard_negatives and 'evidences' in obj and 'retrieval_labels' in obj:
                    evidences = obj['evidences']
                    labels = obj['retrieval_labels']
                    
                    # Get passages that are labeled as negative (label = 0)
                    hard_negatives = [
                        evidences[i] for i in range(len(evidences))
                        if i < len(labels) and labels[i] == 0
                    ]
                    
                    # Randomly sample from hard negatives
                    if hard_negatives:
                        num_hard = min(num_negatives // 2, len(hard_negatives))
                        negatives.extend(random.sample(hard_negatives, num_hard))
                
                # 2. Fill remaining with random negatives from corpus
                num_remaining = num_negatives - len(negatives)
                attempts = 0
                max_attempts = num_remaining * 10
                
                while len(negatives) < num_negatives and attempts < max_attempts:
                    neg_pid = random.choice(corpus_ids)
                    # Ensure it's not a positive passage
                    if neg_pid not in positive_pids and corpus[neg_pid] not in negatives:
                        negatives.append(corpus[neg_pid])
                    attempts += 1
                
                # Add negative examples (label = 0)
                for neg_text in negatives:
                    training_examples.append(
                        InputExample(texts=[query, neg_text], label=0.0)
                    )
    
    print(f"Created {len(training_examples)} training examples")
    # Count positives and negatives
    num_pos = sum(1 for ex in training_examples if ex.label == 1.0)
    num_neg = len(training_examples) - num_pos
    print(f"  Positive examples: {num_pos}")
    print(f"  Negative examples: {num_neg}")
    
    return training_examples

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 80)
    print("Training Reranker Model")
    print("=" * 80)
    print(f"Base model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of negatives per positive: {args.num_negatives}")
    print(f"Use hard negatives: {args.use_hard_negatives}")
    print("=" * 80)
    
    report_gpu()

    # Load model
    print("\nLoading base model...")
    model = CrossEncoder(args.model_name, num_labels=1, max_length=args.max_length)
    
    # Load data
    corpus = load_corpus(args.corpus_file)
    qrels = load_qrels(args.qrels_file)
    training_examples = load_training_data(
        args.train_file, qrels, corpus, 
        args.num_negatives, args.use_hard_negatives
    )
    
    # Shuffle training examples
    random.shuffle(training_examples)
    
    # Split into train and dev (5% for dev)
    dev_ratio = 0.05
    split_idx = int(len(training_examples) * (1 - dev_ratio))
    train_examples = training_examples[:split_idx]
    dev_examples = training_examples[split_idx:]
    
    print(f"Train examples: {len(train_examples)}")
    print(f"Dev examples: {len(dev_examples)}")
    
    # Create dev evaluator
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_examples, name='dev')
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    # Calculate evaluation steps (evaluate after each epoch)
    num_steps_per_epoch = len(train_dataloader)
    evaluation_steps = num_steps_per_epoch
    
    print(f"\nTraining for {args.epochs} epochs with {len(train_examples)} examples...")
    print(f"Steps per epoch: {num_steps_per_epoch}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Evaluation steps: {evaluation_steps}")
    
    # Prepare training metadata storage
    training_metadata = {
        'type': 'reranker',
        'model': args.model_name,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'counts': {
            'training_examples': len(train_examples),
            'dev_examples': len(dev_examples),
            'steps_per_epoch': num_steps_per_epoch,
            'total_steps': num_steps_per_epoch * args.epochs,
        },
        'epochs': []
    }
    
    # Custom callback to log metrics per epoch
    class EpochLogger:
        def __init__(self, metadata: Dict[str, Any], base_evaluator):
            self.metadata = metadata
            self.base_evaluator = base_evaluator
            self.current_epoch = 0
            self.epoch_losses = []
            
        def __call__(self, score, epoch, steps):
            """Called after each evaluation"""
            # score is the accuracy from evaluator
            self.current_epoch = epoch
            
            # Calculate average loss for this epoch
            avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else 0.0
            
            epoch_data = {
                'epoch': epoch,
                'steps': steps,
                'avg_loss': float(avg_loss),
                'dev_accuracy': float(score) if score is not None else None,
                'num_batches': len(self.epoch_losses)
            }
            self.metadata['epochs'].append(epoch_data)
            
            print(f"  Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Dev Accuracy: {score:.4f if score else 0:.4f}")
            
            # Save metadata after each epoch
            self._save_metadata()
            self.epoch_losses = []
        
        def record_loss(self, loss_value):
            """Record loss for current batch"""
            self.epoch_losses.append(float(loss_value))
        
        def _save_metadata(self):
            """Save training metadata to JSON"""
            try:
                os.makedirs('./results/logs/training', exist_ok=True)
                log_path = os.path.join('./results/logs/training', 'reranker_training_log.json')
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[warn] Failed to save training log: {e}")
    
    # Create logger
    epoch_logger = EpochLogger(training_metadata, evaluator)
    
    # Monkey-patch model's fit method to capture loss
    original_fit = model.fit
    
    def logged_fit(*args, **kwargs):
        # Store reference to logger for loss capturing
        model._epoch_logger = epoch_logger
        return original_fit(*args, **kwargs)
    
    # Patch the model's loss calculation to record losses
    if hasattr(model.model, 'forward'):
        original_forward = model.model.forward
        
        def logged_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            # Extract loss if available
            if hasattr(output, 'loss') and output.loss is not None:
                if hasattr(model, '_epoch_logger'):
                    model._epoch_logger.record_loss(output.loss.item())
            return output
        
        model.model.forward = logged_forward
    
    # Train the model with evaluation
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=args.warmup_steps,
        optimizer_params={'lr': args.learning_rate},
        output_path=args.output_dir,
        save_best_model=True,
        show_progress_bar=True,
        callback=epoch_logger
    )
    
    # Save the trained model
    print(f"\nSaving model to {args.output_dir}...")
    model.save(args.output_dir)
    print(f"Training completed! Model saved to {args.output_dir}")
    
    # Final save of training metadata
    training_metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_metadata['output_dir'] = args.output_dir
    
    try:
        os.makedirs('./results/logs/training', exist_ok=True)
        log_path = os.path.join('./results/logs/training', 'reranker_training_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_metadata, f, ensure_ascii=False, indent=2)
        print(f"Training log saved to: {log_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Training Summary:")
        print("=" * 80)
        for epoch_data in training_metadata['epochs']:
            acc = epoch_data.get('dev_accuracy', 'N/A')
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            print(f"Epoch {epoch_data['epoch']}: Loss = {epoch_data['avg_loss']:.4f}, Dev Accuracy = {acc_str}")
        print("=" * 80)
    except Exception as e:
        print(f"[warn] Failed to write final training log: {e}")
    
    del model

if __name__ == "__main__":
    main()
