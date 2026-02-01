"""
Training script for fine-tuning the retriever model (intfloat/multilingual-e5-small).
Uses contrastive learning with MultipleNegativesRankingLoss.
Enhanced with epoch-level loss logging.
"""

import json
import argparse
import random
from datetime import datetime
import os
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train retriever model")
    parser.add_argument("--train_file", type=str, default="./data/train.txt", help="Path to training data")
    parser.add_argument("--corpus_file", type=str, default="./data/corpus.txt", help="Path to corpus")
    parser.add_argument("--qrels_file", type=str, default="./data/qrels.txt", help="Path to qrels")
    parser.add_argument("--model_name", type=str, default="intfloat/multilingual-e5-small", help="Base model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./models/retriever", help="Output directory for trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--num_negatives", type=int, default=4, help="Number of negative samples per query")
    parser.add_argument("--use_hard_negatives", action="store_true", help="Use hard negatives from evidences")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
    Load and construct training examples with query, positive, and negative passages.
    
    Args:
        train_file: Path to training data
        qrels: Query to positive passage mapping
        corpus: Full corpus dictionary
        num_negatives: Number of negative samples per query
        use_hard_negatives: Whether to use hard negatives from evidences
    
    Returns:
        List of InputExample for training
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
            
            # For each positive passage, create training examples with negatives
            for pos_pid in positive_pids:
                if pos_pid not in corpus:
                    continue
                
                positive_text = corpus[pos_pid]
                
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
                
                # Create InputExample with query, positive, and negatives
                # For MultipleNegativesRankingLoss, we create pairs
                # The loss will handle in-batch negatives automatically
                if negatives:
                    # Add prefix for multilingual-e5 model
                    query_with_prefix = "query: " + query
                    positive_with_prefix = "passage: " + positive_text
                    
                    training_examples.append(
                        InputExample(texts=[query_with_prefix, positive_with_prefix])
                    )
    
    print(f"Created {len(training_examples)} training examples")
    return training_examples

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("=" * 80)
    print("Training Retriever Model")
    print("=" * 80)
    print(f"Base model: {args.model_name}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of negatives: {args.num_negatives}")
    print(f"Use hard negatives: {args.use_hard_negatives}")
    print("=" * 80)
    
    # Load model
    print("\nLoading base model...")
    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length
    
    # Load data
    corpus = load_corpus(args.corpus_file)
    qrels = load_qrels(args.qrels_file)
    training_examples = load_training_data(
        args.train_file, qrels, corpus, 
        args.num_negatives, args.use_hard_negatives
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        training_examples, 
        shuffle=True, 
        batch_size=args.batch_size
    )
    
    # Define loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Training configuration
    num_epochs = args.epochs
    warmup_steps = args.warmup_steps
    
    print(f"\nTraining for {num_epochs} epochs with {len(training_examples)} examples...")
    print(f"Steps per epoch: {len(train_dataloader)}")
    print(f"Warmup steps: {warmup_steps}")
    
    # Prepare training metadata storage
    training_metadata = {
        'type': 'retriever',
        'model': args.model_name,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'counts': {
            'training_examples': len(training_examples),
            'steps_per_epoch': len(train_dataloader),
            'total_steps': len(train_dataloader) * num_epochs,
        },
        'epochs': []
    }
    
    # Custom callback to log loss per epoch
    class EpochLogger:
        def __init__(self, metadata: Dict[str, Any], output_dir: str):
            self.metadata = metadata
            self.output_dir = output_dir
            self.current_epoch = 0
            self.epoch_losses = []
            
        def __call__(self, score, epoch, steps):
            """Called after each evaluation"""
            self.current_epoch = epoch
            
            # Calculate average loss for this epoch
            if self.epoch_losses:
                avg_loss = sum(self.epoch_losses) / len(self.epoch_losses)
                epoch_data = {
                    'epoch': epoch,
                    'steps': steps,
                    'avg_loss': float(avg_loss),
                    'num_batches': len(self.epoch_losses)
                }
                self.metadata['epochs'].append(epoch_data)
                print(f"  Epoch {epoch} completed - Avg Loss: {avg_loss:.4f}")
                
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
                log_path = os.path.join('./results/logs/training', 'retriever_training_log.json')
                with open(log_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[warn] Failed to save training log: {e}")
    
    # Create logger
    epoch_logger = EpochLogger(training_metadata, args.output_dir)
    
    # Monkey-patch the train_loss to capture loss values
    original_forward = train_loss.forward
    def logged_forward(sentence_features, labels):
        loss_value = original_forward(sentence_features, labels)
        epoch_logger.record_loss(loss_value.item())
        return loss_value
    train_loss.forward = logged_forward
    
    # Train the model
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        optimizer_params={'lr': args.learning_rate},
        show_progress_bar=True,
        checkpoint_save_steps=len(train_dataloader),  # Save after each epoch
        checkpoint_path=args.output_dir,
        callback=epoch_logger
    )
    
    print(f"\nTraining completed! Model saved to {args.output_dir}")
    
    # Final save of training metadata
    training_metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    training_metadata['output_dir'] = args.output_dir
    
    try:
        os.makedirs('./results/logs/training', exist_ok=True)
        log_path = os.path.join('./results/logs/training', 'retriever_training_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(training_metadata, f, ensure_ascii=False, indent=2)
        print(f"Training log saved to: {log_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("Training Summary:")
        print("=" * 80)
        for epoch_data in training_metadata['epochs']:
            print(f"Epoch {epoch_data['epoch']}: Avg Loss = {epoch_data['avg_loss']:.4f}")
        print("=" * 80)
    except Exception as e:
        print(f"[warn] Failed to write final training log: {e}")

if __name__ == "__main__":
    main()
