"""
Generate training curve plots from training logs.
Supports both retriever and reranker training logs.
"""

import json
import argparse
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Any

def load_training_log(log_path: str) -> Dict[str, Any]:
    """Load training log from JSON file"""
    with open(log_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_retriever_curves(log_data: Dict[str, Any], output_dir: str):
    """Generate training curves for retriever model"""
    print("Generating retriever training curves...")
    
    evaluations = log_data.get('evaluations', [])
    if not evaluations:
        print("No evaluation data found in log!")
        return
    
    # Extract data
    steps = [e['steps'] for e in evaluations]
    epochs = [e['epoch'] for e in evaluations]
    losses = [e['avg_loss'] for e in evaluations]
    
    # Check for metrics
    has_ndcg = 'ndcg@10' in evaluations[0]
    has_recall = 'recall@10' in evaluations[0]
    has_map = 'map' in evaluations[0]
    
    # Create figure with subplots
    num_plots = 1 + sum([has_ndcg, has_recall, has_map])
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot 1: Training Loss
    axes[plot_idx].plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=4)
    axes[plot_idx].set_xlabel('Training Steps', fontsize=12)
    axes[plot_idx].set_ylabel('Average Loss', fontsize=12)
    axes[plot_idx].set_title('Retriever Training Loss Curve', fontsize=14, fontweight='bold')
    axes[plot_idx].grid(True, alpha=0.3)
    
    # Add epoch markers on secondary x-axis
    ax2 = axes[plot_idx].twiny()
    ax2.set_xlim(axes[plot_idx].get_xlim())
    epoch_ticks = [steps[i] for i in range(0, len(steps), max(1, len(steps)//5))]
    epoch_labels = [f"{epochs[steps.index(s)]:.1f}" for s in epoch_ticks]
    ax2.set_xticks(epoch_ticks)
    ax2.set_xticklabels(epoch_labels)
    ax2.set_xlabel('Epoch', fontsize=12)
    plot_idx += 1
    
    # Plot 2: NDCG@10 (if available)
    if has_ndcg:
        ndcg_scores = [e.get('ndcg@10', 0) for e in evaluations]
        axes[plot_idx].plot(steps, ndcg_scores, 'g-', linewidth=2, marker='s', markersize=4)
        axes[plot_idx].set_xlabel('Training Steps', fontsize=12)
        axes[plot_idx].set_ylabel('NDCG@10', fontsize=12)
        axes[plot_idx].set_title('Retriever NDCG@10 on Dev Set', fontsize=14, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Annotate best score
        best_idx = ndcg_scores.index(max(ndcg_scores))
        axes[plot_idx].annotate(f'Best: {ndcg_scores[best_idx]:.4f}',
                               xy=(steps[best_idx], ndcg_scores[best_idx]),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plot_idx += 1
    
    # Plot 3: Recall@10 (if available)
    if has_recall:
        recall_scores = [e.get('recall@10', 0) for e in evaluations]
        axes[plot_idx].plot(steps, recall_scores, 'r-', linewidth=2, marker='^', markersize=4)
        axes[plot_idx].set_xlabel('Training Steps', fontsize=12)
        axes[plot_idx].set_ylabel('Recall@10', fontsize=12)
        axes[plot_idx].set_title('Retriever Recall@10 on Dev Set', fontsize=14, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Annotate best score
        best_idx = recall_scores.index(max(recall_scores))
        axes[plot_idx].annotate(f'Best: {recall_scores[best_idx]:.4f}',
                               xy=(steps[best_idx], recall_scores[best_idx]),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plot_idx += 1
    
    # Plot 4: MAP (if available)
    if has_map:
        map_scores = [e.get('map', 0) for e in evaluations]
        axes[plot_idx].plot(steps, map_scores, 'm-', linewidth=2, marker='D', markersize=4)
        axes[plot_idx].set_xlabel('Training Steps', fontsize=12)
        axes[plot_idx].set_ylabel('MAP', fontsize=12)
        axes[plot_idx].set_title('Retriever MAP on Dev Set', fontsize=14, fontweight='bold')
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Annotate best score
        best_idx = map_scores.index(max(map_scores))
        axes[plot_idx].annotate(f'Best: {map_scores[best_idx]:.4f}',
                               xy=(steps[best_idx], map_scores[best_idx]),
                               xytext=(10, 10), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'retriever_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved retriever curves to: {output_path}")
    
    plt.close()

def plot_reranker_curves(log_data: Dict[str, Any], output_dir: str):
    """Generate training curves for reranker model"""
    print("Generating reranker training curves...")
    
    # Support both 'epochs' and 'evaluations' keys
    epochs_data = log_data.get('epochs', log_data.get('evaluations', []))
    if not epochs_data:
        print("No epoch data found in log!")
        return
    
    # Extract data
    epochs = [e['epoch'] for e in epochs_data]
    steps = [e['steps'] for e in epochs_data]
    losses = [e['avg_loss'] for e in epochs_data]
    dev_accs = [e.get('dev_accuracy', 0) for e in epochs_data if e.get('dev_accuracy') is not None]
    
    # Create figure with subplots
    has_dev_acc = len(dev_accs) > 0
    num_plots = 2 if has_dev_acc else 1
    
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    # Plot 1: Training Loss
    axes[0].plot(steps, losses, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Training Steps', fontsize=12)
    axes[0].set_ylabel('Average Loss', fontsize=12)
    axes[0].set_title('Reranker Training Loss Curve', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Add epoch markers on secondary x-axis
    ax2 = axes[0].twiny()
    ax2.set_xlim(axes[0].get_xlim())
    epoch_ticks = [steps[i] for i in range(0, len(steps), max(1, len(steps)//5))]
    epoch_labels = [f"{epochs[steps.index(s)]:.1f}" for s in epoch_ticks]
    ax2.set_xticks(epoch_ticks)
    ax2.set_xticklabels(epoch_labels)
    ax2.set_xlabel('Epoch', fontsize=12)
    
    # Plot 2: Dev Accuracy (if available)
    if has_dev_acc:
        dev_steps = [e['steps'] for e in epochs_data if e.get('dev_accuracy') is not None]
        axes[1].plot(dev_steps, dev_accs, 'g-', linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Training Steps', fontsize=12)
        axes[1].set_ylabel('Dev Accuracy', fontsize=12)
        axes[1].set_title('Reranker Dev Set Accuracy', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Annotate best score
        if dev_accs:
            best_idx = dev_accs.index(max(dev_accs))
            axes[1].annotate(f'Best: {dev_accs[best_idx]:.4f}',
                           xy=(dev_steps[best_idx], dev_accs[best_idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'reranker_training_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved reranker curves to: {output_path}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate training curve plots")
    parser.add_argument("--log_file", type=str, required=True, help="Path to training log JSON file")
    parser.add_argument("--output_dir", type=str, default="./results/plots", help="Output directory for plots")
    parser.add_argument("--model_type", type=str, choices=['retriever', 'reranker', 'auto'], 
                       default='auto', help="Model type (auto-detect from log)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load log
    print(f"Loading training log from: {args.log_file}")
    log_data = load_training_log(args.log_file)
    
    # Auto-detect model type if needed
    model_type = args.model_type
    if model_type == 'auto':
        if log_data.get('type', '').startswith('retriever'):
            model_type = 'retriever'
        elif log_data.get('type', '').startswith('reranker'):
            model_type = 'reranker'
        else:
            # Check data structure
            if 'evaluations' in log_data:
                model_type = 'retriever'
            elif 'epochs' in log_data:
                model_type = 'reranker'
            else:
                print("Error: Cannot determine model type from log!")
                return
    
    print(f"Detected model type: {model_type}")
    
    # Generate plots
    if model_type == 'retriever':
        plot_retriever_curves(log_data, args.output_dir)
    elif model_type == 'reranker':
        plot_reranker_curves(log_data, args.output_dir)
    
    print("\nTraining curves generated successfully!")
    print(f"Check output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
