"""
Evaluation script for analyzing inference results.
Provides detailed metrics and visualizations for the report.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate inference results")
    parser.add_argument("--result_file", type=str, default="./results/result.json", help="Path to result JSON")
    parser.add_argument("--output_dir", type=str, default="./results/analysis", help="Output directory for analysis")
    return parser.parse_args()

def analyze_results(result_file: str, output_dir: str):
    """Analyze and visualize inference results"""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print(f"Loading results from {result_file}...")
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = data['records']
    print(f"Total queries: {len(records)}")
    print(f"Recall@10: {data.get('recall@10', 'N/A'):.4f}")
    print(f"MRR@10: {data.get('mrr@10', 'N/A'):.4f}")
    print(f"Bi-Encoder CosSim: {data.get('Bi-Encoder_CosSim', 'N/A'):.4f}")
    
    # Analysis 1: Compare performance when correct answer is retrieved vs not
    retrieved_correct = []
    not_retrieved_correct = []
    
    for record in records:
        gold_pids = set(record['gold_pids'])
        retrieved_pids = [r['pid'] for r in record['retrieved'][:10]]
        
        # Check if any gold passage was retrieved
        has_gold = any(pid in gold_pids for pid in retrieved_pids)
        
        # Calculate answer similarity (simplified - check if answers match)
        generated = record['generated'].lower()
        gold = record['gold_answer'].lower()
        
        # Simple match check
        if gold == "cannotanswer":
            is_correct = generated == "cannotanswer"
        else:
            is_correct = gold in generated or generated in gold
        
        if has_gold:
            retrieved_correct.append(1 if is_correct else 0)
        else:
            not_retrieved_correct.append(1 if is_correct else 0)
    
    print(f"\nAnalysis 1: Impact of Retrieval Quality")
    print(f"  Queries with gold passage retrieved: {len(retrieved_correct)}")
    print(f"    Answer accuracy: {np.mean(retrieved_correct):.4f}" if retrieved_correct else "    Answer accuracy: N/A")
    print(f"  Queries without gold passage: {len(not_retrieved_correct)}")
    print(f"    Answer accuracy: {np.mean(not_retrieved_correct):.4f}" if not_retrieved_correct else "    Answer accuracy: N/A")
    
    # Analysis 2: Distribution of reranker scores
    reranker_scores = []
    gold_passage_scores = []
    non_gold_passage_scores = []
    
    for record in records:
        gold_pids = set(record['gold_pids'])
        for item in record['retrieved']:
            score = item['score']
            reranker_scores.append(score)
            if item['pid'] in gold_pids:
                gold_passage_scores.append(score)
            else:
                non_gold_passage_scores.append(score)
    
    print(f"\nAnalysis 2: Reranker Score Distribution")
    print(f"  All passages - Mean: {np.mean(reranker_scores):.4f}, Std: {np.std(reranker_scores):.4f}")
    print(f"  Gold passages - Mean: {np.mean(gold_passage_scores):.4f}, Std: {np.std(gold_passage_scores):.4f}")
    print(f"  Non-gold passages - Mean: {np.mean(non_gold_passage_scores):.4f}, Std: {np.std(non_gold_passage_scores):.4f}")
    
    # Analysis 3: MRR distribution
    mrr_values = []
    for record in records:
        gold_pids = set(record['gold_pids'])
        retrieved_pids = [r['pid'] for r in record['retrieved'][:10]]
        
        mrr = 0.0
        for rank, pid in enumerate(retrieved_pids):
            if pid in gold_pids:
                mrr = 1.0 / (rank + 1)
                break
        mrr_values.append(mrr)
    
    print(f"\nAnalysis 3: MRR Distribution")
    print(f"  Mean MRR: {np.mean(mrr_values):.4f}")
    print(f"  Queries with perfect rank (MRR=1.0): {sum(1 for m in mrr_values if m == 1.0)}")
    print(f"  Queries with no retrieval (MRR=0.0): {sum(1 for m in mrr_values if m == 0.0)}")
    
    # Create visualizations
    try:
        # Plot 1: Retrieval quality impact
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        categories = ['Gold Retrieved', 'Gold Not Retrieved']
        accuracies = [
            np.mean(retrieved_correct) if retrieved_correct else 0,
            np.mean(not_retrieved_correct) if not_retrieved_correct else 0
        ]
        ax.bar(categories, accuracies, color=['green', 'red'])
        ax.set_ylabel('Answer Accuracy')
        ax.set_title('Impact of Retrieval Quality on Answer Accuracy')
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(f"{output_dir}/retrieval_impact.png", dpi=150)
        print(f"\nSaved plot: {output_dir}/retrieval_impact.png")
        
        # Plot 2: Reranker score distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(gold_passage_scores, bins=30, alpha=0.5, label='Gold Passages', color='green')
        ax.hist(non_gold_passage_scores, bins=30, alpha=0.5, label='Non-Gold Passages', color='red')
        ax.set_xlabel('Reranker Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Reranker Score Distribution')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/reranker_scores.png", dpi=150)
        print(f"Saved plot: {output_dir}/reranker_scores.png")
        
        # Plot 3: MRR distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(mrr_values, bins=20, color='blue', alpha=0.7)
        ax.set_xlabel('MRR Value')
        ax.set_ylabel('Frequency')
        ax.set_title('MRR Distribution')
        ax.axvline(np.mean(mrr_values), color='red', linestyle='--', label=f'Mean: {np.mean(mrr_values):.4f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mrr_distribution.png", dpi=150)
        print(f"Saved plot: {output_dir}/mrr_distribution.png")
        
    except Exception as e:
        print(f"\nWarning: Could not create plots: {e}")
        print("You may need to install matplotlib: pip install matplotlib")
    
    # Save detailed analysis to JSON
    analysis = {
        "overall_metrics": {
            "recall@10": data.get('recall@10'),
            "mrr@10": data.get('mrr@10'),
            "bi_encoder_cossim": data.get('Bi-Encoder_CosSim'),
            "total_queries": len(records)
        },
        "retrieval_impact": {
            "queries_with_gold": len(retrieved_correct),
            "queries_without_gold": len(not_retrieved_correct),
            "accuracy_with_gold": float(np.mean(retrieved_correct)) if retrieved_correct else 0.0,
            "accuracy_without_gold": float(np.mean(not_retrieved_correct)) if not_retrieved_correct else 0.0
        },
        "reranker_analysis": {
            "all_passages": {
                "mean": float(np.mean(reranker_scores)),
                "std": float(np.std(reranker_scores))
            },
            "gold_passages": {
                "mean": float(np.mean(gold_passage_scores)),
                "std": float(np.std(gold_passage_scores))
            },
            "non_gold_passages": {
                "mean": float(np.mean(non_gold_passage_scores)),
                "std": float(np.std(non_gold_passage_scores))
            }
        },
        "mrr_analysis": {
            "mean": float(np.mean(mrr_values)),
            "perfect_rank": sum(1 for m in mrr_values if m == 1.0),
            "no_retrieval": sum(1 for m in mrr_values if m == 0.0)
        }
    }
    
    analysis_file = f"{output_dir}/analysis.json"
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\nSaved analysis: {analysis_file}")

if __name__ == "__main__":
    args = parse_args()
    analyze_results(args.result_file, args.output_dir)
