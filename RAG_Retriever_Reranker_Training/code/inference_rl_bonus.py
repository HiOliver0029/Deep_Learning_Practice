"""
Inference script using RL agent to dynamically select Top_M.

This script integrates the trained RL agent into the inference pipeline,
allowing it to decide how many passages to include for each query.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

# Import the RL environment
from rl_top_m import TopMSelectionEnv

def load_rl_agent(model_path: str):
    """Load trained RL agent."""
    print(f"Loading RL agent from {model_path}...")
    model = PPO.load(model_path)
    return model

def run_inference_with_rl(
    rl_model_path: str,
    retriever_result_file: str,
    output_file: str,
    llm_model: str = "MediaTek-Research/Breeze-7B-Instruct-v1_0",
    batch_size: int = 8
):
    """
    Run inference using RL agent to select Top_M dynamically.
    
    Args:
        rl_model_path: Path to trained RL agent
        retriever_result_file: Path to retriever results (with reranker scores)
        output_file: Output file for predictions
        llm_model: LLM model name
        batch_size: Batch size for inference
    """
    # Load RL agent
    rl_model = load_rl_agent(rl_model_path)
    
    # Load retriever results
    print(f"Loading retriever results from {retriever_result_file}...")
    with open(retriever_result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'records' in data:
        queries_data = data['records']
    else:
        queries_data = data
    
    print(f"Loaded {len(queries_data)} queries")
    
    # Initialize temporary environment for feature extraction
    temp_env = TopMSelectionEnv([queries_data[0]], max_passages=10)
    
    # Predict Top_M for each query using RL agent
    print("\nPredicting optimal Top_M for each query...")
    top_m_decisions = []
    
    for query_data in tqdm(queries_data, desc="RL predictions"):
        # Extract features from retrieval results
        scores = [item['score'] for item in query_data['retrieved'][:10]]
        
        if len(scores) == 0:
            # No retrieval results, use default
            top_m_decisions.append(3)
            continue
        
        # Create observation
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        score_diff_top2 = scores[0] - scores[1] if len(scores) > 1 else 0.0
        
        obs = np.array([mean_score, std_score, max_score, min_score, score_diff_top2], dtype=np.float32)
        
        # Predict action (Top_M)
        action, _ = rl_model.predict(obs, deterministic=True)
        top_m = int(action) + 1  # Convert action to Top_M (1-indexed)
        
        top_m_decisions.append(top_m)
    
    # Statistics about RL decisions
    print(f"\nRL Agent Top_M Statistics:")
    print(f"  Mean Top_M: {np.mean(top_m_decisions):.2f}")
    print(f"  Median Top_M: {np.median(top_m_decisions):.0f}")
    print(f"  Std Top_M: {np.std(top_m_decisions):.2f}")
    print(f"  Distribution: {dict(zip(*np.unique(top_m_decisions, return_counts=True)))}")
    
    # Save decisions for analysis
    decisions_file = output_file.replace('.json', '_rl_decisions.json')
    with open(decisions_file, 'w', encoding='utf-8') as f:
        json.dump({
            'top_m_decisions': top_m_decisions,
            'statistics': {
                'mean': float(np.mean(top_m_decisions)),
                'median': float(np.median(top_m_decisions)),
                'std': float(np.std(top_m_decisions)),
                'distribution': {int(k): int(v) for k, v in zip(*np.unique(top_m_decisions, return_counts=True))}
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\nRL decisions saved to {decisions_file}")
    
    # Now run actual inference with LLM using the predicted Top_M values
    print("\nRunning LLM inference with RL-selected Top_M...")
    
    # We need to call the original inference_batch.py but with dynamic Top_M
    # For simplicity, we'll create batches and prepare prompts here
    
    print("\nNote: To complete the inference, you need to:")
    print("1. Modify inference_batch.py to accept per-query Top_M values")
    print("2. Or run inference multiple times with different Top_M subsets")
    print(f"\nFor now, the RL decisions are saved to: {decisions_file}")
    print("\nRecommended approach:")
    print("  - Use the average Top_M from RL agent as a new fixed Top_M")
    print(f"  - Average Top_M selected by RL: {np.mean(top_m_decisions):.0f}")
    print(f"  - Most common Top_M: {np.argmax(np.bincount(top_m_decisions))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with RL-based Top_M selection")
    parser.add_argument("--rl_model", type=str, default="results/models/rl_agent",
                        help="Path to trained RL agent")
    parser.add_argument("--retriever_results", type=str, default="results/result_optimized_ultra_v2.json",
                        help="Path to retriever results")
    parser.add_argument("--output", type=str, default="results/result_rl_bonus.json",
                        help="Output file for predictions")
    parser.add_argument("--llm_model", type=str, default="MediaTek-Research/Breeze-7B-Instruct-v1_0",
                        help="LLM model name")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    
    args = parser.parse_args()
    
    run_inference_with_rl(
        args.rl_model,
        args.retriever_results,
        args.output,
        args.llm_model,
        args.batch_size
    )
