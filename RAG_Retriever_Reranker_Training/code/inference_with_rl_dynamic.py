"""
Complete inference script using RL agent for dynamic Top_M selection.

This script:
1. Loads the trained RL agent
2. Predicts optimal Top_M for each query
3. Runs full LLM inference with dynamic passage counts
4. Evaluates and compares results with fixed Top_M baseline
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from stable_baselines3 import PPO

# Import the RL environment for feature extraction
import sys
sys.path.append('./code')
from rl_top_m import TopMSelectionEnv


def load_rl_agent(model_path: str):
    """Load trained RL agent."""
    print(f"Loading RL agent from {model_path}...")
    
    # Check if path is a directory (unzipped model) or file (.zip)
    from pathlib import Path
    model_path_obj = Path(model_path)
    
    if model_path_obj.is_dir():
        # If it's a directory, PPO expects a zip file
        # We need to look for the policy.pth inside
        print(f"  Model path is a directory, loading from extracted files...")
        
        # Stable Baselines3 can load from directory if we construct the path correctly
        # Try to load by pointing to the directory as if it were extracted
        import tempfile
        import zipfile
        import shutil
        
        # Create a temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            tmp_zip_path = tmp_zip.name
            
        # Zip the directory contents
        with zipfile.ZipFile(tmp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in model_path_obj.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(model_path_obj)
                    zipf.write(file_path, arcname)
        
        # Load from the temporary zip
        model = PPO.load(tmp_zip_path)
        
        # Clean up
        Path(tmp_zip_path).unlink()
    else:
        # It's a file, load directly
        model = PPO.load(model_path)
    
    print("✓ RL agent loaded successfully")
    return model


def load_llm(model_name: str):
    """Load LLM model and tokenizer."""
    print(f"\nLoading LLM: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("✓ LLM loaded successfully")
    return tokenizer, model


def extract_rl_features(retrieved_items: List[Dict]) -> np.ndarray:
    """Extract features for RL agent from retrieval results."""
    scores = [item['score'] for item in retrieved_items[:10]]
    
    if len(scores) == 0:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    score_diff_top2 = scores[0] - scores[1] if len(scores) > 1 else 0.0
    
    return np.array([mean_score, std_score, max_score, min_score, score_diff_top2], dtype=np.float32)


def create_prompt(query: str, passages: List[str], system_prompt: str) -> str:
    """Create prompt for LLM."""
    # Format passages
    formatted_passages = "\n".join([f"[{i+1}] {p}" for i, p in enumerate(passages)])
    
    user_prompt = f"""Question: {query}

Relevant passages:
{formatted_passages}

Answer the question directly using the information above. If you can reasonably infer the answer from the passages, provide it. Only respond "CANNOTANSWER" if there is truly no relevant information.

Answer:"""
    
    return system_prompt + "\n\n" + user_prompt


def generate_answer(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1
) -> str:
    """Generate answer using LLM."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer (remove prompt)
    answer = generated_text[len(prompt):].strip()
    
    # Clean up
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    
    return answer


def run_rl_inference(
    rl_model_path: str,
    retriever_result_file: str,
    output_file: str,
    llm_model_name: str = "MediaTek-Research/Breeze-7B-Instruct-v1_0",
    system_prompt: str = None,
    max_new_tokens: int = 128,
    batch_size: int = 1
):
    """
    Run complete inference with RL-based dynamic Top_M selection.
    """
    # Load models
    rl_agent = load_rl_agent(rl_model_path)
    tokenizer, llm_model = load_llm(llm_model_name)
    
    # Default system prompt
    if system_prompt is None:
        system_prompt = """You are a helpful question-answering assistant. Extract direct answers from the provided passages.

Guidelines:
1. Provide direct, concise answers using information from the context
2. If the context contains relevant information, answer even if not perfectly complete
3. You may reasonably infer from the context when answering
4. Only use "CANNOTANSWER" when there is truly zero relevant information"""
    
    # Load retriever results
    print(f"\nLoading retriever results from {retriever_result_file}...")
    with open(retriever_result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'records' in data:
        queries_data = data['records']
    else:
        queries_data = data
    
    print(f"✓ Loaded {len(queries_data)} queries")
    
    # Step 1: Predict Top_M for all queries
    print("\n" + "="*80)
    print("STEP 1: RL Agent predicting optimal Top_M for each query")
    print("="*80)
    
    top_m_decisions = []
    rl_features = []
    
    for query_data in tqdm(queries_data, desc="RL predictions"):
        features = extract_rl_features(query_data['retrieved'])
        rl_features.append(features.tolist())
        
        # Predict action (Top_M)
        action, _ = rl_agent.predict(features, deterministic=True)
        top_m = int(action) + 1  # Convert action to Top_M (1-indexed)
        
        top_m_decisions.append(top_m)
    
    # Statistics
    print(f"\n✓ RL Agent Top_M Statistics:")
    print(f"  Mean: {np.mean(top_m_decisions):.2f}")
    print(f"  Median: {np.median(top_m_decisions):.0f}")
    print(f"  Std: {np.std(top_m_decisions):.2f}")
    
    distribution = dict(zip(*np.unique(top_m_decisions, return_counts=True)))
    print(f"  Distribution:")
    for k in sorted(distribution.keys()):
        print(f"    Top_M={k}: {distribution[k]} queries ({100*distribution[k]/len(top_m_decisions):.1f}%)")
    
    # Step 2: Run LLM inference with dynamic Top_M
    print("\n" + "="*80)
    print("STEP 2: Running LLM inference with RL-selected Top_M")
    print("="*80)
    
    predictions = []
    
    for idx, (query_data, top_m) in enumerate(tqdm(
        zip(queries_data, top_m_decisions), 
        total=len(queries_data),
        desc="Generating answers"
    )):
        qid = query_data['qid']
        query = query_data['query']
        retrieved = query_data['retrieved']
        
        # Use RL-selected Top_M passages
        passages = [item['text'] for item in retrieved[:top_m]]
        
        # Create prompt
        prompt = create_prompt(query, passages, system_prompt)
        
        # Generate answer
        answer = generate_answer(
            tokenizer, llm_model, prompt,
            max_new_tokens=max_new_tokens
        )
        
        predictions.append({
            'qid': qid,
            'query': query,
            'retrieve': [item['pid'] for item in retrieved[:top_m]],
            'answer': answer,
            'top_m_used': top_m,
            'rl_features': rl_features[idx],
            # Add fields for compatibility with evaluation scripts
            'retrieved': retrieved,  # Full retrieved list for evaluation
            'gold_pids': query_data.get('gold_pids', []),
            'gold_answer': query_data.get('gold_answer', ''),
            'generated': answer
        })
    
    # Save results
    print(f"\n✓ Saving results to {output_file}...")
    
    # Create output in standard format for evaluation
    output_data = {
        'records': predictions,  # Use 'records' key for compatibility
        'predictions': predictions,  # Keep for backward compatibility
        'metadata': {
            'rl_model': rl_model_path,
            'llm_model': llm_model_name,
            'total_queries': len(predictions),
            'top_m_statistics': {
                'mean': float(np.mean(top_m_decisions)),
                'median': float(np.median(top_m_decisions)),
                'std': float(np.std(top_m_decisions)),
                'distribution': {int(k): int(v) for k, v in distribution.items()}
            }
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Also save RL decisions separately
    decisions_file = output_file.replace('.json', '_rl_decisions.json')
    with open(decisions_file, 'w', encoding='utf-8') as f:
        json.dump({
            'decisions': [
                {
                    'qid': q['qid'],
                    'top_m': tm,
                    'features': feat
                }
                for q, tm, feat in zip(queries_data, top_m_decisions, rl_features)
            ],
            'statistics': output_data['metadata']['top_m_statistics']
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✓ RL decisions saved to {decisions_file}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_file}")
    print(f"Next step: Evaluate using evaluation script")
    print(f"\nExample:")
    print(f"  python code/evaluate.py --result_file {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL-based dynamic Top_M inference")
    parser.add_argument("--rl_model", type=str, required=True,
                        help="Path to trained RL agent")
    parser.add_argument("--retriever_results", type=str, required=True,
                        help="Path to retriever results with reranker scores")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file for predictions")
    parser.add_argument("--llm_model", type=str, 
                        default="MediaTek-Research/Breeze-7B-Instruct-v1_0",
                        help="LLM model name")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (currently only supports 1)")
    
    args = parser.parse_args()
    
    run_rl_inference(
        args.rl_model,
        args.retriever_results,
        args.output,
        args.llm_model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size
    )
