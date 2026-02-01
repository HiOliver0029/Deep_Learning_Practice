#!/usr/bin/env python3
"""
Zero-Shot Evaluation Script
使用原始Qwen3-4B模型(未微調)進行零樣本評估
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
from tqdm import tqdm
from utils import get_prompt
import numpy as np
import os

def get_bnb_config():
    """獲取4-bit量化配置"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # 改為 float16
        bnb_4bit_use_double_quant=True,
        # 移除 llm_int8_enable_fp32_cpu_offload，這可能導致問題
    )

# def get_prompt(instruction: str) -> str:
#     """零樣本prompt設計"""
#     return f"你是古文專家。請根據以下指令完成任務。\n\n指令：{instruction}\n回答："

def load_model_and_tokenizer(model_name: str):
    """載入模型和分詞器"""
    print(f"Loading model: {model_name}")
    
    # 載入分詞器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 載入模型(4-bit量化)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=get_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # 與量化配置保持一致
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"❌ 量化載入失敗: {e}")
        print("🔄 嘗試不使用量化載入...")
        # 如果量化失敗，嘗試不使用量化
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    """生成回應"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 解碼回應
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(prompt):].strip()
    
    return response

def calculate_perplexity(model, tokenizer, test_data: list) -> float:
    """計算perplexity"""
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    
    for item in tqdm(test_data, desc="Calculating perplexity"):
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        # 構建完整文本
        prompt = get_prompt(instruction)
        full_text = prompt + output
        
        # 編碼
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            
            # 只計算output部分的loss
            prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            prompt_length = prompt_tokens['input_ids'].shape[1]
            
            if inputs['input_ids'].shape[1] > prompt_length:
                # 計算output部分的token數量
                output_tokens = inputs['input_ids'].shape[1] - prompt_length
                total_loss += outputs.loss.item() * output_tokens
                total_tokens += output_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity

def evaluate_zero_shot(model_name: str, test_data_path: str, output_dir: str, num_samples: int = 250):
    """零樣本評估主函數"""
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入模型
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # 載入測試數據
    print(f"Loading test data from: {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 限制樣本數量以節省時間
    if len(test_data) > num_samples:
        test_data = test_data[:num_samples]
        print(f"Using first {num_samples} samples for evaluation")
    
    # 計算perplexity
    print("Calculating perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, test_data)
    print(f"Zero-shot Perplexity: {perplexity:.4f}")
    
    # 生成一些樣本回應
    print("Generating sample responses...")
    sample_responses = []
    
    for i, item in enumerate(test_data[:10]):  # 只生成前10個樣本
        instruction = item.get('instruction', '')
        expected_output = item.get('output', '')
        
        prompt = get_prompt(instruction)
        generated_response = generate_response(model, tokenizer, prompt)
        
        sample_responses.append({
            'sample_id': i,
            'instruction': instruction,
            'expected_output': expected_output,
            'generated_response': generated_response,
            'prompt_used': prompt
        })
        
        print(f"Sample {i+1}:")
        print(f"Instruction: {instruction}")
        print(f"Generated: {generated_response}")
        print("-" * 250)
    
    # 保存結果
    results = {
        'model_name': model_name,
        'evaluation_type': 'zero_shot',
        'test_samples': len(test_data),
        'perplexity': perplexity,
        'sample_responses': sample_responses,
        'prompt_template': get_prompt("{instruction}")
    }
    
    output_file = os.path.join(output_dir, 'zero_shot_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    return perplexity, sample_responses

def main():
    parser = argparse.ArgumentParser(description='Zero-shot evaluation')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B', 
                       help='Model name or path')
    parser.add_argument('--test_data_path', type=str, required=True,
                       help='Path to test data JSON file')
    parser.add_argument('--output_dir', type=str, default='zero_shot_eval',
                       help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=250,
                       help='Number of samples to evaluate (for speed)')
    
    args = parser.parse_args()
    
    perplexity, sample_responses = evaluate_zero_shot(
        args.model_name, 
        args.test_data_path, 
        args.output_dir,
        args.num_samples
    )
    
    print(f"\n=== Zero-Shot Evaluation Results ===")
    print(f"Model: {args.model_name}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Sample responses generated: {len(sample_responses)}")

if __name__ == "__main__":
    main()