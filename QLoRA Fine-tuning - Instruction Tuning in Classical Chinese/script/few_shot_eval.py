#!/usr/bin/env python3
"""
Few-Shot Evaluation Script
使用原始Qwen3-4B模型進行少樣本學習評估
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
from tqdm import tqdm
import numpy as np
import os
import random
from utils import get_prompt_few_shot, get_bnb_config

# def get_bnb_config():
#     """獲取4-bit量化配置"""
#     return BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )

# def get_prompt_few_shot(instruction: str, examples: list) -> str:
#     """少樣本prompt設計"""
#     prompt = "你是古文專家。以下是一些範例：\n\n"
    
#     # 添加範例
#     for i, example in enumerate(examples, 1):
#         prompt += f"範例{i}:\n"
#         prompt += f"指令：{example['instruction']}\n"
#         prompt += f"回答：{example['output']}\n\n"
    
#     # 添加當前任務
#     prompt += "現在請根據以下指令完成任務：\n"
#     prompt += f"指令：{instruction}\n"
#     prompt += "回答："
    
#     return prompt

def select_examples(train_data: list, num_examples: int, method: str = "random") -> list:
    """選擇少樣本範例"""
    if method == "random":
        return random.sample(train_data, min(num_examples, len(train_data)))
    elif method == "first":
        return train_data[:num_examples]
    else:
        # 可以添加其他選擇方法，如基於相似性的選擇
        return random.sample(train_data, min(num_examples, len(train_data)))

def load_model_and_tokenizer(model_name: str):
    """載入模型和分詞器"""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    """生成回應"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
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
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
    
    return response

def calculate_perplexity(model, tokenizer, test_data: list, examples: list) -> float:
    """計算少樣本perplexity"""
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    
    for item in tqdm(test_data, desc="Calculating few-shot perplexity"):
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        
        # 構建few-shot prompt
        # prompt = get_prompt_few_shot(instruction, examples)
        prompt = get_prompt_few_shot(instruction)
        full_text = prompt + output
        
        # 編碼
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            
            # 只計算output部分的loss
            prompt_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            prompt_length = prompt_tokens['input_ids'].shape[1]
            
            if inputs['input_ids'].shape[1] > prompt_length:
                output_tokens = inputs['input_ids'].shape[1] - prompt_length
                total_loss += outputs.loss.item() * output_tokens
                total_tokens += output_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity

def evaluate_few_shot(model_name: str, train_data_path: str, test_data_path: str, 
                     output_dir: str, num_examples: int = 3, num_samples: int = 250,
                     selection_method: str = "random"):
    """少樣本評估主函數"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入模型
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # 載入數據
    print(f"Loading train data from: {train_data_path}")
    with open(train_data_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"Loading test data from: {test_data_path}")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 選擇少樣本範例
    examples = select_examples(train_data, num_examples, selection_method)
    print(f"Selected {len(examples)} examples using {selection_method} method")
    
    # 限制測試樣本數量
    if len(test_data) > num_samples:
        test_data = test_data[:num_samples]
        print(f"Using first {num_samples} samples for evaluation")
    
    # 計算perplexity
    print("Calculating few-shot perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, test_data, examples)
    print(f"Few-shot Perplexity: {perplexity:.4f}")
    
    # 生成樣本回應
    print("Generating sample responses...")
    sample_responses = []
    
    for i, item in enumerate(test_data[:10]):
        instruction = item.get('instruction', '')
        expected_output = item.get('output', '')
        
        prompt = get_prompt_few_shot(instruction)
        generated_response = generate_response(model, tokenizer, prompt)
        
        sample_responses.append({
            'sample_id': i,
            'instruction': instruction,
            'expected_output': expected_output,
            'generated_response': generated_response,
            'prompt_used': prompt[:500] + "..." if len(prompt) > 500 else prompt  # 截斷過長的prompt
        })
        
        print(f"Sample {i+1}:")
        print(f"Instruction: {instruction}")
        print(f"Generated: {generated_response}")
        print("-" * 50)
    
    # 保存結果
    results = {
        'model_name': model_name,
        'evaluation_type': 'few_shot',
        'num_examples': num_examples,
        'selection_method': selection_method,
        'test_samples': len(test_data),
        'perplexity': perplexity,
        'examples_used': examples,
        'sample_responses': sample_responses,
        'prompt_template': "get_prompt_few_shot(instruction)"
    }
    
    output_file = os.path.join(output_dir, f'few_shot_results_{num_examples}shot.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    return perplexity, sample_responses, examples

def main():
    parser = argparse.ArgumentParser(description='Few-shot evaluation')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B',
                       help='Model name or path')
    parser.add_argument('--train_data_path', type=str, required=True, default='./data/train.json',
                       help='Path to training data JSON file')
    parser.add_argument('--test_data_path', type=str, required=True, default='./data/public_test.json',
                       help='Path to test data JSON file')
    parser.add_argument('--output_dir', type=str, default='few_shot_eval',
                       help='Output directory for results')
    parser.add_argument('--num_examples', type=int, default=3,
                       help='Number of examples for few-shot learning')
    parser.add_argument('--num_samples', type=int, default=250,
                       help='Number of test samples to evaluate')
    parser.add_argument('--selection_method', type=str, default='random',
                       choices=['random', 'first'],
                       help='Method to select few-shot examples')
    
    args = parser.parse_args()
    
    perplexity, sample_responses, examples = evaluate_few_shot(
        args.model_name,
        args.train_data_path,
        args.test_data_path,
        args.output_dir,
        args.num_examples,
        args.num_samples,
        args.selection_method
    )
    
    print(f"\n=== Few-Shot Evaluation Results ===")
    print(f"Model: {args.model_name}")
    print(f"Examples used: {args.num_examples}")
    print(f"Selection method: {args.selection_method}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Sample responses generated: {len(sample_responses)}")

if __name__ == "__main__":
    main()