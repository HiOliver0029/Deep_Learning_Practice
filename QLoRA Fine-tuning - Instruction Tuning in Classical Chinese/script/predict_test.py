#!/usr/bin/env python3
"""
Prediction script without quantization for testing fixes
不使用量化的預測腳本，用於測試修復效果
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
from utils import get_prompt, get_prompt_few_shot

# def get_prompt(instruction: str) -> str:
#     '''Format the instruction as a prompt for LLM.'''
#     return f"你是古文專家，負責文言文與白話文的轉換。USER: {instruction} ASSISTANT:"

def remove_repetition(text, max_repeat=3):
    import re
    
    # 修復正則表達式（原本的\\1應該是\1）
    text = re.sub(r'(.)\1{' + str(max_repeat-1) + ',}', r'\1', text)
    
    # 移除重複的句子模式
    sentences = text.split('。')
    cleaned_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen_sentences:
            cleaned_sentences.append(sentence)
            seen_sentences.add(sentence)
    
    return '。'.join(cleaned_sentences)

def clean_user_assistant_output(text):
    """Clean output to remove USER/ASSISTANT artifacts"""
    import re
    
    # Remove lines that start with USER: or ASSISTANT:
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip lines that start with these patterns
        if line.startswith(('USER:', 'ASSISTANT:', 'user:', 'assistant:')):
            continue
        # Skip lines that only contain these patterns
        if line in ['USER:', 'ASSISTANT:', 'user:', 'assistant:']:
            continue
        if line:
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Remove common artifacts
    artifacts = [
        'USER:',
        'ASSISTANT:',
        'user:',
        'assistant:',
        '答案：',
        '回答：',
        '答：'
    ]
    
    for artifact in artifacts:
        # Remove artifacts at the beginning
        if text.startswith(artifact):
            text = text[len(artifact):].strip()
        # Remove artifacts that appear after spaces
        text = re.sub(r'\s+' + re.escape(artifact), '', text)
    
    return text.strip()

def generate_response(model, tokenizer, instruction, max_new_tokens=128, temperature=0.7, top_p=0.9):
    """Generate response for a single instruction."""
    # Format prompt
    prompt = get_prompt(instruction)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=50,  # 添加 top_k 限制
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3,  # 增加重複懲罰
            no_repeat_ngram_size=3,  # 禁止3字重複
            # 移除 early_stopping 以避免警告
        )
    
    # Decode only the generated part (exclude the input prompt)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Remove repetition and clean up
    response = remove_repetition(response)
    
    # Clean USER/ASSISTANT artifacts
    response = clean_user_assistant_output(response)
    
    # Stop at common ending patterns
    stop_patterns = [
        "\n\n",
        "USER:",
        "ASSISTANT:",
        "指令：",
        "回答：",
        "翻譯成古文：",
        "翻譯成白話文："
    ]
    
    for pattern in stop_patterns:
        if pattern in response:
            response = response.split(pattern)[0]
    
    return response.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B", help="Base model name")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for generation")
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum samples to process (for testing)")
    
    args = parser.parse_args()
    
    print("🚀 開始測試修復後的預測功能...")
    print(f"模型: {args.model_name}")
    print(f"輸入文件: {args.input_file}")
    print(f"輸出文件: {args.output_file}")
    print(f"最大樣本數: {args.max_samples}")
    
    # Load model and tokenizer (without quantization)
    print("📥 載入模型和分詞器...")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,  # 使用 float16 而非量化
        device_map="auto",
        trust_remote_code=True,
        use_cache=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    # Load input data
    print("📊 載入測試數據...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # 限制測試樣本數量
    if len(input_data) > args.max_samples:
        input_data = input_data[:args.max_samples]
        print(f"⚠️ 限制測試樣本為 {args.max_samples} 條")
    
    # Generate responses
    print("🎯 開始生成回答...")
    results = []
    
    for i, item in enumerate(input_data):
        print(f"處理 {i+1}/{len(input_data)}: {item['instruction'][:50]}...")
        
        instruction = item["instruction"]
        response = generate_response(
            model, tokenizer, instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print(f"   生成回答: {response[:50]}...")
        
        results.append({
            "id": item["id"],
            "output": response
        })
    
    # Save results
    print(f"💾 保存結果到 {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("✅ 測試完成！")
    
    # 簡單分析結果
    print("\n📋 結果分析:")
    for i, result in enumerate(results[:3]):  # 顯示前3個結果
        print(f"樣本 {i+1}:")
        print(f"  回答: {result['output']}")
        print(f"  長度: {len(result['output'])} 字符")
        
        # 檢查是否有問題模式
        problems = ["USER:", "ASSISTANT:", "翻譯成古文：", "李德裕、李紳、李德裕"]
        found = [p for p in problems if p in result['output']]
        if found:
            print(f"  ⚠️ 發現問題: {found}")
        else:
            print(f"  ✅ 無明顯問題")
        print()

if __name__ == "__main__":
    main()