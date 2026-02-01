import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import get_prompt, get_prompt_few_shot, get_bnb_config
import argparse

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
            # early_stopping=True,  # 移除不支援的參數
        )
    
    # Decode only the generated part (exclude the input prompt)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Remove repetition and clean up
    response = remove_repetition(response)
    
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
    parser.add_argument("--base_model_path", type=str, required=True, default="Qwen/Qwen3-4B", help="Path to base model")
    parser.add_argument("--adapter_path", type=str, required=True, default="./adapter_checkpoint", help="Path to adapter")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for generation")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print("Loading base model...")
    bnb_config = get_bnb_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,  # Essential for Qwen models
        use_cache=True  # Enable cache for inference
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load adapter
    print("Loading adapter...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    
    # Load input data
    print("Loading input data...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Generate responses
    print("Generating responses...")
    results = []
    
    for i, item in enumerate(input_data):
        print(f"Processing {i+1}/{len(input_data)}")
        
        instruction = item["instruction"]
        response = generate_response(
            model, tokenizer, instruction,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        results.append({
            "id": item["id"],
            "output": response
        })
    
    # Save results
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("Generation completed!")


if __name__ == "__main__":
    main()