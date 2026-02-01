import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from torch.utils.data import DataLoader
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to trained paragraph selection model (e.g., ./mc_output)")
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--context_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="./selected_test.json")
    parser.add_argument("--max_seq_length", type=int, default=512)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 載入訓練好的模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_path)
    model.eval()
    
    # 載入測試數據和context
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    with open(args.context_file, 'r', encoding='utf-8') as f:
        contexts = json.load(f)
    
    # 處理每個測試樣本
    selected_test_data = []
    
    for example in test_data:
        question = example["question"]
        paragraph_ids = example["paragraphs"]
        
        # 準備多選題格式的輸入
        questions = [question] * len(paragraph_ids)
        paragraphs = []
        for pid in paragraph_ids:
            try:
                paragraphs.append(contexts[int(pid)])
            except:
                paragraphs.append("")
        
        # Tokenize
        inputs = tokenizer(
            questions,
            paragraphs,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Reshape for multiple choice
        num_choices = len(paragraph_ids)
        input_ids = inputs["input_ids"].view(1, num_choices, -1)
        attention_mask = inputs["attention_mask"].view(1, num_choices, -1)
        
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        if "token_type_ids" in inputs:
            model_inputs["token_type_ids"] = inputs["token_type_ids"].view(1, num_choices, -1)
        
        # 預測最佳段落
        with torch.no_grad():
            outputs = model(**model_inputs)
            predicted_choice = outputs.logits.argmax(dim=-1).item()
        
        # 創建新的測試樣本，格式與原始test.json相同，但只包含選中的段落
        best_paragraph_id = paragraph_ids[predicted_choice]
        
        selected_example = {
            "id": example["id"],
            "question": example["question"],
            "paragraphs": [best_paragraph_id],  # 只保留最佳段落ID
            "relevant": best_paragraph_id       # 添加relevant欄位指向最佳段落
        }
        
        selected_test_data.append(selected_example)
    
    # 儲存結果，格式與test.json相同但已篩選段落
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(selected_test_data, f, ensure_ascii=False, indent=2)
    
    print(f"Selected test data saved to {args.output_file}")
    print(f"Processed {len(selected_test_data)} examples")
    
    # 顯示一些統計資訊
    original_total_paragraphs = sum(len(ex["paragraphs"]) for ex in test_data)
    selected_total_paragraphs = len(selected_test_data)  # 每個例子現在只有1個段落
    
    print(f"Original total paragraphs: {original_total_paragraphs}")
    print(f"Selected total paragraphs: {selected_total_paragraphs}")
    print(f"Reduction ratio: {selected_total_paragraphs/original_total_paragraphs:.2%}")

if __name__ == "__main__":
    main()