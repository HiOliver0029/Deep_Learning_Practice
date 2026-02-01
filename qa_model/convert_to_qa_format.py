import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paragraph_predictions", type=str, required=True,
                       help="Output from paragraph selection inference")
    parser.add_argument("--original_train", type=str, default=None)
    parser.add_argument("--original_valid", type=str, default=None)
    parser.add_argument("--original_test", type=str, default=None)
    parser.add_argument("--output_train", type=str, default="./train_qa.json")
    parser.add_argument("--output_valid", type=str, default="./valid_qa.json")
    parser.add_argument("--output_test", type=str, default="./qa_test.json")
    return parser.parse_args()

def convert_to_qa_format(original_file, output_file, use_best_paragraph=False, paragraph_predictions=None):
    """
    將原始數據轉換為QA格式
    """
    if original_file is None:
        return
        
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # 如果有段落預測結果，建立查找表
    prediction_map = {}
    if paragraph_predictions:
        with open(paragraph_predictions, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
        for pred in predictions:
            prediction_map[pred["id"]] = pred["best_paragraph_text"]
    
    qa_data = []
    for item in original_data:
        qa_entry = {
            "id": item["id"],
            "question": item["question"],
        }
        
        if use_best_paragraph and item["id"] in prediction_map:
            # 使用段落選擇的結果（測試集）
            qa_entry["context"] = prediction_map[item["id"]]
        else:
            # 使用原始的relevant段落（訓練集/驗證集）
            with open("./context.json", 'r', encoding='utf-8') as f:
                contexts = json.load(f)
            relevant_id = item.get("relevant", item["paragraphs"][0])
            qa_entry["context"] = contexts[int(relevant_id)]
        
        # 添加答案（如果有的話）
        if "answer" in item:
            qa_entry["answers"] = {
                "text": [item["answer"]["text"]],
                "answer_start": [item["answer"]["start"]]
            }
        
        qa_data.append(qa_entry)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

def main():
    args = parse_args()
    
    # 轉換訓練集（不使用段落預測，使用原始relevant）
    if args.original_train:
        convert_to_qa_format(args.original_train, args.output_train, use_best_paragraph=False)
    
    # 轉換驗證集（不使用段落預測，使用原始relevant）
    if args.original_valid:
        convert_to_qa_format(args.original_valid, args.output_valid, use_best_paragraph=False)
    
    # 轉換測試集（使用段落預測結果）
    if args.original_test:
        convert_to_qa_format(args.original_test, args.output_test, use_best_paragraph=True, 
                           paragraph_predictions=args.paragraph_predictions)
    
    print("QA format conversion completed!")

if __name__ == "__main__":
    main()