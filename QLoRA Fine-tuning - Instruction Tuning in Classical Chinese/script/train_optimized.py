#!/usr/bin/env python3
"""
優化的 QLoRA Fine-tuning 腳本
目標：將 public perplexity 降到 7.2 以下
"""

import json
import torch
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import get_prompt, get_bnb_config
import argparse


def format_example_optimized(example):
    """優化的樣本格式化函數"""
    instruction = example["instruction"]
    output = example["output"]
    
    # 使用的 prompt 格式
    # prompt = f"你是古文專家。請根據以下指令完成任務。\n\n指令：{instruction}\n回答："
    prompt = get_prompt(instruction)

    # 組合 prompt 和輸出
    text = prompt + output
    
    return {"text": text}


def preprocess_function(examples, tokenizer, max_length=256):
    """預處理函數"""
    # 修復: 當 batched=True 時，examples 是字典，包含列表
    texts = examples["text"]  # 直接獲取 text 列表
    
    # Tokenize
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )
    
    # 設置 labels（用於計算 loss）
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized


def load_and_prepare_data(train_file, tokenizer, max_length=512):
    """載入和準備訓練數據"""
    print(f"📚 載入訓練數據: {train_file}")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"✅ 載入 {len(train_data)} 筆訓練數據")
    
    # 格式化數據
    formatted_data = [format_example_optimized(ex) for ex in train_data]
    
    # 創建 Dataset
    dataset = Dataset.from_list(formatted_data)
    
    # 預處理
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    print(f"🔄 預處理完成，共 {len(tokenized_dataset)} 筆數據")
    
    return tokenized_dataset


def get_optimized_lora_config():
    """獲取優化的 LoRA 配置"""
    return LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=32,  # 增加 rank
        lora_alpha=64,  # 增加 alpha
        lora_dropout=0.05,  # 降低 dropout
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )


def get_optimized_training_args(output_dir, num_train_epochs=5):
    """獲取優化的訓練參數"""
    return TrainingArguments(
        output_dir=output_dir,
        
        # 基本設置
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,  # 增加 batch size
        gradient_accumulation_steps=8,  # 使用梯度累積
        
        # 學習率設置
        learning_rate=4e-4,  # 提高學習率
        lr_scheduler_type="cosine",  # 使用 cosine scheduler
        warmup_ratio=0.1,  # 添加 warmup
        
        # 優化器設置
        optim="adamw_torch",
        weight_decay=0.01,
        adam_beta2=0.999,
        
        # 保存和評估
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=50,
        
        # 其他設置
        dataloader_drop_last=True,
        remove_unused_columns=False,
        group_by_length=True,  # 按長度分組提高效率
        
        # 混合精度
        fp16=True,
        
        # 防止過擬合
        save_safetensors=True,
        
        # 報告
        report_to=None,  # 禁用 wandb
    )


def train_optimized_model(
    base_model_name="Qwen/Qwen3-4B",  # 使用更穩定的版本
    train_file="data/train.json",
    output_dir="./optimized_adapter_checkpoint",
    max_length=512,
    num_epochs=5
):
    """訓練優化的模型"""
    
    print("🚀 開始優化的 QLoRA Fine-tuning")
    print("=" * 60)
    
    # 載入 tokenizer
    print(f"📥 載入 tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # 設置 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 載入模型
    print(f"📥 載入模型: {base_model_name}")
    bnb_config = get_bnb_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 準備模型用於訓練
    model = prepare_model_for_kbit_training(model)
    
    # 應用 LoRA
    lora_config = get_optimized_lora_config()
    model = get_peft_model(model, lora_config)
    
    print(f"🔧 LoRA 配置:")
    print(f"   - Rank (r): {lora_config.r}")
    print(f"   - Alpha: {lora_config.lora_alpha}")
    print(f"   - Dropout: {lora_config.lora_dropout}")
    print(f"   - Target modules: {lora_config.target_modules}")
    
    # 載入和準備數據
    train_dataset = load_and_prepare_data(train_file, tokenizer, max_length)
    
    # 設置訓練參數
    training_args = get_optimized_training_args(output_dir, num_epochs)
    
    print(f"🎯 訓練配置:")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   - Learning rate: {training_args.learning_rate}")
    print(f"   - Scheduler: {training_args.lr_scheduler_type}")
    
    # 數據收集器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # 創建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 開始訓練
    print(f"\n🏋️ 開始訓練...")
    trainer.train()
    
    # 保存模型
    print(f"💾 保存模型到: {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ 訓練完成！")
    print(f"📁 模型保存位置: {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="優化的 QLoRA Fine-tuning")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B", help="基礎模型名稱")
    parser.add_argument("--train_file", default="data/train.json", help="訓練數據文件")
    parser.add_argument("--output_dir", default="./optimized_adapter_checkpoint", help="輸出目錄")
    parser.add_argument("--max_length", type=int, default=256, help="最大序列長度")
    parser.add_argument("--epochs", type=int, default=3, help="訓練輪數")
    
    args = parser.parse_args()
    
    # 檢查訓練數據是否存在
    if not os.path.exists(args.train_file):
        print(f"❌ 找不到訓練數據: {args.train_file}")
        return
    
    # 開始訓練
    try:
        output_dir = train_optimized_model(
            base_model_name=args.base_model,
            train_file=args.train_file,
            output_dir=args.output_dir,
            max_length=args.max_length,
            num_epochs=args.epochs
        )
        
        print(f"\n🎯 下一步:")
        print(f"1. 測試新模型的 perplexity:")
        print(f"   python ppl.py --base_model_path {args.base_model} --peft_path {output_dir} --test_data_path data/public_test.json")
        
        print(f"2. 如果 perplexity 仍然很高，可以:")
        print(f"   - 增加 epochs: --epochs 8")
        print(f"   - 調整學習率")
        print(f"   - 使用更大的模型")
        
    except Exception as e:
        print(f"❌ 訓練失敗: {e}")
        print(f"💡 可能的解決方案:")
        print(f"   1. 檢查 GPU 記憶體是否足夠")
        print(f"   2. 減少 batch_size")
        print(f"   3. 減少 max_length")


if __name__ == "__main__":
    main()