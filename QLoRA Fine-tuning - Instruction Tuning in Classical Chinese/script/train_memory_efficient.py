#!/usr/bin/env python3
"""
記憶體優化版 QLoRA Fine-tuning 腳本（已修正版：改良 preprocess 與 data collator，labels 中 prompt token 設為 -100）
"""

import json
import torch
import os
import gc
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import get_prompt, get_prompt_few_shot, get_bnb_config
import argparse
from typing import List, Dict, Callable, Any


def setup_memory_optimization():
    """設置記憶體優化"""
    print("🔧 設置記憶體優化.")
    
    # 清理現有記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # 設置環境變數
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 設置 PyTorch 後端
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    print("✅ 記憶體優化設置完成")


def format_example_optimized(example):
    """優化的樣本格式化函數"""
    instruction = example["instruction"]
    output = example["output"]
    
    # 使用優化的 prompt 格式
    prompt = get_prompt(instruction)
    
    # 組合 prompt 和輸出（僅用於儲存原始文字）
    text = prompt + output
    
    return {"text": text, "instruction": instruction, "output": output}


# -----------------------------
# 新版 preprocess: 會分開 tokenize prompt 與 output，labels 中 prompt 設為 -100
# -----------------------------
def preprocess_function_memory_efficient(
    examples: Dict[str, List[str]],
    tokenizer,
    max_length: int = 512,
    get_prompt_fn: Callable[[str], str] = None,
    instruction_key: str = "instruction",
    output_key: str = "output",
):
    """
    將 examples（含 instruction 與 output 欄位）轉為 input_ids, attention_mask, labels。
    labels 中 prompt tokens 設為 -100，使 loss 僅計算 output 部分。
    若 prompt+output 長度超過 max_length，會優先保留 output，從 prompt 左側截斷。
    返回值為 dict of lists：{'input_ids': [...], 'attention_mask': [...], 'labels': [...]}，適合 HF Dataset.map(batched=True)
    """
    if get_prompt_fn is None:
        get_prompt_fn = get_prompt

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    instrs = examples.get(instruction_key, [])
    outs = examples.get(output_key, [])

    for instruction, output in zip(instrs, outs):
        prompt = get_prompt_fn(instruction)

        # 分別 tokenize prompt 與 output，不加入 special tokens（由我們控制）
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]

        # 在 output 結尾加 eos（若有設定）
        if tokenizer.eos_token_id is not None:
            output_ids = output_ids + [tokenizer.eos_token_id]

        # 保證不超過 max_length：優先保留 output
        total_len = len(prompt_ids) + len(output_ids)
        if total_len > max_length:
            overflow = total_len - max_length
            # 若需要刪除，先從 prompt 左側刪
            if overflow >= len(prompt_ids):
                # prompt 會被全部移除，剩下還需刪除 overflow - len(prompt_ids) tokens 從 output 左側
                prompt_ids = []
                rem = overflow - len(prompt_ids)
                if rem >= len(output_ids):
                    # degenerate: output 比 max_length 還長 -> 截取 output 的尾部
                    output_ids = output_ids[-max_length:]
                else:
                    output_ids = output_ids[rem:]
            else:
                # 一般情況：從 prompt 開頭刪除 overflow 個 token
                prompt_ids = prompt_ids[overflow:]

        input_ids = prompt_ids + output_ids
        attention_mask = [1] * len(input_ids)

        # labels: prompt -> -100, output -> token ids
        labels = [-100] * len(prompt_ids) + output_ids.copy()

        assert len(input_ids) == len(attention_mask) == len(labels)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
    }


# -----------------------------
# Data collator for causal LM: pads input_ids & attention_mask, pads labels with -100
# -----------------------------
class DataCollatorForCausalLMWithPad:
    """
    Pads a batch of dicts with keys: input_ids (list[int]), attention_mask (list[int]), labels (list[int]).
    Uses tokenizer.pad to pad input_ids & attention_mask and pads labels with -100.
    Returns PyTorch tensors.
    """
    def __init__(self, tokenizer, pad_to_multiple_of: int = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

        if self.tokenizer.pad_token_id is None:
            # 若 tokenizer 沒 pad_token，使用 eos 作為 pad
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 準備要給 tokenizer.pad 的結構
        inputs = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]

        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # labels pad
        max_len = batch["input_ids"].shape[1]
        labels = []
        for f in features:
            lab = f.get("labels", None)
            if lab is None:
                labels.append([-100] * max_len)
            else:
                lab_len = len(lab)
                if lab_len < max_len:
                    lab_padded = lab + [-100] * (max_len - lab_len)
                else:
                    lab_padded = lab[:max_len]
                labels.append(lab_padded)

        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


def get_memory_efficient_lora_config():
    """獲取記憶體高效的 LoRA 配置"""
    return LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=64,  # 減少 rank 以節省記憶體
        lora_alpha=64,  # 相應減少 alpha
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "v_proj",  # 只針對關鍵模組，減少參數量
            "o_proj", "down_proj"
        ],
        bias="none",
    )


def get_memory_efficient_training_args(output_dir, num_train_epochs=6):
    """獲取記憶體高效的訓練參數"""
    return TrainingArguments(
        output_dir=output_dir,
        
        # 基本設置
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,  # 最小 batch size
        gradient_accumulation_steps=8,  # 增加梯度累積補償
        
        # 學習率設置
        learning_rate=2e-4,  # 稍微降低學習率
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        
        # 優化器設置
        optim="adamw_torch",
        weight_decay=0.01,
        
        # 保存和評估
        save_strategy="epoch",
        save_total_limit=6,  # 減少保存的模型數量
        logging_steps=50,
        
        # 記憶體優化
        dataloader_drop_last=True,
        remove_unused_columns=False,
        group_by_length=True,
        
        # 混合精度和記憶體優化
        fp16=True,
        dataloader_pin_memory=False,  # 關閉 pin memory
        gradient_checkpointing=True,  # 開啟梯度檢查點
        
        # 其他優化
        save_safetensors=True,
        report_to=None,
    )


def load_and_prepare_data_efficient(train_file, tokenizer, max_length=512):
    """記憶體高效的數據載入"""
    print(f"📚 載入訓練數據: {train_file}")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"✅ 載入 {len(train_data)} 筆訓練數據")
    
    # 格式化數據（保留 instruction 與 output 欄位以供新版 preprocess 使用）
    formatted_data = [format_example_optimized(ex) for ex in train_data]
    
    # 創建 Dataset
    dataset = Dataset.from_list(formatted_data)
    
    # 分批預處理以節省記憶體
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function_memory_efficient(
            examples, tokenizer, max_length, get_prompt_fn=get_prompt, instruction_key="instruction", output_key="output"
        ),
        batched=True,
        batch_size=100,  # 小批次處理
        remove_columns=dataset.column_names,
    )
    
    print(f"🔄 預處理完成，共 {len(tokenized_dataset)} 筆數據")
    
    return tokenized_dataset


def train_memory_efficient_model(
    base_model_name="Qwen/Qwen3-4B",
    train_file="data/train.json",
    output_dir="./memory_efficient_adapter",
    max_length=512,
    num_epochs=6
):
    """記憶體高效的模型訓練"""
    
    print("🚀 開始記憶體優化 QLoRA Fine-tuning")
    print("=" * 60)
    
    # 設置記憶體優化
    setup_memory_optimization()
    
    # 載入 tokenizer
    print(f"📥 載入 tokenizer: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 載入模型（使用更積極的量化）
    print(f"📥 載入模型: {base_model_name}")
    bnb_config = get_bnb_config()
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 強制使用 fp16
        low_cpu_mem_usage=True,     # 降低 CPU 記憶體使用
    )
    
    # 準備模型用於訓練
    model = prepare_model_for_kbit_training(model)
    
    # 應用記憶體高效的 LoRA
    lora_config = get_memory_efficient_lora_config()
    model = get_peft_model(model, lora_config)
    
    print(f"🔧 記憶體優化 LoRA 配置:")
    print(f"   - Rank (r): {lora_config.r}")
    print(f"   - Alpha: {lora_config.lora_alpha}")
    print(f"   - Target modules: {lora_config.target_modules}")
    
    # 顯示模型參數量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 可訓練參數: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # 載入和準備數據
    train_dataset = load_and_prepare_data_efficient(train_file, tokenizer, max_length)
    
    # 設置訓練參數
    training_args = get_memory_efficient_training_args(output_dir, num_epochs)
    
    print(f"🎯 記憶體優化訓練配置:")
    print(f"   - Epochs: {num_epochs}")
    print(f"   - Batch size: {training_args.per_device_train_batch_size}")
    print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"   - Max length: {max_length}")
    print(f"   - Gradient checkpointing: {training_args.gradient_checkpointing}")
    
    # 使用自製的 causal-lm data collator（會把 labels 補 -100）
    data_collator = DataCollatorForCausalLMWithPad(tokenizer)
    
    # 創建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # 清理記憶體
    torch.cuda.empty_cache()
    gc.collect()
    
    # 開始訓練
    print(f"\n🏋️ 開始記憶體優化訓練.")
    print(f"💾 預估 GPU 記憶體需求: ~3-4 GB")
    
    try:
        trainer.train()
        
        # 保存模型
        print(f"💾 保存模型到: {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        print(f"✅ 訓練完成！")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"❌ 記憶體仍然不足！")
            print(f"💡 請嘗試以下解決方案:")
            print(f"   1. 進一步減少 batch_size 到 1")
            print(f"   2. 減少 max_length 到 128")
            print(f"   3. 減少 LoRA rank 到 8")
            print(f"   4. 手動終止其他 GPU 進程")
            raise e
        else:
            raise e
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="記憶體優化 QLoRA Fine-tuning")
    parser.add_argument("--base_model", default="Qwen/Qwen3-4B", help="基礎模型名稱")
    parser.add_argument("--train_file", default="data/train.json", help="訓練數據文件")
    parser.add_argument("--output_dir", default="./memory_efficient_adapter", help="輸出目錄")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列長度")
    parser.add_argument("--epochs", type=int, default=6, help="訓練輪數")
    
    args = parser.parse_args()
    
    # 檢查 GPU 記憶體
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔍 GPU 總記憶體: {gpu_memory:.2f} GB")
        
        if gpu_memory < 8:
            print(f"⚠️ GPU 記憶體較少，將使用超保守設置")
            args.max_length = 128
    
    # 開始訓練
    try:
        output_dir = train_memory_efficient_model(
            base_model_name=args.base_model,
            train_file=args.train_file,
            output_dir=args.output_dir,
            max_length=args.max_length,
            num_epochs=args.epochs
        )
        
        print(f"\n🎯 訓練完成！下一步:")
        print(f"1. 測試新模型的 perplexity:")
        print(f"   python ppl.py --base_model_path {args.base_model} --peft_path {output_dir} --test_data_path data/public_test.json")
        
    except Exception as e:
        print(f"❌ 訓練失敗: {e}")


if __name__ == "__main__":
    main()
