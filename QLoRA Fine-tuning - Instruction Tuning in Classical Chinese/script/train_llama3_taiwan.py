#!/usr/bin/env python3
"""
Llama3-Taiwan Training Script
使用 QLoRA 對 Llama3-Taiwan-8B 進行微調
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import json
import argparse
from datasets import Dataset
import os
from utils import get_prompt, get_prompt_few_shot

# single-batch forward + backward smoke test (放在 trainer.train() 之前)
from torch.utils.data import DataLoader
import time, torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# === 新增：helper 函式（整合自 assistant 的建議） ===
def _gpu_supports_bf16() -> bool:
    """檢查 CUDA GPU 是否支援 bfloat16（若沒有 GPU 則回 False）。"""
    if not torch.cuda.is_available():
        return False
    is_bf16 = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16):
        try:
            return torch.cuda.is_bf16_supported()
        except Exception:
            pass
    return False


def get_bnb_config(use_4bit: bool = True, use_8bit: bool = False) -> BitsAndBytesConfig:
    """
    回傳適當的 BitsAndBytesConfig。
    預設使用 4-bit（nf4）；若環境不支援或想切 8-bit，可改參數。
    """
    if use_4bit:
        compute_dtype = torch.bfloat16 if _gpu_supports_bf16() else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    elif use_8bit:
        # 8-bit config（視 bitsandbytes / transformers 版本而定）
        return BitsAndBytesConfig(
            load_in_4bit=False,
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        return None
# === end helper ===


def get_lora_config(r: int = 64, alpha: int = 64, dropout: float = 0.1):
    """獲取LoRA配置"""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

def prepare_dataset(data_path: str, tokenizer, max_length: int = 192):
    """準備數據集 — 修正版：確保 batch 裡有正確 padding/truncation，labels 形狀正確"""
    print(f"Loading data from: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_dict = {
        'instruction': [item['instruction'] for item in data],
        'output': [item['output'] for item in data]
    }
    dataset = Dataset.from_dict(dataset_dict)

    def tokenize_function(examples):
        texts = []
        instrs = examples["instruction"]
        outs = examples["output"]

        for instr, out in zip(instrs, outs):
            # 假設你有 get_prompt 函式在同個專案中（在 utils.py），
            # 如果沒有，請把 get_prompt 的實作貼在這裡或改用簡單拼接。
            try:
                from utils import get_prompt
            except Exception:
                # fallback 簡單 prompt
                def get_prompt(x): return f"Instruction:\n{x}\n\nResponse:\n"
            prompt = get_prompt(instr)
            full_text = prompt + out + (tokenizer.eos_token or "")
            texts.append(full_text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors=None,
            return_attention_mask=True,
        )

        tokenized["labels"] = [list(x) for x in tokenized["input_ids"]]
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=16,
        remove_columns=dataset.column_names,
        desc="Tokenizing data",
    )

    print("Sample tokenized dataset entries (first 2):")
    for i in range(min(2, len(tokenized_dataset))):
        sample = tokenized_dataset[i]
        print({k: (type(v), (len(v) if isinstance(v, list) else None)) for k, v in sample.items()})
        if "input_ids" in sample:
            print(" len(input_ids):", len(sample["input_ids"]))
        if "labels" in sample:
            print(" len(labels):", len(sample["labels"]))

    return tokenized_dataset

def train_llama3_taiwan(
    model_name: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 1,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    max_length: int = 192,
    prefer_4bit: bool = True,
    prefer_8bit: bool = False,
    force_full_fp16: bool = False,
):
    """訓練Llama3-Taiwan模型"""
    
    print(f"Training Llama3-Taiwan: {model_name}")
    print(f"Output directory: {output_dir}")
    
    # 載入分詞器
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 決定 quant config 與 dtype
    use_4bit = prefer_4bit and not force_full_fp16
    use_8bit = prefer_8bit and not force_full_fp16
    quant_config = get_bnb_config(use_4bit=use_4bit, use_8bit=use_8bit) if (use_4bit or use_8bit) else None

    # device_map 決策
    device_map = "auto" if torch.cuda.is_available() else None

    # 若使用量化，transformers 會以 quant_config 的 bnb_4bit_compute_dtype 為主。
    # 若不使用量化，則根據 GPU 支援選 dtype。
    chosen_dtype = None
    if not quant_config:
        chosen_dtype = torch.bfloat16 if _gpu_supports_bf16() else torch.float16

    print("Loading model (with quantization if configured)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=chosen_dtype,
            low_cpu_mem_usage=True,
        )
        print("Model loaded successfully.")
    except Exception as e:
        print("Primary model load failed:", e)
        print("Trying fallback: load without quantization...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=chosen_dtype,
                low_cpu_mem_usage=True,
            )
            print("Fallback (no-quant) model load successful.")
        except Exception as e2:
            print("Fallback failed:", e2)
            print("Attempting to load model on CPU as last resort (very slow)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map={"": "cpu"},
                trust_remote_code=True,
            )
            print("Model loaded on CPU.")

    # 準備模型用於 k-bit 訓練
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    
    # 準備LoRA
    print("Setting up LoRA...")
    lora_config = get_lora_config(lora_r, lora_alpha)
    model = get_peft_model(model, lora_config)
    
    # 打印可訓練參數
    model.print_trainable_parameters()
    
    # 準備數據
    print("Preparing dataset...")
    train_dataset = prepare_dataset(train_data_path, tokenizer, max_length)
    
    print(f"Training samples: {len(train_dataset)}")

    # quick collate test
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    batch_samples = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
    collated = data_collator(batch_samples)
    print("Collated keys:", collated.keys())
    try:
        print("input_ids shape:", getattr(collated["input_ids"], "shape", "n/a"))
        print("labels shape:", getattr(collated["labels"], "shape", "n/a"))
    except Exception:
        pass

    # 數據收集器（正式）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 訓練參數
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=False,
        gradient_checkpointing=False,
        max_steps=10,
        logging_steps=1,
        save_steps=500,
        save_total_limit=1,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        optim="adamw_torch",
        # lr_scheduler_type="cosine",
        report_to=None,
    )
    
    # 創建訓練器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )



    print("=== START: single-batch forward/backward smoke test ===")
    test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(test_loader))

    device = next(model.parameters()).device
    def to_tensor_if_needed(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

    # move tensors to device safely
    for k, v in batch.items():
        batch[k] = to_tensor_if_needed(v).to(device)

    print("Batch moved to device:", device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    model.train()
    try:
        out = model(**{k:v for k,v in batch.items() if k in ("input_ids","attention_mask","labels")})
        loss = out.loss if hasattr(out, "loss") else None
        print("Forward ok. Loss:", loss)
        # small backward (如果 loss 為 None 可 skip)
        if loss is not None:
            loss.backward()
            print("Backward ok.")
    except Exception as e:
        print("Exception during single-batch forward/backward:", repr(e))
        import traceback; traceback.print_exc()
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("Elapsed:", time.time()-t0)
    print("=== END: single-batch test ===")
    
    # 開始訓練
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print("Saving model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"Training completed! Model saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Train Llama3-Taiwan with QLoRA')
    parser.add_argument('--model_name', type=str, 
                       default='yentinglin/Llama-3-Taiwan-8B-Instruct',
                       help='Llama3-Taiwan model name')
    parser.add_argument('--train_data_path', type=str, default='data/train.json',
                       help='Path to training data JSON file')
    parser.add_argument('--output_dir', type=str, default='llama3_taiwan_adapter',
                       help='Output directory for trained model')
    parser.add_argument('--num_epochs', type=int, default=1,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--lora_r', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                       help='Gradient accumulation steps')
    parser.add_argument('--max_length', type=int, default=192,
                       help='Maximum sequence length')
    parser.add_argument('--prefer_4bit', action='store_true', default=True,
                       help='Prefer 4-bit quantization (default True)')
    parser.add_argument('--prefer_8bit', action='store_true', default=False,
                       help='Prefer 8-bit quantization')
    parser.add_argument('--force_full_fp16', action='store_true', default=False,
                       help='Force full fp16 (no quantization)')
    
    args = parser.parse_args()
    
    # 檢查GPU記憶體（簡單提示）
    if torch.cuda.is_available():
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {gpu_mem:.1f} GB")
            if gpu_mem < 10:
                print("Warning: Llama3-8B may require more GPU memory. Consider reducing batch size or max_length.")
        except Exception:
            pass
    
    train_llama3_taiwan(
        model_name=args.model_name,
        train_data_path=args.train_data_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        prefer_4bit=args.prefer_4bit,
        prefer_8bit=args.prefer_8bit,
        force_full_fp16=args.force_full_fp16,
    )

if __name__ == "__main__":
    main()
