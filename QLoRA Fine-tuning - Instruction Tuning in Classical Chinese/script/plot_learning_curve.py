# plot_learning_curve.py (step-based + training loss)
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse
import os
from tqdm import tqdm

def _tokenize_prompt_and_output(tokenizer, prompt: str, output: str, max_length: int):
    """
    Tokenize prompt and output separately, do NOT add special tokens for prompt/output.
    Append eos to output if tokenizer has eos_token_id.
    Return input_ids list, attention_mask list, prompt_len, output_len.
    Ensures truncation: prefer keeping output; if overflow, drop from prompt left.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]

    if tokenizer.eos_token_id is not None:
        output_ids = output_ids + [tokenizer.eos_token_id]

    total_len = len(prompt_ids) + len(output_ids)
    if total_len > max_length:
        overflow = total_len - max_length
        # drop from prompt left first
        if overflow >= len(prompt_ids):
            rem = overflow - len(prompt_ids)
            prompt_ids = []
            if rem >= len(output_ids):
                output_ids = output_ids[-max_length:]
            else:
                output_ids = output_ids[rem:]
        else:
            prompt_ids = prompt_ids[overflow:]

    input_ids = prompt_ids + output_ids
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, len(prompt_ids), len(output_ids)


def calculate_perplexity_for_checkpoint(model_path, adapter_path, tokenizer, test_data, max_length=2048, dtype_str="bfloat16"):
    """Calculate perplexity for a specific checkpoint using prompt-masked labels (-100)."""
    # Load model with adapter
    bnb_config = get_bnb_config()

    # Choose dtype
    torch_dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True
    )

    if os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    ppls = []

    device = next(model.parameters()).device

    for item in tqdm(test_data, desc="Calculating perplexity"):
        instruction = item["instruction"]
        output = item["output"]

        prompt = get_prompt(instruction)

        input_ids_list, attention_mask_list, prompt_len, output_len = _tokenize_prompt_and_output(
            tokenizer, prompt, output, max_length
        )

        if len(input_ids_list) < 2:
            # cannot compute causal loss for too short sequence
            continue

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.tensor([attention_mask_list], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        labels = input_ids.clone()
        if prompt_len > 0:
            labels[:, :prompt_len] = -100

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        loss = loss_fct(flat_logits, flat_labels)  # per-token loss
        mask = flat_labels != -100
        valid_count = int(mask.sum().item())
        if valid_count <= 0:
            continue

        mean_nll = loss[mask].mean().item()
        try:
            ppl = float(np.exp(mean_nll))
        except OverflowError:
            ppl = float("inf")
        ppls.append(ppl)

    try:
        del model
    except Exception:
        pass
    torch.cuda.empty_cache()

    if len(ppls) == 0:
        return float("inf")
    return float(np.mean(ppls))


def collect_trainer_loss_logs(checkpoint_dir):
    """
    Search checkpoint_dir and its immediate children for any 'trainer_state.json' files,
    parse 'log_history' and extract (step, loss) pairs. Return sorted lists (steps, losses).
    """
    all_entries = []
    # Search both root and subfolders
    candidates = []
    for root, dirs, files in os.walk(checkpoint_dir):
        if "trainer_state.json" in files:
            candidates.append(os.path.join(root, "trainer_state.json"))

    if not candidates:
        return [], []

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                tstate = json.load(f)
            log_history = tstate.get("log_history", []) or tstate.get("log_history", [])
            # log_history is usually a list of dicts with keys like {'loss': x, 'step': n}
            for entry in log_history:
                if "loss" in entry and "step" in entry:
                    # keep numeric values
                    try:
                        step = int(entry["step"])
                        loss = float(entry["loss"])
                        all_entries.append((step, loss))
                    except Exception:
                        continue
        except Exception:
            continue

    if not all_entries:
        return [], []

    # deduplicate by step keeping the last occurrence
    step_to_loss = {}
    for s, l in sorted(all_entries, key=lambda x: (x[0], x[1])):
        step_to_loss[s] = l

    steps = sorted(step_to_loss.keys())
    losses = [step_to_loss[s] for s in steps]
    return steps, losses


def plot_two_panel(steps_val, ppls, steps_loss, losses, output_path="learning_curve.png"):
    """Plot two subplots: validation perplexity (top) and training loss (bottom)."""
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True,
                            gridspec_kw={'height_ratios': [2, 1]})

    # Top: Perplexity (log scale)
    axs[0].plot(steps_val, ppls, 'o-', linewidth=2, markersize=6)
    axs[0].set_ylabel("Perplexity")
    axs[0].set_yscale("log")
    axs[0].set_title("Validation Perplexity vs Training Step")
    axs[0].grid(True, alpha=0.3)

    # Annotate some points
    for i, (s, ppl) in enumerate(zip(steps_val, ppls)):
        if i % max(1, len(steps_val)//6) == 0:
            lbl = f"{ppl:.2f}" if np.isfinite(ppl) else "inf"
            axs[0].annotate(lbl, (s, ppl), textcoords="offset points", xytext=(0,8), ha='center')

    # Bottom: Training Loss (linear)
    if steps_loss and losses:
        axs[1].plot(steps_loss, losses, 'o-', linewidth=1.5, markersize=4)
        axs[1].set_ylabel("Training Loss")
        axs[1].set_xlabel("Training Step")
        axs[1].set_title("Training Loss (from trainer_state.json log_history)")
        axs[1].grid(True, alpha=0.3)
    else:
        axs[1].text(0.5, 0.5, "No training loss logs found (searching for trainer_state.json)", ha='center', va='center', transform=axs[1].transAxes)
        axs[1].set_xlabel("Training Step")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Learning curve saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Base model name")
    parser.add_argument("--checkpoint_dir", type=str, default="./adapter_checkpoint", help="Directory containing checkpoints")
    parser.add_argument("--test_data", type=str, default="data/public_test.json", help="Test data path")
    parser.add_argument("--output_dir", type=str, default="./plots", help="Output directory for plots")
    parser.add_argument("--max_samples", type=int, default=250, help="Maximum test samples (for quick evaluation)")
    parser.add_argument("--max_length", type=int, default=2048, help="Max tokens for combined prompt+output")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="Torch dtype for model load")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    # Find checkpoints
    checkpoint_steps = []
    checkpoint_paths = []

    if os.path.exists(args.checkpoint_dir):
        for item in os.listdir(args.checkpoint_dir):
            item_path = os.path.join(args.checkpoint_dir, item)
            if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                try:
                    step = int(item.replace("checkpoint-", ""))
                    checkpoint_steps.append(step)
                    checkpoint_paths.append(item_path)
                except ValueError:
                    continue

        if checkpoint_steps:
            sorted_data = sorted(zip(checkpoint_steps, checkpoint_paths))
            checkpoint_steps, checkpoint_paths = zip(*sorted_data)
        else:
            checkpoint_steps = [0]
            checkpoint_paths = [args.checkpoint_dir]
    else:
        print(f"Checkpoint directory {args.checkpoint_dir} not found!")
        return

    print(f"Found {len(checkpoint_steps)} checkpoints: {checkpoint_steps}")

    # Evaluate perplexity for each checkpoint (step-based)
    perplexities = []
    results = []
    for step, checkpoint_path in zip(checkpoint_steps, checkpoint_paths):
        print(f"Evaluating checkpoint at step {step} ({checkpoint_path}) ...")
        try:
            ppl = calculate_perplexity_for_checkpoint(
                args.model_name, checkpoint_path, tokenizer, test_data, max_length=args.max_length, dtype_str=args.dtype
            )
            perplexities.append(ppl)
            results.append({"step": int(step), "perplexity": ppl})
            print(f"Step {step}: Perplexity = {ppl:.4f}")
        except Exception as e:
            print(f"Error evaluating checkpoint at step {step}: {e}")
            perplexities.append(float("inf"))
            results.append({"step": int(step), "perplexity": float("inf")})
            continue

    if not perplexities:
        print("No successful evaluations. Cannot plot learning curve.")
        return

    # Save validation perplexities
    results_file = os.path.join(args.output_dir, "learning_curve_data.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Validation perplexity data saved to {results_file}")

    # Collect training loss logs (search trainer_state.json)
    steps_loss, losses = collect_trainer_loss_logs(args.checkpoint_dir)
    if steps_loss and losses:
        loss_data_file = os.path.join(args.output_dir, "training_loss_data.json")
        with open(loss_data_file, 'w', encoding='utf-8') as f:
            json.dump([{"step": int(s), "loss": float(l)} for s, l in zip(steps_loss, losses)], f, indent=2, ensure_ascii=False)
        print(f"Training loss data saved to {loss_data_file}")
    else:
        print("No training loss logs found (no trainer_state.json discovered). Skipping training loss save.")

    # Plot both curves (steps are step numbers from checkpoint names)
    plot_path = os.path.join(args.output_dir, "learning_curve.png")
    plot_two_panel(list(checkpoint_steps), perplexities, steps_loss, losses, plot_path)

    # Summary
    print("\n=== LEARNING CURVE SUMMARY ===")
    finite_idxs = [i for i, v in enumerate(perplexities) if np.isfinite(v)]
    if finite_idxs:
        best_idx = finite_idxs[np.argmin([perplexities[i] for i in finite_idxs])]
        print(f"Best perplexity: {perplexities[best_idx]:.4f} at step {checkpoint_steps[best_idx]}")
    else:
        print("No finite perplexity values found.")
    print(f"Final perplexity: {perplexities[-1]:.4f} at step {checkpoint_steps[-1]}")

    if len(perplexities) > 1:
        # compute improvement among finite values if available
        if finite_idxs:
            improvement = perplexities[finite_idxs[0]] - perplexities[best_idx]
            print(f"Improvement (first finite -> best): {improvement:.4f}")
        else:
            print("Improvement: N/A (no finite perplexities)")

if __name__ == "__main__":
    main()
