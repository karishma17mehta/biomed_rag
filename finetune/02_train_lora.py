"""
finetune/02_train_lora.py

LoRA fine-tune Mistral-7B-Instruct-v0.3 on PubMedQA using Unsloth (QLoRA).

Requirements: CUDA GPU with ≥16 GB VRAM (RTX 3090/4090, A10, A100).
For Google Colab: Runtime → Change runtime type → A100.

Usage:
    python finetune/02_train_lora.py [--epochs 3] [--batch_size 2] [--output_dir finetune/lora-adapter]

Outputs:
    finetune/lora-adapter/   — LoRA adapter weights (≈100 MB, push to HuggingFace)
"""

import argparse
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Unsloth must be imported before transformers/trl to patch correctly
# ---------------------------------------------------------------------------
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"  # 4-bit quantised base
MAX_SEQ_LEN = 2048
LORA_R = 16          # rank — increase to 32/64 for higher quality at memory cost
LORA_ALPHA = 16      # usually equal to r
LORA_DROPOUT = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--data_dir", default="finetune/data")
    p.add_argument("--output_dir", default="finetune/lora-adapter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    p.add_argument("--lora_r", type=int, default=LORA_R)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # ------------------------------------------------------------------
    # 1. Load base model + tokenizer (4-bit QLoRA)
    # ------------------------------------------------------------------
    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        dtype=None,        # auto-detect (bf16 on Ampere+, fp16 otherwise)
        load_in_4bit=True,
    )

    # ------------------------------------------------------------------
    # 2. Attach LoRA adapters
    # ------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_r,          # keep alpha == r for stable training
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",  # 30% memory saving vs standard
        random_state=args.seed,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # 3. Dataset — messages format, tokenised via chat template
    # ------------------------------------------------------------------
    train_ds = load_dataset("json", data_files=str(data_dir / "train.jsonl"), split="train")
    print(f"Train examples: {len(train_ds):,}")

    def format_chat(batch):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in batch["messages"]
        ]
        return {"text": texts}

    train_ds = train_ds.map(format_chat, batched=True, remove_columns=["messages"])

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            fp16=not _is_bf16_supported(),
            bf16=_is_bf16_supported(),
            logging_steps=25,
            save_strategy="epoch",
            output_dir=args.output_dir,
            seed=args.seed,
            report_to="none",       # swap to "wandb" if you have a W&B account
            packing=True,           # pack short sequences for 2-3x throughput
        ),
    )

    print("\nStarting training...")
    trainer_stats = trainer.train()
    print(f"\nTraining complete. Runtime: {trainer_stats.metrics['train_runtime']:.0f}s")

    # ------------------------------------------------------------------
    # 5. Save adapter only (≈100 MB — do NOT save full merged model here)
    # ------------------------------------------------------------------
    out = Path(args.output_dir)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nLoRA adapter saved → {out}")
    print("To push to HuggingFace Hub:")
    print(f"  huggingface-cli upload <your-org>/<repo-name> {out}")


def _is_bf16_supported() -> bool:
    try:
        import torch
        return torch.cuda.is_bf16_supported()
    except Exception:
        return False


if __name__ == "__main__":
    main()
