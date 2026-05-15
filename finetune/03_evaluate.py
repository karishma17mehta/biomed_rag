"""
finetune/03_evaluate.py

Evaluate the fine-tuned LoRA adapter on the PubMedQA labeled test set (PQA-L).

Metrics reported:
  - Accuracy          (yes/no/maybe 3-way classification)
  - Per-class F1      (macro and per-label)
  - Confusion matrix  (printed to stdout)

Usage:
    # Evaluate fine-tuned adapter
    python finetune/03_evaluate.py --adapter finetune/lora-adapter

    # Evaluate base model (zero-shot baseline)
    python finetune/03_evaluate.py --base_only

    # Compare both
    python finetune/03_evaluate.py --adapter finetune/lora-adapter --run_baseline
"""

import argparse
import json
import re
from pathlib import Path

LABELS = ["yes", "no", "maybe"]
ANSWER_RE = re.compile(r"(?i)\bAnswer:\s*(yes|no|maybe)\b")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None, help="Path to LoRA adapter dir")
    p.add_argument("--base_model", default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
    p.add_argument("--data_file", default="finetune/data/test.jsonl")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--base_only", action="store_true", help="Evaluate base model (no adapter)")
    p.add_argument("--run_baseline", action="store_true", help="Also run base model for comparison")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


def load_test_data(path: str) -> tuple[list[dict], list[str]]:
    """Returns (chat_messages_without_assistant, gold_labels)."""
    inputs, golds = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            msgs = row["messages"]
            # Gold label is embedded in the assistant turn
            assistant_content = msgs[-1]["content"]
            m = ANSWER_RE.search(assistant_content)
            gold = m.group(1).lower() if m else "maybe"
            golds.append(gold)
            # Strip assistant turn — we generate it
            inputs.append(msgs[:-1])
    return inputs, golds


def extract_prediction(generated_text: str) -> str:
    m = ANSWER_RE.search(generated_text)
    if m:
        return m.group(1).lower()
    # Fallback: first word that is a valid label
    for word in generated_text.lower().split():
        word = word.strip(".,;:")
        if word in LABELS:
            return word
    return "maybe"  # conservative default


def run_evaluation(model, tokenizer, inputs: list[dict], max_new_tokens: int) -> list[str]:
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    predictions = []

    for chat in inputs:
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        out = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=0.0,      # greedy — deterministic eval
            do_sample=False,
        )
        generated = tokenizer.decode(out[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        predictions.append(extract_prediction(generated))

    return predictions


def compute_metrics(preds: list[str], golds: list[str]) -> dict:
    n = len(golds)
    correct = sum(p == g for p, g in zip(preds, golds))
    accuracy = correct / n

    # Per-class precision, recall, F1
    metrics: dict = {"accuracy": accuracy, "n": n, "per_class": {}}
    f1s = []
    for label in LABELS:
        tp = sum(p == label and g == label for p, g in zip(preds, golds))
        fp = sum(p == label and g != label for p, g in zip(preds, golds))
        fn = sum(p != label and g == label for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        metrics["per_class"][label] = {"precision": prec, "recall": rec, "f1": f1}
    metrics["macro_f1"] = sum(f1s) / len(f1s)
    return metrics


def print_confusion(preds: list[str], golds: list[str]) -> None:
    print("\nConfusion matrix (rows=gold, cols=pred):")
    header = f"{'':>8}" + "".join(f"{l:>8}" for l in LABELS)
    print(header)
    for g in LABELS:
        row = f"{g:>8}" + "".join(
            f"{sum(p == pr and go == g for p, go in zip(preds, golds)):>8}"
            for pr in LABELS
        )
        print(row)


def print_metrics(tag: str, metrics: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  {tag}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}  ({int(metrics['accuracy']*metrics['n'])}/{metrics['n']})")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    print(f"  Per-class :")
    for label, vals in metrics["per_class"].items():
        print(f"    {label:>5}  P={vals['precision']:.3f}  R={vals['recall']:.3f}  F1={vals['f1']:.3f}")


def load_model(base_model: str, adapter: str | None):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        print(f"Loaded adapter: {adapter}")
    else:
        print("Running base model (no adapter)")
    return model, tokenizer


def main():
    args = parse_args()
    inputs, golds = load_test_data(args.data_file)
    print(f"Test examples: {len(inputs)}")

    # Fine-tuned evaluation
    if not args.base_only and args.adapter:
        model, tokenizer = load_model(args.base_model, args.adapter)
        preds = run_evaluation(model, tokenizer, inputs, args.max_new_tokens)
        metrics = compute_metrics(preds, golds)
        print_metrics(f"Fine-tuned ({Path(args.adapter).name})", metrics)
        print_confusion(preds, golds)

        # Save results
        out_path = Path(args.adapter) / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved → {out_path}")

    # Baseline (zero-shot base model)
    if args.base_only or args.run_baseline:
        model_b, tokenizer_b = load_model(args.base_model, adapter=None)
        preds_b = run_evaluation(model_b, tokenizer_b, inputs, args.max_new_tokens)
        metrics_b = compute_metrics(preds_b, golds)
        print_metrics("Base model (zero-shot)", metrics_b)
        print_confusion(preds_b, golds)


if __name__ == "__main__":
    main()
