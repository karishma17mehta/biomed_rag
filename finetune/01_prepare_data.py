"""
finetune/01_prepare_data.py

Download PubMedQA from HuggingFace and emit two JSONL files:
  finetune/data/train.jsonl  — ~800 PQA-L + up to 2000 PQA-A examples
  finetune/data/test.jsonl   — PQA-L test split (~200 examples)

Each line is an OpenAI-style chat message list that unsloth's SFTTrainer
can consume directly.

Usage:
    python finetune/01_prepare_data.py [--pqa_a_limit N]
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are a biomedical research assistant. "
    "Given a research question and the body of a PubMed abstract "
    "(the conclusion is withheld), answer with exactly one word — "
    "'yes', 'no', or 'maybe' — then write the conclusion.\n\n"
    "Format your response as:\n"
    "Answer: <yes|no|maybe>\n\n"
    "Conclusion: <your conclusion>"
)


def format_context(contexts: list[str]) -> str:
    return " ".join(contexts).strip()


def build_message(question: str, context_text: str, decision: str, long_answer: str) -> dict:
    user_text = f"Question: {question}\n\nAbstract: {context_text}"
    assistant_text = f"Answer: {decision}\n\nConclusion: {long_answer.strip()}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def process_pqa_l(split_name: str) -> list[dict]:
    """Load the PQA-L labeled fold0 source split."""
    ds = load_dataset(
        "bigbio/pubmed_qa",
        name="pubmed_qa_labeled_fold0_source",
        split=split_name,
        trust_remote_code=True,
    )
    records = []
    for row in ds:
        ctx = format_context(row["context"]["contexts"])
        records.append(build_message(row["question"], ctx, row["final_decision"], row["long_answer"]))
    return records


def process_pqa_a(limit: int, seed: int = 42) -> list[dict]:
    """Load a random sample from PQA-A (artificially labeled, 211k examples)."""
    ds = load_dataset(
        "bigbio/pubmed_qa",
        name="pubmed_qa_artificial_source",
        split="train",
        trust_remote_code=True,
    )
    # PQA-A only has "yes"/"no" labels (no "maybe") — still valuable for domain adaptation
    indices = list(range(len(ds)))
    random.seed(seed)
    random.shuffle(indices)
    selected = indices[:limit]

    records = []
    for i in selected:
        row = ds[i]
        ctx = format_context(row["context"]["contexts"])
        records.append(build_message(row["question"], ctx, row["final_decision"], row["long_answer"]))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  wrote {len(records):,} examples → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pqa_a_limit",
        type=int,
        default=2000,
        help="How many PQA-A examples to mix into training (0 = PQA-L only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(__file__).parent / "data"

    print("Loading PQA-L train split...")
    train_records = process_pqa_l("train")
    print(f"  PQA-L train: {len(train_records)} examples")

    if args.pqa_a_limit > 0:
        print(f"Loading {args.pqa_a_limit:,} PQA-A examples...")
        pqa_a = process_pqa_a(args.pqa_a_limit, seed=args.seed)
        train_records.extend(pqa_a)

    random.seed(args.seed)
    random.shuffle(train_records)

    print("Loading PQA-L test split...")
    test_records = process_pqa_l("test")
    print(f"  PQA-L test:  {len(test_records)} examples")

    print("\nWriting files...")
    write_jsonl(train_records, out_dir / "train.jsonl")
    write_jsonl(test_records, out_dir / "test.jsonl")

    # Print label distribution for train
    label_counts: dict[str, int] = {}
    for r in train_records:
        ans = r["messages"][2]["content"].split("\n")[0].replace("Answer: ", "").strip()
        label_counts[ans] = label_counts.get(ans, 0) + 1
    print(f"\nTrain label distribution: {label_counts}")


if __name__ == "__main__":
    main()
