import json
from pathlib import Path

RUN_PATH = Path("eval/runs/retrieval_run_ragas.jsonl")
OUT_PATH = Path("eval/runs/ragas_input.jsonl")

def load_jsonl(p):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    rows_out = []
    for row in load_jsonl(RUN_PATH):
        q = row["query"]
        hits = row.get("hits", [])  # top-k hits
        contexts = []
        for h in hits:
            # prefer full text; fallback to preview
            t = h.get("text") or h.get("text_preview") or ""
            if t:
                contexts.append(t)

        rows_out.append({
            "question": q,
            "contexts": contexts,
            # leave answer blank for now; we'll fill it in Step 2
            "answer": "",
            # optional: if you have ground truth later
            # "ground_truth": row.get("ground_truth", "")
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")

    print(f"Saved RAGAS input JSONL: {OUT_PATH} (rows={len(rows_out)})")

if __name__ == "__main__":
    main()