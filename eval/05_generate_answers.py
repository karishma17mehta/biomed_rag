import os, json
from pathlib import Path
from openai import OpenAI

IN_PATH  = Path("eval/runs/ragas_input.jsonl")
OUT_PATH = Path("eval/runs/ragas_ready.jsonl")

MODEL = "gpt-4o-mini"  # cheap + decent; change if you want
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM = """You are a biomedical RAG assistant.
Answer using ONLY the provided contexts.
If contexts do not contain enough evidence, say: "Insufficient evidence in provided documents."
Keep the answer concise and factual.
"""

def load_jsonl(p):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    out = []
    for row in load_jsonl(IN_PATH):
        q = row["question"]
        ctxs = row["contexts"]

        # format contexts for the model
        ctx_block = "\n\n".join([f"[Context {i+1}]\n{c}" for i, c in enumerate(ctxs[:8])])

        prompt = f"""Question: {q}

Contexts:
{ctx_block}

Write an answer. If you use a claim, it must be supported by the contexts.
"""

        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )

        ans = resp.choices[0].message.content.strip()

        row["answer"] = ans
        out.append(row)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    print(f"Saved: {OUT_PATH} (rows={len(out)})")

if __name__ == "__main__":
    main()