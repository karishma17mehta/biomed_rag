import json
import re
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi

INDEX_DIR = Path("outputs/index_openai")
META_PATH = INDEX_DIR / "meta_tagged_v2.jsonl"
OUT_BM25 = INDEX_DIR / "bm25.pkl"

TOKEN_RE = re.compile(r"[A-Za-z0-9\-_/]+")  # keeps gene symbols, drug names, endpoints

def body_text(full_text: str) -> str:
    if not full_text:
        return ""
    parts = full_text.split("\n---\n", 1)
    return parts[1] if len(parts) == 2 else full_text

def tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())

def main():
    metas = []
    corpus_tokens = []

    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            m = json.loads(line)
            metas.append(m)
            body = body_text(m.get("text", ""))
            corpus_tokens.append(tokenize(body))

    bm25 = BM25Okapi(corpus_tokens)

    payload = {
        "bm25": bm25,
        "token_re": TOKEN_RE.pattern,
        "meta_path": str(META_PATH),
        "n_docs": len(metas),
    }

    with OUT_BM25.open("wb") as f:
        pickle.dump(payload, f)

    print(f"Saved BM25 index: {OUT_BM25}")
    print(f"Docs indexed: {len(metas)}")

if __name__ == "__main__":
    main()