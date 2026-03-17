import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
import faiss
import tiktoken

from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
INPUT_JSONL = Path("outputs/chunks_filtered_v2.jsonl")
OUT_DIR = Path("outputs/index_openai")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FAISS = OUT_DIR / "faiss.index"
OUT_META  = OUT_DIR / "meta.jsonl"

MODEL = "text-embedding-3-large"
BATCH_SIZE = 256          # safe for throughput; adjust if needed
SLEEP_ON_ERROR = 2.0

# If you want to inject minimal metadata into embeddings (optional):
# KEEP THIS OFF initially to avoid changing the embedding space.
INJECT_TAGS = False



client = OpenAI(api_key=os.environ.get("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"))

def get_text(obj: Dict[str, Any]) -> str:
    return obj.get("text") or obj.get("chunk_text") or obj.get("content") or ""

def get_meta(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "paper_id": obj.get("paper_id") or obj.get("doc_id") or obj.get("paper") or "",
        "chunk_id": obj.get("chunk_id") or obj.get("id") or "",
        "cancer_type": obj.get("cancer_type") or obj.get("label") or "",
        "section": obj.get("section") or "",
        "tokens": obj.get("tokens"),
        # Keep raw text in meta for easy downstream use
        "text": get_text(obj),
    }

def maybe_inject_tags(meta: Dict[str, Any], text: str) -> str:
    if not INJECT_TAGS:
        return text
    # Very light touch. DO NOT dump full metadata.
    prefix = f"CancerType: {meta['cancer_type']}\nSection: {meta['section']}\n"
    return prefix + text

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Calls OpenAI embeddings endpoint with batching + basic retry.
    Embeddings endpoint supports arrays of strings as input. :contentReference[oaicite:2]{index=2}
    """
    while True:
        try:
            resp = client.embeddings.create(
                model=MODEL,
                input=texts
            )
            # OpenAI returns embeddings aligned to inputs in order
            return [d.embedding for d in resp.data]
        except Exception as e:
            print(f"[embed] error: {e} — retrying in {SLEEP_ON_ERROR}s")
            time.sleep(SLEEP_ON_ERROR)

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms

def estimate_tokens(texts: List[str]) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(t)) for t in texts)

def main():
    # 1) Load
    metas: List[Dict[str, Any]] = []
    texts: List[str] = []

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            meta = get_meta(obj)
            text = meta["text"].strip()
            if not text:
                continue
            meta["text"] = text  # store cleaned text
            metas.append(meta)
            texts.append(maybe_inject_tags(meta, text))

    print(f"Loaded chunks: {len(texts)}")

    # 2) Embed in batches
    all_vecs: List[np.ndarray] = []
    dim = None

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i:i+BATCH_SIZE]
        vecs = embed_texts(batch)
        arr = np.array(vecs, dtype=np.float32)
        if dim is None:
            dim = arr.shape[1]
        all_vecs.append(arr)

    X = np.vstack(all_vecs)
    print("Embeddings matrix:", X.shape)

    # 3) Normalize for cosine similarity + build FAISS IndexFlatIP
    Xn = l2_normalize(X)
    dim = Xn.shape[1]
    index = faiss.IndexFlatIP(dim)  
    Xn = np.ascontiguousarray(Xn.astype("float32"))
    index.add(Xn)

    faiss.write_index(index, str(OUT_FAISS))
    print("Saved FAISS:", OUT_FAISS)


    # 4) Save metadata aligned to FAISS row ids
    with OUT_META.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("Saved META:", OUT_META)
                # inside main(), after loading texts:
    print(f"Loaded chunks: {len(texts)}")
    print("Example text length:", len(texts[0]))

    tok_total = estimate_tokens(texts[:2000])  # sample to estimate
    tok_avg = tok_total / 2000
    est_total = int(tok_avg * len(texts))
    print(f"Estimated avg tokens/chunk ~ {tok_avg:.1f}")
    print(f"Estimated total tokens ~ {est_total/1e6:.2f}M")
    print(f"Estimated cost (3-large @ $0.13/M) ~ ${est_total/1e6*0.13:.2f}")


if __name__ == "__main__":
    main()