import json
from pathlib import Path
import numpy as np
import faiss
from openai import OpenAI
import os

FAISS_PATH = Path("outputs/index_openai/faiss.index")
META_PATH  = Path("outputs/index_openai/meta_tagged_v2.jsonl")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "text-embedding-3-large"

def embed(q: str) -> np.ndarray:
    r = client.embeddings.create(model=MODEL, input=[q])
    v = np.array(r.data[0].embedding, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.reshape(1, -1)

def load_meta(n=5):
    metas = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            metas.append(json.loads(line))
    return metas

def main():
    index = faiss.read_index(str(FAISS_PATH))
    print("Index ntotal:", index.ntotal)

    q = "What targeted therapies are investigated and how do they work mechanistically?"
    qv = embed(q)

    scores, ids = index.search(qv, 5)
    print("Top IDs:", ids[0].tolist())
    print("Top scores:", [round(float(s), 4) for s in scores[0]])

    # print metadata for returned hits
    metas = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            metas.append(json.loads(line))
    for rank, idx in enumerate(ids[0].tolist(), 1):
        m = metas[idx]
        print(f"\n#{rank} {m['cancer_type']} | {m['section']} | {m['paper_id']}::{m['chunk_id']}")
        print(m["text"][:350].replace("\n", " ") + "...")

if __name__ == "__main__":
    main()