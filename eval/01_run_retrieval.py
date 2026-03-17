# eval/01_run_retrieval.py
import json
from pathlib import Path
from typing import Dict, Any, List

from app.retrieve_faiss import retrieve  # your hybrid retriever

IN_QUERIES = Path("eval/queries_ragas.jsonl")
OUT_RUN    = Path("eval/runs/retrieval_run_ragas.jsonl")

PREVIEW_CHARS = 2000  # <-- important for entity evaluation

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def save_jsonl(p: Path, rows: List[Dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    queries = load_jsonl(IN_QUERIES)
    out: List[Dict[str, Any]] = []

    for q in queries:
        query_id = q.get("id") or q.get("query_id") or ""
        query = q["query"]

        expected_cancer = (
            q.get("expected_cancer")
            or q.get("cancer_type")
            or q.get("cancer")
            or ""
        )

        hits = retrieve(
            query,
            top_n_dense=250,
            top_n_bm25=250,
            top_k=10,
            max_per_paper=1,
            max_per_hash=1,
        )

        row_intent = hits[0].get("intent") if hits else (q.get("intent") or "")

        out.append({
            "query_id": query_id,
            "query": query,
            "expected_cancer": expected_cancer,
            "intent": row_intent,
            "hits": [
                {
                    "score": h.get("score"),
                    "dense_score": h.get("dense_score"),
                    "bm25_score": h.get("bm25_score"),
                    "intent": h.get("intent"),

                    # entity rerank debug (if you included them)
                    "entity_overlap": h.get("entity_overlap"),
                    "query_hi_entities": h.get("query_hi_entities"),

                    "paper_id": h.get("paper_id"),
                    "chunk_id": h.get("chunk_id"),
                    "cancer_type": h.get("cancer_type"),
                    "section": h.get("section"),
                    "is_reference_list": h.get("is_reference_list", False),
                    "content_hash": h.get("content_hash"),

                    # store enough text for evaluation
                    "text_preview": (h.get("text") or "")[:PREVIEW_CHARS].replace("\n", " "),
                }
                for h in hits
            ],
        })

    save_jsonl(OUT_RUN, out)
    print(f"Saved: {OUT_RUN} (queries={len(out)})")

if __name__ == "__main__":
    main()