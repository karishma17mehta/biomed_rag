# eval/03_entity_hit_rate.py
"""
High-signal entity hit-rate evaluation for retrieval results.

Adds rank-sensitive metrics:
- query_hit@1, @3, @K
- entity_coverage@1, @3, @K (micro-average)

Keeps original metrics:
- query hit rate (any entity found in top_k)
- entity coverage (micro avg) over top_k

Usage:
  python3 -m eval.03_entity_hit_rate --run eval/runs/retrieval_run_ragas.jsonl --top_k 10
  python3 -m eval.02_health_report --run eval/runs/retrieval_run_ragas.jsonl --top_k 10
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from app.query_router import infer_intent

# -----------------------
# HIGH-SIGNAL EXTRACTORS
# -----------------------

CHECKPOINT_PAT = re.compile(r"\b(PD-?1|PD-?L1|CTLA-?4|LAG-?3|TIM-?3|TIGIT)\b", re.IGNORECASE)

PATHWAY_PAT = re.compile(
    r"\b(MAPK|PI3K/?AKT|AKT|mTOR|WNT|TGF-?β|TGF-?B|JAK/?STAT|NF-?κB|NF-?KB)\b",
    re.IGNORECASE,
)

DRUG_PAT = re.compile(
    r"\b[a-zA-Z][a-zA-Z\-]{3,40}"
    r"(?:nib|tinib|mab|zumab|ximab|umab|cept|parib|ciclib|platin|taxel|cycline)\b",
    re.IGNORECASE,
)

GENE_PAT = re.compile(r"\b[A-Z][A-Z0-9]{1,9}\b")

GENE_STOP = {
    "CI", "SD", "SE", "NS", "NA", "USA",
    "BMC", "AJCC", "WHO", "SEER",
    "DNA", "RNA", "MRNA",
    "FIG", "TABLE", "SUPP", "ETAL",
    "I", "II", "III", "IV", "V", "VI",
    "CT", "MRI", "PET", "FDG", "SUV",
    "OS", "PFS", "DFS", "RFS", "ORR", "DCR", "HR", "PK", "PD", "AUC",
}


def extract_highsignal_entities(text: str) -> List[str]:
    """
    Extract high-signal entities from a query string.
    Returns a de-duped list (genes/checkpoints/pathways uppercase, drugs lowercase).
    """
    if not text:
        return []

    ents: List[str] = []

    ents += [m.upper() for m in CHECKPOINT_PAT.findall(text)]
    ents += [m.upper() for m in PATHWAY_PAT.findall(text)]
    ents += [m.lower() for m in DRUG_PAT.findall(text)]

    for g in GENE_PAT.findall(text.upper()):
        if g in GENE_STOP:
            continue
        if re.fullmatch(r"\d+", g):
            continue
        if len(g) < 2:
            continue
        ents.append(g)

    return sorted(set(ents))


def normalize_body(hit: Dict) -> str:
    """
    Prefer full 'text' if present, else 'text_preview'. Fall back to any string-ish field.
    """
    if not hit:
        return ""
    if isinstance(hit.get("text"), str) and hit["text"].strip():
        return hit["text"]
    if isinstance(hit.get("text_preview"), str) and hit["text_preview"].strip():
        return hit["text_preview"]
    parts = []
    for k in ("title", "snippet", "chunk", "content"):
        v = hit.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v)
    return "\n".join(parts)


def entity_in_text(entity: str, text: str) -> bool:
    """
    Word-boundary match, case-insensitive.
    """
    if not entity or not text:
        return False
    pat = re.compile(rf"\b{re.escape(entity)}\b", re.IGNORECASE)
    return bool(pat.search(text))


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to retrieval_run.jsonl")
    ap.add_argument("--top_k", type=int, default=10, help="How many hits per query to evaluate")
    args = ap.parse_args()

    run_path = Path(args.run)
    rows = load_jsonl(run_path)

    # Original overall metrics (top_k)
    per_intent = defaultdict(lambda: {"queries": 0, "hit_queries": 0, "entities": 0, "entities_found": 0})
    per_query_rows = []

    total_queries = 0
    hit_queries = 0
    total_entities = 0
    total_entities_found = 0

    # Rank-sensitive metrics (@1, @3, @K)
    K = args.top_k
    eval_ks = [1, 3, K]

    hit_q_at = {k: 0 for k in eval_ks}          # queries with any entity found in top-k
    ent_total_at = {k: 0 for k in eval_ks}      # total entities across queries (micro denom)
    ent_found_at = {k: 0 for k in eval_ks}      # entities found across queries (micro num)

    per_intent_at = defaultdict(lambda: {
        1: {"queries": 0, "hit_queries": 0, "entities": 0, "entities_found": 0},
        3: {"queries": 0, "hit_queries": 0, "entities": 0, "entities_found": 0},
        K: {"queries": 0, "hit_queries": 0, "entities": 0, "entities_found": 0},
    })

    for row in rows:
        q = row.get("query", "") or ""
        hits = (row.get("hits") or [])[:K]

        # Intent resolution
        intent = row.get("intent") or row.get("intent_name")
        if not intent:
            if hits and isinstance(hits[0], dict) and hits[0].get("intent"):
                intent = hits[0]["intent"]
            else:
                intent = infer_intent(q)

        q_entities = extract_highsignal_entities(q)

        # If no high-signal entities, skip from entity metrics but still list in per-query output
        if not q_entities:
            per_query_rows.append((0.0, 0.0, intent, q, "(no high-signal entities in query)"))
            continue

        hit_texts = [normalize_body(h) for h in hits]

        def found_entities_at(k: int) -> List[str]:
            blob_k = "\n".join(hit_texts[:k])
            return [e for e in q_entities if entity_in_text(e, blob_k)]

        # Rank-sensitive accumulation
        for k in eval_ks:
            found_k = found_entities_at(k)
            query_hit_k = 1 if found_k else 0

            hit_q_at[k] += query_hit_k
            ent_total_at[k] += len(q_entities)
            ent_found_at[k] += len(found_k)

            per_intent_at[intent][k]["queries"] += 1
            per_intent_at[intent][k]["hit_queries"] += query_hit_k
            per_intent_at[intent][k]["entities"] += len(q_entities)
            per_intent_at[intent][k]["entities_found"] += len(found_k)

        # Original top_k metrics (for compatibility)
        found = found_entities_at(K)
        coverage = len(found) / max(1, len(q_entities))
        query_hit = 1 if found else 0

        total_queries += 1
        hit_queries += query_hit
        total_entities += len(q_entities)
        total_entities_found += len(found)

        per_intent[intent]["queries"] += 1
        per_intent[intent]["hit_queries"] += query_hit
        per_intent[intent]["entities"] += len(q_entities)
        per_intent[intent]["entities_found"] += len(found)

        per_query_rows.append((coverage, query_hit, intent, q, ", ".join(found) if found else "(none)"))

    # -----------------------
    # PRINT REPORT
    # -----------------------
    print("\n=== HIGH-SIGNAL ENTITY METRICS ===")
    print(f"Run file: {run_path}")
    print(f"Evaluated top_k: {K}\n")

    if total_queries == 0:
        print("No queries with high-signal entities were found. (All queries had 0 extracted entities.)")
        return

    print("=== OVERALL (TOP_K) ===")
    print(f"Queries with high-signal entities: {total_queries}")
    print(f"Query hit rate (any entity found in top_k): {hit_queries}/{total_queries} = {hit_queries/total_queries*100:.2f}%")
    print(f"Entity coverage (micro avg): {total_entities_found}/{total_entities} = {total_entities_found/total_entities*100:.2f}%")

    print("\n=== RANK-SENSITIVE OVERALL ===")
    for k in eval_ks:
        qhit = (hit_q_at[k] / total_queries * 100) if total_queries else 0.0
        ecov = (ent_found_at[k] / ent_total_at[k] * 100) if ent_total_at[k] else 0.0
        print(f"@{k:<2d}  query_hit={qhit:6.2f}%  entity_cov={ecov:6.2f}%")

    print("\n=== PER INTENT (TOP_K) ===")
    for intent, stats in sorted(per_intent.items(), key=lambda x: x[0]):
        if stats["queries"] == 0:
            continue
        qhit = stats["hit_queries"] / stats["queries"] * 100
        ecov = (stats["entities_found"] / stats["entities"] * 100) if stats["entities"] else 0.0
        print(f"{intent:20s}  query_hit={qhit:6.2f}%  entity_cov={ecov:6.2f}%  (n={stats['queries']})")

    print("\n=== RANK-SENSITIVE PER INTENT ===")
    for intent, block in sorted(per_intent_at.items(), key=lambda x: x[0]):
        for k in eval_ks:
            s = block[k]
            if s["queries"] == 0:
                continue
            qhit = s["hit_queries"] / s["queries"] * 100
            ecov = (s["entities_found"] / s["entities"] * 100) if s["entities"] else 0.0
            print(f"{intent:20s} @{k:<2d}  query_hit={qhit:6.2f}%  entity_cov={ecov:6.2f}%  (n={s['queries']})")

    print("\n=== PER QUERY (LOWEST COVERAGE FIRST, TOP_K) ===")
    per_query_rows.sort(key=lambda x: x[0])
    for cov, qhit, intent, q, found in per_query_rows:
        prefix = f"{cov:0.3f} | hit={int(qhit)} | {intent:16s}"
        q_short = (q[:90] + "...") if len(q) > 90 else q
        print(f"{prefix} | {q_short}")
        # Uncomment if you want per-query found entities printed:
        # print(f"    found: {found}")


if __name__ == "__main__":
    main()