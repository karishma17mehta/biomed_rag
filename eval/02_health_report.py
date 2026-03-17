# eval/02_health_report.py
import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Iterable, List

def load_jsonl(p: Path) -> Iterable[Dict[str, Any]]:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

CANCER_MUST = {
  "Thyroid_Cancer": ["thyroid", "papillary", "follicular", "medullary", "anaplastic", "ptc", "ftc", "mtc", "atc"],
  "Colon_Cancer": ["colon", "colorectal", "rectal", "crc", "mcrc", "sigmoid", "cecum", "bowel"],
  "Lung_Cancer": ["lung", "pulmonary", "nsclc", "sclc", "adenocarcinoma", "squamous"],
}

def text_mentions_expected(expected: str, hit: dict) -> bool:
    text = (hit.get("text_preview") or hit.get("text") or "").lower()
    kws = CANCER_MUST.get(expected, [])
    return any(k in text for k in kws)

def hits_all_consistent(expected: str, hits: List[dict]) -> bool:
    if not expected:
        return True
    return all(text_mentions_expected(expected, h) for h in hits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to eval/runs/*.jsonl")
    ap.add_argument("--top_k", type=int, default=10, help="Evaluate only first top_k hits per query")
    args = ap.parse_args()

    run_path = Path(args.run)

    total_q = 0

    # evidence-based @1
    ev1_total = 0
    ev1_ok = 0

    # evidence-based strict @top_k (all hits mention expected)
    strict_total = 0
    strict_ok = 0

    ref_hits = 0
    total_hits = 0

    sec_ctr = Counter()
    dup_hash_total = 0
    dup_paper_total = 0

    for row in load_jsonl(run_path):
        total_q += 1
        expected = row.get("expected_cancer") or ""
        hits = (row.get("hits") or [])[: args.top_k]

        if expected:
            ev1_total += 1
            if hits and text_mentions_expected(expected, hits[0]):
                ev1_ok += 1

            strict_total += 1
            if hits and hits_all_consistent(expected, hits):
                strict_ok += 1

        hashes = []
        papers = []
        for h in hits:
            total_hits += 1
            sec_ctr[h.get("section", "UNKNOWN")] += 1
            if h.get("is_reference_list"):
                ref_hits += 1
            if h.get("content_hash"):
                hashes.append(h["content_hash"])
            if h.get("paper_id"):
                papers.append(h["paper_id"])

        dup_hash_total += (len(hashes) - len(set(hashes)))
        dup_paper_total += (len(papers) - len(set(papers)))

    print(f"Run: {run_path}")
    print(f"Queries: {total_q}")

    if ev1_total:
        print(f"Cancer evidence match @1 (keyword-based): {ev1_ok}/{ev1_total} = {ev1_ok/ev1_total:.2%}")
    if strict_total:
        print(f"Cancer evidence strict @top_k (all hits): {strict_ok}/{strict_total} = {strict_ok/strict_total:.2%}")

    print(f"Ref hits: {ref_hits}/{total_hits} = {ref_hits/total_hits:.2%}")
    print(f"Duplicate hashes within top_k (total): {dup_hash_total}")
    print(f"Duplicate papers within top_k (total): {dup_paper_total}")

    print("\nTop sections:")
    for sec, c in sec_ctr.most_common(12):
        print(f"  {sec:22s} {c}")

if __name__ == "__main__":
    main()