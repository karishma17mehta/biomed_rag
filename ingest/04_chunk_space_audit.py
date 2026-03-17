# ingest/04_chunk_space_audit.py

import json
import re
import hashlib
from pathlib import Path
import numpy as np

CHUNKS_PATH = Path("outputs/chunks_filtered_v2.jsonl")

def safe_print(*args):
    print(*args, flush=True)

# -------------------------
# Tokenizer
# -------------------------
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def tok_count(s: str) -> int:
        return len(enc.encode(s or ""))
    TOKENIZER = "tiktoken(cl100k_base)"
except Exception as e:
    safe_print("⚠️ tiktoken not available, falling back to regex token count. Error:", repr(e))
    def tok_count(s: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", s or ""))
    TOKENIZER = "regex_fallback"

# -------------------------
# Merge detection rules
# -------------------------
STOPWORD_GLUE_RE = re.compile(
    r"\b[a-z]{5,}(?:of|the|and|in|to|for|with|by)[a-z]{3,}\b"
)

LOWER_RUN_RE = re.compile(r"\b[a-z]{22,}\b")
ALNUM_RUN_RE  = re.compile(r"\b[a-z0-9]{26,}\b")
CAMELISH_RE = re.compile(r"[a-z]{3,}[A-Z][a-z]{2,}")

def merged_word_score(text: str) -> float:
    t = text or ""
    tl = t.lower()

    lower_runs = LOWER_RUN_RE.findall(tl)
    alnum_runs = ALNUM_RUN_RE.findall(tl)
    glue = STOPWORD_GLUE_RE.findall(tl)
    camelish = CAMELISH_RE.findall(t)

    sr = (t.count(" ") / max(len(t), 1))

    score = 0.0
    score += 3.0 * len(alnum_runs)
    score += 2.0 * len(lower_runs)
    score += 1.5 * len(glue)
    score += 1.0 * len(camelish)

    if sr < 0.085:
        score += 2.0
    if sr < 0.070:
        score += 3.0

    return float(score)

# -------------------------
# Fingerprint for dedup
# -------------------------
def text_fingerprint(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return hashlib.md5(t.encode("utf-8")).hexdigest()

# -------------------------
# Main
# -------------------------
def main():

    safe_print("=== SPACE AUDIT START ===")
    safe_print("File:", CHUNKS_PATH.resolve())

    if not CHUNKS_PATH.exists():
        safe_print("❌ File not found.")
        return

    size = CHUNKS_PATH.stat().st_size
    safe_print("File size (bytes):", size)

    rows = []
    bad_json = 0
    missing_text = 0

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                continue

            text = obj.get("text") or obj.get("chunk_text") or obj.get("content") or ""
            text = str(text)

            if not text.strip():
                missing_text += 1

            rows.append({
                "paper_id": obj.get("paper_id") or obj.get("doc_id") or "",
                "section": obj.get("section") or "",
                "chunk_id": obj.get("chunk_id") or obj.get("id") or "",
                "tokens": tok_count(text),
                "space_ratio": text.count(" ") / max(len(text), 1),
                "merged_score": merged_word_score(text),
                "preview": re.sub(r"\s+", " ", text)[:260],
                "full_text": text
            })

    safe_print("Parsed rows:", len(rows))
    safe_print("Bad JSON lines:", bad_json)
    safe_print("Empty text rows:", missing_text)
    safe_print("Tokenizer:", TOKENIZER)

    if not rows:
        safe_print("❌ No rows parsed.")
        return

    tokens = np.array([r["tokens"] for r in rows], dtype=float)
    ms = np.array([r["merged_score"] for r in rows], dtype=float)
    sr = np.array([r["space_ratio"] for r in rows], dtype=float)

    safe_print("\n=== TOKEN STATS ===")
    safe_print("min:", int(np.min(tokens)))
    safe_print("p50:", float(np.percentile(tokens, 50)))
    safe_print("p90:", float(np.percentile(tokens, 90)))
    safe_print("p99:", float(np.percentile(tokens, 99)))
    safe_print("max:", int(np.max(tokens)))
    safe_print("mean:", float(np.mean(tokens)))

    # Thresholds
    SUSPICIOUS_MS = 6.0
    VERY_BAD_MS = 10.0
    SUSPICIOUS_SR = 0.085
    VERY_BAD_SR = 0.070

    suspicious = (ms >= SUSPICIOUS_MS) | (sr < SUSPICIOUS_SR)
    very_bad   = (ms >= VERY_BAD_MS)   | (sr < VERY_BAD_SR)

    safe_print("\n=== SPACING HEALTH ===")
    safe_print(f"Suspicious chunks: {int(suspicious.sum())} ({100*suspicious.mean():.2f}%)")
    safe_print(f"Very bad chunks: {int(very_bad.sum())} ({100*very_bad.mean():.2f}%)")

    safe_print("\n=== FLAG BREAKDOWN ===")
    safe_print("Suspicious by score only:", int((ms >= SUSPICIOUS_MS).sum()))
    safe_print("Suspicious by spacing only:", int((sr < SUSPICIOUS_SR).sum()))
    safe_print("Very bad by score only:", int((ms >= VERY_BAD_MS).sum()))
    safe_print("Very bad by spacing only:", int((sr < VERY_BAD_SR).sum()))

    # -------------------------
    # Deduplicated worst texts
    # -------------------------
    by_fp = {}

    for r in rows:
        fp = text_fingerprint(r["full_text"])
        if fp not in by_fp:
            by_fp[fp] = {
                "merged_score": r["merged_score"],
                "space_ratio": r["space_ratio"],
                "tokens": r["tokens"],
                "examples": [(r["paper_id"], r["section"], r["chunk_id"])],
                "preview": r["preview"]
            }
        else:
            by_fp[fp]["examples"].append((r["paper_id"], r["section"], r["chunk_id"]))
            by_fp[fp]["merged_score"] = max(by_fp[fp]["merged_score"], r["merged_score"])
            by_fp[fp]["space_ratio"] = min(by_fp[fp]["space_ratio"], r["space_ratio"])

    safe_print("\nUnique texts:", len(by_fp), "out of", len(rows))

    unique_rows = []
    for fp, v in by_fp.items():
        unique_rows.append({
            "merged_score": v["merged_score"],
            "space_ratio": v["space_ratio"],
            "tokens": v["tokens"],
            "n_dupes": len(v["examples"]),
            "examples": v["examples"][:5],
            "preview": v["preview"]
        })

    unique_sorted = sorted(
        unique_rows,
        key=lambda x: (-x["merged_score"], x["space_ratio"])
    )[:25]

    safe_print("\n=== WORST 25 UNIQUE TEXTS ===")
    for u in unique_sorted:
        safe_print(
            f"- merged_score={u['merged_score']:.1f} "
            f"space_ratio={u['space_ratio']:.3f} "
            f"tokens={u['tokens']} "
            f"occurrences={u['n_dupes']}"
        )
        safe_print("  examples:",
                   "; ".join([f"{p}/{s}/{cid}" for (p, s, cid) in u["examples"]]))
        safe_print("  " + u["preview"])
        safe_print()

    safe_print("=== SPACE AUDIT END ===")


if __name__ == "__main__":
    main()