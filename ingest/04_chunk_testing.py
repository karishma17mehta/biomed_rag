import json
import re
import random
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
CHUNKS_PATH = Path("outputs/chunks_filtered_v2.jsonl")  # <- change to chunks_final.jsonl if you created it
OUT_CSV = Path("outputs/chunk_quality_report.csv")
RANDOM_SEED = 7
SAMPLE_N_GOOD = 12

# -----------------------------
# Helpers
# -----------------------------
WS_RE = re.compile(r"\s+")
DNA_RUN_RE = re.compile(r"\b[ACGT]{12,}\b")
RSID_RE = re.compile(r"\brs\d{3,}\b", re.IGNORECASE)
URL_RE = re.compile(r"(www\.|https?://)", re.IGNORECASE)
TABLE_SIG_RE = re.compile(r"(table\s*\d|figure\s*\d|t00\d|journal\w+|dbcond\d{6,})", re.IGNORECASE)

def norm_space(s: str) -> str:
    return WS_RE.sub(" ", (s or "")).strip()

def space_ratio(text: str) -> float:
    t = text or ""
    return t.count(" ") / max(len(t), 1)

def digit_ratio(text: str) -> float:
    t = text or ""
    digits = sum(1 for c in t if c.isdigit())
    return digits / max(len(t), 1)

def upper_ratio(text: str) -> float:
    t = text or ""
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 0.0
    upp = sum(1 for c in letters if c.isupper())
    return upp / len(letters)

def tokenish_count(text: str) -> int:
    # cheap token proxy (good enough for quality)
    return len(re.findall(r"\w+|[^\w\s]", text or ""))

def sentence_count(text: str) -> int:
    # very rough
    return len(re.findall(r"[.!?]", text or ""))

def bullet_count(text: str) -> int:
    return len(re.findall(r"(^|\s)[•\-\u2022]\s", text or ""))

def keyword_soup_score(text: str) -> float:
    t = norm_space(text)
    if not t:
        return 0.0

    # remove your prefix if present (so it doesn't bias scoring)
    t2 = re.sub(r"^\[Cancer:.*?\]\s*Key idea:\s*", "", t)

    words = re.findall(r"[A-Za-z]{3,}", t2)
    if not words:
        return 5.0

    sc = sentence_count(t2)
    sr = space_ratio(t2)
    dr = digit_ratio(t2)

    # Heuristics:
    # - low sentence count
    # - low space ratio
    # - high digit ratio
    # - lots of separators
    seps = sum(1 for c in t2 if c in ",;:/|")
    sep_rate = seps / max(len(t2), 1)

    score = 0.0
    if sc <= 1: score += 1.0
    if sr < 0.11: score += 1.0
    if dr > 0.18: score += 1.0
    if sep_rate > 0.015: score += 1.0

    return score

def is_dna_or_genotype(text: str) -> bool:
    t = text or ""
    return bool(RSID_RE.search(t)) or (len(DNA_RUN_RE.findall(t.upper())) >= 2)

def is_table_or_header_junk(text: str) -> bool:
    t = text or ""
    sr = space_ratio(t)
    dr = digit_ratio(t)
    ur = upper_ratio(t)
    # heuristics: low spaces + numeric heavy OR has known table signatures/urls
    if sr < 0.11 and dr > 0.12:
        return True
    if URL_RE.search(t):
        return True
    if TABLE_SIG_RE.search(t):
        return True
    # extremely uppercase + numeric -> often codes/headers
    if ur > 0.85 and dr > 0.10:
        return True
    return False

def answerability_score(text: str) -> float:
    t = norm_space(text)
    if not t:
        return 0.0
    tok = tokenish_count(t)
    sc = sentence_count(t)
    soup = keyword_soup_score(t)

    score = 0.0
    if tok >= 150: score += 1.0
    if tok <= 800: score += 0.5
    if sc >= 2: score += 1.0
    if soup >= 3.0: score -= 2.0
    if is_table_or_header_junk(t): score -= 3.0
    return score

def fingerprint(text: str) -> str:
    t = norm_space(text).lower()
    t = re.sub(r"[^a-z0-9 ]+", "", t)
    t = t[:400]
    return hashlib.md5(t.encode("utf-8")).hexdigest()

# -----------------------------
# Main
# -----------------------------
def main():
    random.seed(RANDOM_SEED)
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(CHUNKS_PATH)

    rows = []
    fps = []

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("chunk_text") or obj.get("content") or ""
            text = str(text)

            sec = (obj.get("section") or "UNKNOWN").strip()
            pid = obj.get("paper_id") or obj.get("doc_id") or obj.get("paper") or ""
            cid = obj.get("chunk_id") or obj.get("id") or ""

            sr = space_ratio(text)
            dr = digit_ratio(text)
            ur = upper_ratio(text)

            row = {
                "paper_id": pid,
                "chunk_id": cid,
                "section": sec,
                "len_chars": len(text),
                "tokens_proxy": tokenish_count(text),
                "sentences": sentence_count(text),
                "bullets": bullet_count(text),
                "space_ratio": sr,
                "digit_ratio": dr,
                "upper_ratio": ur,
                "keyword_soup": keyword_soup_score(text),
                "is_table_junk": is_table_or_header_junk(text),
                "is_dna_genotype": is_dna_or_genotype(text),
                "answerability": answerability_score(text),
                "preview": norm_space(text)[:320],
            }

            fp = fingerprint(text)
            fps.append(fp)
            row["fingerprint"] = fp
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print("✅ Saved report:", OUT_CSV)

    # Dupes
    fp_counts = Counter(fps)
    dupe_fps = {fp: c for fp, c in fp_counts.items() if c >= 3}
    dupe_chunks = sum(c for c in dupe_fps.values())
    print("\n=== DUPLICATION ===")
    print("Chunks:", len(df))
    print("Fingerprints with >=3 repeats:", len(dupe_fps))
    print("Total chunks in those repeated groups:", dupe_chunks)

    # Global quality rates
    print("\n=== QUALITY FLAGS (GLOBAL) ===")
    print("table/header junk:", int(df["is_table_junk"].sum()), f"({100*df['is_table_junk'].mean():.2f}%)")
    print("dna/genotype:", int(df["is_dna_genotype"].sum()), f"({100*df['is_dna_genotype'].mean():.2f}%)")
    print("low space_ratio <0.11:", int((df["space_ratio"] < 0.11).sum()), f"({100*(df['space_ratio']<0.11).mean():.2f}%)")
    print("keyword_soup > 2.0:", int((df["keyword_soup"] > 2.0).sum()), f"({100*(df['keyword_soup']>2.0).mean():.2f}%)")
    print("answerability < 1.0:", int((df["answerability"] < 1.0).sum()), f"({100*(df['answerability']<1.0).mean():.2f}%)")

    # Section stats
    print("\n=== SECTION BREAKDOWN (top 12) ===")
    sec_stats = (
        df.groupby("section")
          .agg(n=("chunk_id","count"),
               tok_med=("tokens_proxy","median"),
               junk_rate=("is_table_junk","mean"),
               dna_rate=("is_dna_genotype","mean"),
               ans_med=("answerability","median"))
          .sort_values("n", ascending=False)
          .head(12)
    )
    print(sec_stats)

    # Show top repeated previews
    if dupe_fps:
        print("\n=== TOP REPEATED CHUNKS (preview) ===")
        top = sorted(dupe_fps.items(), key=lambda x: x[1], reverse=True)[:8]
        for fp, c in top:
            ex = df[df["fingerprint"] == fp].iloc[0]
            print(f"- repeats={c} section={ex['section']} preview={ex['preview'][:220]}")

    # Random “good chunk” samples
    good = df[
        (~df["is_table_junk"]) &
        (~df["is_dna_genotype"]) &
        (df["answerability"] >= 1.5) &
        (df["tokens_proxy"].between(180, 650))
    ]
    print("\n=== RANDOM GOOD CHUNK SAMPLES ===")
    if len(good) == 0:
        print("No chunks matched the 'good' criteria (may be too strict).")
    else:
        for _, r in good.sample(min(SAMPLE_N_GOOD, len(good)), random_state=RANDOM_SEED).iterrows():
            print(f"- {r['paper_id']} | {r['section']} | tokens≈{int(r['tokens_proxy'])} | sentences={int(r['sentences'])}")
            print("  " + r["preview"])
            print()

if __name__ == "__main__":
    main()