import re
import json
import hashlib
import pandas as pd
import ftfy
from pathlib import Path
from tqdm import tqdm

DATA_PATH = Path("data/biomedical.csv")
OUT_JSONL = Path("outputs/clean_papers.jsonl")
OUT_CSV   = Path("outputs/clean_papers.csv")
OUT_DUPES = Path("outputs/clean_papers_duplicates.csv")

SECTION_HEADINGS = [
    "abstract", "introduction", "background", "methods", "materials and methods",
    "results", "discussion", "conclusion", "conclusions", "references"
]

def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = ftfy.fix_text(text)
    text = text.replace("ï¬", "")

    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()

def detect_headings(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in SECTION_HEADINGS)

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))

def canonical_for_hash(text: str) -> str:
    """
    Canonical form for dedupe hashing.
    Keep it conservative so we only dedupe true duplicates.
    """
    t = text or ""
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def text_fingerprint(text: str) -> str:
    t = canonical_for_hash(text)
    return hashlib.sha1(t.encode("utf-8")).hexdigest()

def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin1")

    if "0" not in df.columns:
        raise ValueError(f"Expected cancer label column '0' not found. Found: {list(df.columns)}")
    if "a" not in df.columns:
        raise ValueError(f"Expected text column 'a' not found. Found: {list(df.columns)}")

    seen = {}  # fingerprint -> kept_paper_id
    records = []
    dupes = []

    kept_idx = 0

    for i, row in tqdm(df.iterrows(), total=len(df)):
        cancer_type = str(row["0"]).strip()
        raw_text = row["a"]

        clean_text = basic_clean(raw_text)
        fp = text_fingerprint(clean_text)

        # Drop empty/near-empty docs early (optional but helpful)
        if len(clean_text) < 200:
            dupes.append({
                "row_index": i,
                "cancer_type": cancer_type,
                "reason": "too_short",
                "fingerprint": fp,
                "kept_paper_id": "",
            })
            continue

        # Dedupe: keep first occurrence only
        if fp in seen:
            dupes.append({
                "row_index": i,
                "cancer_type": cancer_type,
                "reason": "duplicate_exact",
                "fingerprint": fp,
                "kept_paper_id": seen[fp],
            })
            continue

        paper_id = f"paper_{kept_idx:06d}"
        kept_idx += 1
        seen[fp] = paper_id

        rec = {
            "paper_id": paper_id,
            "source_row_index": int(i),
            "cancer_type": cancer_type,
            "fingerprint": fp,
            "raw_text": "" if not isinstance(raw_text, str) else raw_text,
            "clean_text": clean_text,
            "n_chars": len(clean_text),
            "n_words": word_count(clean_text),
            "has_headings": detect_headings(clean_text),
        }
        records.append(rec)

    # Save JSONL
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save CSV
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Save duplicates report
    dup_df = pd.DataFrame(dupes)
    dup_df.to_csv(OUT_DUPES, index=False, encoding="utf-8")

    print("✅ Saved:")
    print(f"- {OUT_JSONL}")
    print(f"- {OUT_CSV}")
    print(f"- {OUT_DUPES}")

    print("\n=== DEDUPE STATS ===")
    print("Original rows:", len(df))
    print("Kept unique docs:", len(out_df))
    print("Dropped rows:", len(dup_df))
    if len(dup_df) > 0:
        print("\nDropped reason counts:")
        print(dup_df["reason"].value_counts())

    print("\nKept docs by cancer_type:")
    print(out_df.groupby("cancer_type")["paper_id"].count().sort_values(ascending=False))

if __name__ == "__main__":
    main()