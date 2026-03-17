# ingest/05_filter_chunks.py
import json
import re
from pathlib import Path

IN_PATH = Path("outputs/chunks.jsonl")
OUT_PATH = Path("outputs/chunks_filtered.jsonl")

# --- heuristics ---
DNA_RUN_RE = re.compile(r"\b[ACGT]{12,}\b")
RSID_RE = re.compile(r"\brs\d{3,}\b", re.IGNORECASE)

# --- spacing repair patterns (conservative) ---
LOWER_UPPER_RE = re.compile(r"([a-z])([A-Z])")          # highresolution -> high resolution
LETTER_DIGIT_RE = re.compile(r"([A-Za-z])(\d)")         # p53 -> p 53 (sometimes ok), keep conservative
DIGIT_LETTER_RE = re.compile(r"(\d)([A-Za-z])")         # 10mg -> 10 mg
MULTISPACE_RE = re.compile(r"\s+")

def fix_spacing(text: str) -> str:
    """
    Repair common PDF extraction "word-join" artifacts.
    Conservative: mostly inserts spaces at clear boundaries.
    """
    t = text or ""

    # Preserve newlines? For retrieval + BM25, single-space is usually fine.
    # If you want to keep paragraph breaks, do NOT collapse \n here.
    # We'll normalize all whitespace to single spaces for consistency.
    t = t.replace("\u00ad", "")  # soft hyphen
    t = t.replace("\ufb01", "fi").replace("\ufb02", "fl")  # ligatures if present

    t = LOWER_UPPER_RE.sub(r"\1 \2", t)
    t = LETTER_DIGIT_RE.sub(r"\1 \2", t)
    t = DIGIT_LETTER_RE.sub(r"\1 \2", t)

    # Fix missing spaces after punctuation: "study.This" -> "study. This"
    t = re.sub(r"([.\?!;:])([A-Za-z])", r"\1 \2", t)

    # Normalize whitespace
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t

def space_ratio(text: str) -> float:
    t = text or ""
    return t.count(" ") / max(len(t), 1)

def dna_heavy(text: str) -> bool:
    t = text or ""
    dna_runs = DNA_RUN_RE.findall(t.upper())
    return (len(dna_runs) >= 3) or bool(RSID_RE.search(t))

def uppercase_heavy(text: str) -> bool:
    t = text or ""
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return (upper / len(letters)) > 0.80

def numeric_heavy(text: str) -> bool:
    t = text or ""
    if not t:
        return False
    digits = sum(1 for c in t if c.isdigit())
    return (digits / len(t)) > 0.18

def weird_artifact(text: str) -> bool:
    t = (text or "").lower()
    # reversed / OCR garbage signatures you've observed
    if "cnoitavonn" in t:   # catches "0 cnoitavonn" and similar
        return True
    if "ie h t" in t or "ieh t" in t:
        return True
    if "0cxu" in t or "0cnoitav" in t:
        return True
    return False

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(IN_PATH)

    kept = 0
    dropped = 0
    repaired = 0

    drop_reasons = {
        "low_space": 0,
        "dna_heavy": 0,
        "upper+numeric": 0,
        "artifact": 0,
    }

    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)

            # Use 'text' if present; else fallback.
            text = obj.get("text") or obj.get("chunk_text") or obj.get("content") or ""
            text = str(text)

            fixed = fix_spacing(text)
            if fixed != text:
                repaired += 1

            # Evaluate heuristics on fixed text
            sr = space_ratio(fixed)
            dh = dna_heavy(fixed)
            uh = uppercase_heavy(fixed)
            nh = numeric_heavy(fixed)
            wa = weird_artifact(fixed)

            drop = False
            if sr < 0.10:
                drop = True
                drop_reasons["low_space"] += 1
            elif dh:
                drop = True
                drop_reasons["dna_heavy"] += 1
            elif (sr < 0.12) and uh and nh:
                drop = True
                drop_reasons["upper+numeric"] += 1
            elif wa:
                drop = True
                drop_reasons["artifact"] += 1

            if drop:
                dropped += 1
                continue

            # IMPORTANT: write back the fixed text so downstream BM25/FAISS see clean text
            obj["text"] = fixed

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print("✅ Saved:", OUT_PATH)
    print("Repaired text rows:", repaired)
    print("Kept:", kept)
    print("Dropped:", dropped)
    print("Drop reasons:", drop_reasons)

if __name__ == "__main__":
    main()