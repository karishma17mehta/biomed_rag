# ingest/08_tag_reference_lists.py
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

IN_META  = Path("outputs/index_openai/meta_hashed.jsonl")
OUT_META = Path("outputs/index_openai/meta_tagged_v2.jsonl")

# ---- Tunables ----
THRESHOLD = 6          # start here; tune after inspecting stats
PRINT_TOP = 15         # show top-N suspicious chunks
MIN_WORDS_FOR_REF = 40 # avoid flagging tiny snippets too aggressively

# ---- Patterns ----
RE_URL   = re.compile(r"\bhttps?://|www\.", re.IGNORECASE)
RE_DOI   = re.compile(r"\bdoi\b|10\.\d{4,9}/\S+", re.IGNORECASE)
RE_YEAR  = re.compile(r"\b(19|20)\d{2}\b")

# PMID / PMCID style
RE_PMID  = re.compile(r"\bpmid\b|\b\d{7,8}\b", re.IGNORECASE)
RE_PMCID = re.compile(r"\bpmc\d+\b", re.IGNORECASE)

# "et al" appears frequently in reference lists
RE_ETAL  = re.compile(r"\bet\s+al\.?\b", re.IGNORECASE)

# Numbered references: "1." / "2)" / "[12]"
RE_NUMREF = re.compile(r"(^|\n)\s*(\[\d{1,3}\]|\d{1,3}[.)])\s+")
RE_BRACKET_CIT = re.compile(r"\[\d{1,3}\]")

# Common pages/volume patterns like: 64 884 892 or 98(3):123-130
RE_VOL_ISS_PG = re.compile(
    r"\b\d{1,3}\s*\(\s*\d{1,3}\s*\)\s*:\s*\d{1,4}\s*[-–]\s*\d{1,4}\b"
)
RE_PAGES = re.compile(r"\b\d{1,4}\s*[-–]\s*\d{1,4}\b")

# Author-ish: "Surname AB" / "O'Connell D" / "Van Noesel C"
RE_AUTHOR1 = re.compile(r"\b[A-Z][a-z]+(?:['-][A-Z][a-z]+)?\s+[A-Z]{1,3}\b")
RE_AUTHOR2 = re.compile(r"\b(?:van|von|de|del|di|da)\s+[A-Z][a-z]+\b", re.IGNORECASE)

# Journal cue tokens (small list, but helps)
JOURNAL_HINTS = [
    "bmj", "lancet", "ann oncol", "jama", "n engl j med", "clin oncol",
    "head neck", "esmo", "nat rev", "jco", "pediatr", "crit care",
    "cancer res", "cancer sci", "j clin oncol", "j clin pathol",
    "nature", "cell", "proc natl acad sci", "plos", "oncology",
]

def ref_score(text: str) -> int:
    if not text:
        return 0

    word_count = len(text.split())
    if word_count < MIN_WORDS_FOR_REF:
        # tiny chunks can look like refs accidentally (e.g., one citation line)
        # still allow strong signals to flag
        tiny_penalty = 2
    else:
        tiny_penalty = 0

    t_lower = text.lower()

    url_hits = len(RE_URL.findall(text))
    doi_hits = len(RE_DOI.findall(text))
    year_hits = len(RE_YEAR.findall(text))
    pmid_hits = len(RE_PMID.findall(text))
    pmcid_hits = len(RE_PMCID.findall(text))
    etal_hits = len(RE_ETAL.findall(text))
    numref_hits = len(RE_NUMREF.findall(text))
    bracket_hits = len(RE_BRACKET_CIT.findall(text))
    author_hits = len(RE_AUTHOR1.findall(text)) + len(RE_AUTHOR2.findall(text))
    journal_hits = sum(1 for j in JOURNAL_HINTS if j in t_lower)

    # "Sentence-ish" heuristic: reference lists have few real sentences but many separators
    sentence_like = text.count(". ")
    comma_like = text.count(",")
    semicolon_like = text.count(";")
    newline_like = text.count("\n")

    voliss_hits = len(RE_VOL_ISS_PG.findall(text))
    pages_hits = len(RE_PAGES.findall(text))

    score = 0

    # Strong signals
    if doi_hits >= 1: score += 4
    if url_hits >= 1: score += 2
    if pmcid_hits >= 1: score += 3
    if pmid_hits >= 2: score += 2  # many PMIDs/long numeric IDs

    # Medium signals
    if year_hits >= 3: score += 2
    if year_hits >= 6: score += 1  # extra
    if etal_hits >= 1: score += 2
    if journal_hits >= 1: score += 2
    if author_hits >= 8: score += 2
    if author_hits >= 14: score += 1  # extra
    if numref_hits >= 2: score += 2
    if bracket_hits >= 4: score += 2
    if voliss_hits >= 1: score += 2
    if pages_hits >= 4: score += 2

    # Weak structure cues (help catch “just a list of citations”)
    if newline_like >= 3 and sentence_like <= 2: score += 1
    if comma_like >= 10 and sentence_like <= 3: score += 1
    if semicolon_like >= 5: score += 1

    # If section is explicitly REFERENCES, slightly bias toward flagging
    # (we’ll apply this outside if you want; leaving it out keeps it “text-only”)

    # penalty for tiny chunks unless they have strong signals
    score = max(0, score - tiny_penalty)

    return score

def is_reference_list(text: str) -> bool:
    return ref_score(text) >= THRESHOLD

def main():
    n = 0
    flagged = 0

    # diagnostics
    sec_counts = Counter()
    sec_flagged = Counter()
    top = []  # store (score, section, paper_id, chunk_id, preview)

    with IN_META.open("r", encoding="utf-8") as fin, OUT_META.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "") or ""
            sec = (obj.get("section") or "UNKNOWN").upper()
            sec_counts[sec] += 1

            score = ref_score(text)
            flag = score >= THRESHOLD
            obj["ref_score"] = int(score)
            obj["is_reference_list"] = bool(flag)

            if flag:
                flagged += 1
                sec_flagged[sec] += 1

            # keep some top suspicious examples for inspection
            if score > 0:
                preview = text.replace("\n", " ")[:240]
                top.append((score, sec, obj.get("paper_id",""), obj.get("chunk_id",""), preview))

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1

    print(f"Tagged {n} rows")
    print(f"is_reference_list=true: {flagged} ({flagged/n*100:.2f}%)")
    print(f"Saved: {OUT_META}\n")

    print("=== FLAG RATE BY SECTION (top 15 sections) ===")
    for sec, cnt in sec_counts.most_common(15):
        fl = sec_flagged.get(sec, 0)
        pct = (fl / cnt * 100.0) if cnt else 0.0
        print(f"{sec:22s} total={cnt:5d}  flagged={fl:5d}  ({pct:5.2f}%)")

    print(f"\n=== TOP {PRINT_TOP} BY ref_score ===")
    top.sort(key=lambda x: x[0], reverse=True)
    for i, (score, sec, pid, cid, preview) in enumerate(top[:PRINT_TOP], 1):
        print(f"\n#{i} score={score} section={sec} {pid}::{cid}")
        print(preview)

if __name__ == "__main__":
    main()


