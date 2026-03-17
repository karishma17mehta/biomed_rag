# ingest/05b_filter_chunks_spacing.py
import json
import re
from pathlib import Path

IN_PATH = Path("outputs/chunks_filtered.jsonl")
OUT_PATH = Path("outputs/chunks_filtered_v2.jsonl")

# -------------------------
# Table / header junk rules
# -------------------------

# 05b: add reference/citation junk detection
REF_CUE_RE = re.compile(
    r"(?i)\b("
    r"references|bibliography|et al|doi|pmid|arxiv|preprint|copyright|"
    r"springer|elsevier|wiley|nature\.com|sciencedirect|"
    r"vol\.?\s*\d+|issue\s*\d+|\b\d+\s*\(\d+\)\s*:\s*\d+"
    r")\b"
)

# e.g. "Taenzer P; Bultz BD; Carlson LE" or "Taenzer P, Bultz BD, Carlson LE"
AUTHOR_PAIR_WITH_SEP_RE = re.compile(
    r"(?i)\b[a-z]{2,}\s+[a-z]{1,3}(?=\s*[,;])"
)

# journal-ish abbreviations: "health qual life outcomes", "clin cancer res"
JOURNALISH_RE = re.compile(
    r"(?i)\b("
    r"health\s+qual\s+life\s+outcomes|"
    r"clin(?:ical)?\s+cancer\s+res|"
    r"j\s+clin\s+oncol|"
    r"proc\s+natl\s+acad\s+sci|"
    r"lancet|"
    r"radiother\s+oncol|"
    r"int\s+j\s+radiat"
    r")\b"
)

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

FIG_TABLE_NOISE_RE = re.compile(r"(?i)\b(fig(?:ure)?|table)\s*\d+\b")
PAGE_ARTIFACT_RE = re.compile(r"(?i)\b(page|diagnostic pathology)\b")
OCR_CODE_RE = re.compile(r"(?i)\b0\s*cli\b|\b0c[a-z]{2,}\b")  # catches "0 cli", "0cnoitav", etc.

URL_RE = re.compile(r"(www\.|https?://)", re.IGNORECASE)
CAPTION_RE = re.compile(r"(?im)^\s*(table|figure)\s*\d+\b")
ASSAY_TABLE_RE = re.compile(r"(ic50|abts|frap|trolox|equivalents|assay)", re.IGNORECASE)
PUBLISHER_HEADER_RE = re.compile(r"(scientific\s*reports|nature\.com|vol\d{3,}|issn)", re.IGNORECASE)

# Matches "surname initial(s)" in lowercase too: "taenzer p", "bultz bd"
AUTHOR_INITIALS_ANYCASE_RE = re.compile(r"(?i)\b[a-z]{2,}\s+[a-z]{1,3}\b")

# If you want extra-strong “citation vibe” cues
CITATION_VIBE_RE = re.compile(r"(?i)\b(impact of|outcomes|trial|randomi[sz]ed|meta-?analysis)\b")

def looks_like_reference_block(text: str) -> bool:
    t = text or ""
    if len(t) < 220:
        return False

    # strong cues
    if REF_CUE_RE.search(t):
        return True

    author_pairs = len(AUTHOR_PAIR_WITH_SEP_RE.findall(t))
    years = len(YEAR_RE.findall(t))

    # true reference lists usually have lots of separators
    sep_count = t.count(";") + t.count(",")

    # stricter rules (won’t trigger on normal prose)
    if author_pairs >= 8 and sep_count >= 8:
        return True
    if author_pairs >= 6 and years >= 1 and sep_count >= 6:
        return True
    if JOURNALISH_RE.search(t) and author_pairs >= 5 and sep_count >= 6:
        return True

    return False

def space_ratio(text: str) -> float:
    t = text or ""
    return t.count(" ") / max(len(t), 1)

def numeric_heavy(text: str) -> bool:
    t = text or ""
    if not t:
        return False
    digits = sum(1 for c in t if c.isdigit())
    return (digits / len(t)) > 0.18

def many_pipes_or_delims(text: str) -> bool:
    t = text or ""
    return (t.count("|") >= 6) or (t.count("\t") >= 6)

def is_table_header_junk(text: str) -> bool:
    t = text or ""
    # low space + numeric heavy + any header/table signature
    if space_ratio(t) < 0.11 and numeric_heavy(t):
        if (URL_RE.search(t) or PUBLISHER_HEADER_RE.search(t) or CAPTION_RE.search(t)):
            return True
    # assay table mash often has those keywords + low space
    if space_ratio(t) < 0.11 and ASSAY_TABLE_RE.search(t):
        return True
    # delimiter-heavy tables
    if many_pipes_or_delims(t) and space_ratio(t) < 0.13:
        return True
    # captions like "Table 2" / "Figure 1" (only if it looks non-prose)
    if CAPTION_RE.search(t) and space_ratio(t) < 0.14:
        return True
    return False

def looks_like_figure_table_dump(text: str) -> bool:
    t = text or ""
    if len(t) < 160:
        return False

    # If it contains fig/table artifacts AND is not very prose-like, drop it.
    if (FIG_TABLE_NOISE_RE.search(t) or PAGE_ARTIFACT_RE.search(t) or OCR_CODE_RE.search(t)):
        # require at least one additional "non-prose" signature
        if numeric_heavy(t) or many_pipes_or_delims(t) or space_ratio(t) < 0.14:
            return True
    return False

# -------------------------
# Biomedical-safe spacing fix
# -------------------------

# 1) Remove some recurring OCR garbage tokens (you can expand this list as you see them)
OCR_GARBAGE_PAT = re.compile(
    r"\b(0cnoitavonn|cnoitavonn|0cxu|0cli|Ieh|ehT)\b", re.IGNORECASE
)

# 2) Fix common stopword joins: "ofpemigatinib" -> "of pemigatinib"
# Only splits if BOTH sides are alphabetic and reasonably long.
STOPWORDS = [
    "of","the","and","or","to","in","for","with","from","by","as","at","on",
    "into","over","under","between","after","before","during","within","without",
    "among","through","via","per"
]
STOPWORD_JOIN_RE = re.compile(
    r"\b(" + "|".join(STOPWORDS) + r")([a-z]{4,})\b", re.IGNORECASE
)

# word+stopword joins: "behaviourand" -> "behaviour and"
WORD_STOPWORD_JOIN_RE = re.compile(
    r"\b([a-z]{4,})(of|the|and|or|in|to|for|with|by|from|as|at|on)\b",
    re.IGNORECASE
)

# 3) Biomedical suffix joins: "cellcarcinoma" -> "cell carcinoma"
# Keep this conservative; expand slowly based on audit.
SUFFIX_SPLITS = [
    ("cell", ["carcinoma","lines","line","culture","cultures"]),
    ("thyroid", ["cancer","carcinoma","tumor","tumour"]),
    ("lung", ["cancer","carcinoma"]),
    ("colon", ["cancer","carcinoma"]),
    ("breast", ["cancer","carcinoma"]),
    ("prostate", ["cancer","carcinoma"]),
    ("high", ["resolution","risk","grade"]),
    ("low", ["risk","grade"]),
    ("multi", ["omics","center","centre","drug"]),
    ("micro", ["environment","biome","biota"]),
]
def apply_suffix_splits(t: str) -> str:
    for left, rights in SUFFIX_SPLITS:
        for right in rights:
            # case-insensitive, but preserve original by just inserting a space
            pat = re.compile(rf"\b({re.escape(left)})({re.escape(right)})\b", re.IGNORECASE)
            t = pat.sub(r"\1 \2", t)
    return t

# 4) Drug boundary split: "pemigatinibis" -> "pemigatinib is"
DRUG_BOUNDARY_RE = re.compile(
    r"\b([a-zA-Z]{4,}(?:nib|mab|ciclib|parib|tinib|zumab|ximab|umab|cept))([a-z]{2,})\b",
    re.IGNORECASE,
)

# 5) Simple camel-case split (rare in your text, but safe)
CAMEL_RE = re.compile(r"([a-z])([A-Z])")

# 6) DO NOT split gene-like tokens (BRAF, KRASG12D, TP53, PD-L1)
GENEISH_RE = re.compile(r"\b[A-Z]{2,10}[0-9]{0,4}\b")
MUT_RE = re.compile(r"\b[A-Z]{2,10}\s*G\s*\d{1,4}\s*[A-Z]\b")  # loose safeguard

COMPOUND_SPLITS = [
    ("factor", ["analysis"]),
    ("quality", ["of", "life"]),  # careful: we'll handle separately below
    ("postoperative", ["pain"]),
    ("clinical", ["trial"]),
    ("chronic", ["liver", "patients"]),
    ("patient", ["satisfaction"]),
    ("health", ["outcomes"]),
    ("life", ["outcomes"]),
]

COMMON_SUFFIX_RE = re.compile(
    r"\b([a-z]{4,})(patients|analysis|outcomes|therapy|carcinoma|disease|syndrome|screening|satisfaction)\b",
    re.IGNORECASE
)

def fix_spacing(text: str) -> str:
    """
    Biomedical-safe spacing/cleanup.
    Does NOT use wordninja. Avoids breaking gene/drug tokens.
    """
    if not isinstance(text, str):
        return ""

    t = text

    # normalize weird whitespace first
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    # remove known OCR garbage tokens
    t = OCR_GARBAGE_PAT.sub(" ", t)

    # stopword joins
    t = STOPWORD_JOIN_RE.sub(r"\1 \2", t)

    t = WORD_STOPWORD_JOIN_RE.sub(r"\1 \2", t)

    # drug boundary (only adds a space, doesn't break inside the drug name)
    t = DRUG_BOUNDARY_RE.sub(r"\1 \2", t)

    # suffix splits
    t = apply_suffix_splits(t)

    t = COMMON_SUFFIX_RE.sub(r"\1 \2", t)

    # camel-case split (safe)
    t = CAMEL_RE.sub(r"\1 \2", t)

    # cleanup: spacing around punctuation
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"(\()\s+", r"\1", t)
    t = re.sub(r"\s+(\))", r"\1", t)

    # collapse multiple spaces again
    t = re.sub(r"[ \t]{2,}", " ", t).strip()

    return t

# -------------------------
# Main
# -------------------------
def main():
    kept = 0
    dropped = 0
    reasons = {"low_space": 0, "table_header_junk": 0, "reference_block": 0}

    with open(IN_PATH, "r", encoding="utf-8") as fin, open(OUT_PATH, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)

            # grab text field
            text = obj.get("text") or obj.get("chunk_text") or obj.get("content") or ""
            text = str(text)

            # 1) spacing cleanup first (so table/header detection improves)
            text_fixed = fix_spacing(text)

            # 2) drop table/header junk
            sr = space_ratio(text_fixed)

            if sr < 0.11:
                if numeric_heavy(text_fixed) or is_table_header_junk(text_fixed):
                    dropped += 1
                    reasons["low_space"] += 1
                    continue

            if is_table_header_junk(text_fixed):
                dropped += 1
                reasons["table_header_junk"] += 1
                continue

            if looks_like_reference_block(text_fixed):
                dropped += 1
                reasons["reference_block"] += 1
                continue

            if looks_like_figure_table_dump(text_fixed):
                dropped += 1
                reasons["figure_table_dump"] = reasons.get("figure_table_dump", 0) + 1
                continue

            # write back cleaned text
            obj["text"] = text_fixed
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1

    print("✅ Saved:", OUT_PATH)
    print("Kept:", kept)
    print("Dropped:", dropped)
    print("Reasons:", reasons)

if __name__ == "__main__":
    main()