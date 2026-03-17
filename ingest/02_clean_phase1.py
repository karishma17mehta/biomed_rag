# ingest/02_clean_phase1_parallel.py

import os
import re
import json
import pandas as pd
import ftfy
from pathlib import Path
from tqdm import tqdm
from multiprocessing import get_context

DATA_PATH = Path("outputs/clean_papers.csv")
OUT_JSONL = Path("outputs/phase1_clean_papers.jsonl")
OUT_CSV   = Path("outputs/phase1_clean_papers.csv")

LABEL_COL = "cancer_type"
TEXT_COL  = "clean_text"

# -----------------------------
# 0) Cheap triggers / constants
# -----------------------------
YEAR_HINT_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
MOJIBAKE_HINT_RE = re.compile(r"(Ã|â€™|â€“|â€œ|â€|�)")

REF_HEADINGS = [
    "references and notes", "reference list", "literature cited", "works cited",
    "cited literature", "bibliography", "references", "notes", "footnotes"
]
REF_HEADINGS_UP = [h.upper() for h in REF_HEADINGS]
REF_TAIL_CHARS = 12000

# -----------------------------
# 1) Heading reconstruction (conditional)
# -----------------------------
HEADINGS = [
    "ABSTRACT", "BACKGROUND", "INTRODUCTION", "OBJECTIVE", "OBJECTIVES",
    "METHODS", "MATERIALS AND METHODS", "PATIENTS AND METHODS",
    "RESULTS", "DISCUSSION", "CONCLUSION", "CONCLUSIONS",
    "LIMITATIONS", "FUNDING", "CONFLICT OF INTEREST", "ACKNOWLEDGEMENTS",
    "REFERENCES", "BIBLIOGRAPHY", "REFERENCE LIST", "LITERATURE CITED",
    "WORKS CITED", "REFERENCES AND NOTES",
]
heading_re = re.compile(
    r"(?i)(" + "|".join(re.escape(h) for h in sorted(HEADINGS, key=len, reverse=True)) + r")"
)

def unflatten_headings(text: str) -> str:
    return heading_re.sub(r"\n\1\n", text)

# -----------------------------
# 2) Boilerplate removal (multiline)
# -----------------------------
BOILERPLATE_LINE_RE = re.compile(
    r"(?im)^\s*(?:"
    r".*\b[\w\.-]+@[\w\.-]+\.\w+\b.*|"
    r".*\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b.*|"
    r".*(?:https?://\S+|www\.\S+).*|"
    r".*\bdoi:\s*\S+.*|"
    r".*\b(vol(?:ume)?\s+\d+|issue\s+\d+|page\s+\d+)\b.*|"
    r".*(creative\s+commons|open\s+access|cc[-\s]?by|all\s+rights\s+reserved|copyright).*"
    r")\s*$\n?"
)
def remove_boilerplate_lines(text: str) -> str:
    return BOILERPLATE_LINE_RE.sub("", text)

# -----------------------------
# 2.5) Junk lines / supplement blocks (conditional)
# -----------------------------
JUNK_LINES_RE = re.compile(
    r"(?im)^\s*(?:"
    r"submit\s+your\s+manuscript\b.*|"
    r"(?:https?://|www\.)\S+.*|"
    r".*\bdovepress\b.*|"
    r".*\bplos\s+one\b.*|"
    r".*\bjournalpone\d+\b.*|"
    r".*\bmdpi\b.*|"
    r".*\bhindawi\b.*|"
    r".*\bspringer\b.*|"
    r".*\bwiley\b.*|"
    r".*\belsevier\b.*|"
    r".*\bfrontiers\b.*"
    r")\s*$\n?"
)
SUPPLEMENT_BLOCK_RE = re.compile(
    r"(?is)\b(?:additional\s+file|supplementary\s+(?:material|appendix|table|figure)|"
    r"(?:figure|table)\s*s\d+)\b.*?(?:\n\n|$)"
)

def remove_junk_lines_and_supplements(text: str) -> str:
    text = JUNK_LINES_RE.sub("", text)
    text = SUPPLEMENT_BLOCK_RE.sub("\n\n", text)
    return text

# -----------------------------
# 3) Figure/table mentions (cheap)
# -----------------------------
FIG_TABLE_LABEL_RE = re.compile(r"(?i)\b(figure|fig\.?|table)\s*\d+[a-z]?\b[:\.\)]?")
FIG_TABLE_INLINE_RE = re.compile(r"(?i)\((?:see\s+)?(figure|fig\.?|table)\s*\d+[a-z]?(?:[^)]*)\)")
AS_SHOWN_RE = re.compile(r"(?i)\b(as\s+shown\s+in|shown\s+in|see)\s+(figure|fig\.?|table)\s*\d+[a-z]?\b[^\.]*\.")

def strip_figure_table_mentions(text: str) -> str:
    text = FIG_TABLE_INLINE_RE.sub("", text)
    text = AS_SHOWN_RE.sub("", text)
    text = FIG_TABLE_LABEL_RE.sub("", text)
    return text

# -----------------------------
# 4) Table dumps (conditional)
# -----------------------------
TABLE_DUMP_RE = re.compile(
    r"(?is)\n(?:table\s*\d+[^\n]*\n)?"
    r"(?:[^\n]*\d+[^\n]*\n){6,}"
)
def remove_table_dumps(text: str) -> str:
    return TABLE_DUMP_RE.sub("\n[TABLE_OMITTED]\n", text)

# -----------------------------
# 5) Inline citations (conditional + fast order)
# -----------------------------
NUMERIC_BRACKET_CIT_RE = re.compile(r"\[\s*\d+(?:\s*[\-,–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[\-,–]\s*\d+)?)?\s*\]")
NUMERIC_PAREN_CIT_RE = re.compile(r"\(\s*\d+(?:\s*,\s*\d+)*\s*\)")

AUTHOR_YEAR_PAREN_RE = re.compile(r"\(\s*[A-Z][A-Za-z'’\-]+(?:\s+et\s+al\.?)?\s*,\s*\d{4}[a-z]?\s*\)")
AUTHOR_YEAR_NARR_RE = re.compile(r"\b[A-Z][A-Za-z'’\-]+(?:\s+et\s+al\.?)?\s*\(\s*\d{4}[a-z]?\s*\)")

DOI_RE = re.compile(r"(?i)\bdoi\s*:\s*\S+|\b10\.\d{4,9}/\S+\b")

def remove_inline_citations(text: str) -> str:
    if "[" in text:
        text = NUMERIC_BRACKET_CIT_RE.sub("", text)
    if "(" in text:
        text = NUMERIC_PAREN_CIT_RE.sub("", text)

    if "(" in text and YEAR_HINT_RE.search(text):
        text = AUTHOR_YEAR_PAREN_RE.sub("", text)
        text = AUTHOR_YEAR_NARR_RE.sub("", text)

    tlow = text.lower()
    if "doi" in tlow or "10." in text:
        text = DOI_RE.sub("", text)

    return text

# -----------------------------
# 6) Fast reference cutting (TAIL ONLY)
# -----------------------------
def cut_references_stronger(text: str) -> str:
    """
    Cut everything after the first occurrence of a references-like heading
    found in the last portion of the document.
    """

    if not text:
        return text

    tail = text[-20000:]  # increase window
    tail_up = tail.upper()

    # More aggressive patterns
    patterns = [
        r"\nREFERENCES\b",
        r"\nREFERENCE\b",
        r"\nBIBLIOGRAPHY\b",
        r"\nLITERATURE\s+CITED\b",
        r"\nACKNOWLEDGMENT[S]?\b",
        r"\nDECLARATION[S]?\b",
        r"\nCONFLICT[S]?\s+OF\s+INTEREST\b",
        r"\nAUTHOR[S]?\s+CONTRIBUTION[S]?\b",
    ]

    for pat in patterns:
        m = re.search(pat, tail_up)
        if m:
            cut_at = len(text) - len(tail) + m.start()
            return text[:cut_at].strip()

    # fallback: detect heavy citation tail
    if tail.count(" PMID ") > 5 or tail.count(" PubMed ") > 5:
        # cut at first PMID occurrence in tail
        idx = tail_up.find(" PMID ")
        if idx != -1:
            cut_at = len(text) - len(tail) + idx
            return text[:cut_at].strip()

    return text

# -----------------------------
# 7) Final normalize (single pass)
# -----------------------------
WS_RE_1 = re.compile(r"[ \t]+")
WS_RE_2 = re.compile(r"\n{3,}")

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = WS_RE_1.sub(" ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = WS_RE_2.sub("\n\n", text)
    return text.strip()

def maybe_ftfy(text: str) -> str:
    # keep this guard VERY tight
    if MOJIBAKE_HINT_RE.search(text):
        return ftfy.fix_text(text)
    return text

def basic_clean_fast(text: str, remove_tables: bool = True) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    text = maybe_ftfy(text).replace("ï¬", "")

    # cut refs early
    text = cut_references_stronger(text)

    # headings only if likely
    t_upper = text[:4000].upper()
    if ("ABSTRACT" in t_upper or "INTRODUCTION" in t_upper or "METHODS" in t_upper or
        "RESULTS" in t_upper or "DISCUSSION" in t_upper or "CONCLUSION" in t_upper or
        "REFERENCES" in t_upper):
        text = unflatten_headings(text)

    text = remove_boilerplate_lines(text)

    tlow = text.lower()

    if "figure" in tlow or "fig." in tlow or "table" in tlow:
        text = strip_figure_table_mentions(text)

    # citations only if likely
    if ("[" in text) or ("(" in text and YEAR_HINT_RE.search(text)) or ("doi" in tlow) or ("10." in text):
        text = remove_inline_citations(text)

    tlow = text.lower()
    if ("dovepress" in tlow or "plos" in tlow or "journalpone" in tlow or
        "submit your manuscript" in tlow or "supplement" in tlow or
        "additional file" in tlow or "http" in tlow or "www." in tlow):
        text = remove_junk_lines_and_supplements(text)

    if remove_tables:
        tlow = text.lower()
        if ("table" in tlow or "\t" in text) and "\n" in text:
            text = remove_table_dumps(text)

    return normalize_whitespace(text)

# faster than regex word tokenization
def word_count_fast(text: str) -> int:
    if not text:
        return 0
    return len(text.split())

# -----------------------------
# Parallel worker
# -----------------------------
def process_row(args):
    i, cancer_type, raw_text = args
    clean_text = basic_clean_fast(raw_text, remove_tables=True)
    rec = {
        "paper_id": f"paper_{i:06d}",
        "cancer_type": str(cancer_type).strip(),
        "raw_text_preview": (raw_text[:300] if isinstance(raw_text, str) else ""),
        "clean_text": clean_text,
        "n_chars": len(clean_text),
        "n_words": word_count_fast(clean_text),
    }
    return rec

def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, encoding="utf-8", engine="c")
    df = df[[LABEL_COL, TEXT_COL]].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    tasks = [(i, ct, txt) for i, (ct, txt) in enumerate(df[[LABEL_COL, TEXT_COL]].itertuples(index=False, name=None))]
    n = len(tasks)

    # M3 Pro: 10–12 workers is usually the sweet spot
    workers = min(12, os.cpu_count() or 8)

    # IMPORTANT on macOS: use spawn
    ctx = get_context("spawn")

    # Buffering: fewer writes = much faster
    jsonl_buffer = []
    buffer_flush_every = 200  # tune: 200–1000

    records_for_csv = []

    with ctx.Pool(processes=workers) as pool, open(OUT_JSONL, "w", encoding="utf-8") as jf:
        for rec in tqdm(pool.imap_unordered(process_row, tasks, chunksize=50), total=n):
            records_for_csv.append({
                "paper_id": rec["paper_id"],
                "cancer_type": rec["cancer_type"],
                "clean_text": rec["clean_text"],
                "n_chars": rec["n_chars"],
                "n_words": rec["n_words"],
            })

            jsonl_buffer.append(json.dumps(rec, ensure_ascii=False))

            if len(jsonl_buffer) >= buffer_flush_every:
                jf.write("\n".join(jsonl_buffer) + "\n")
                jsonl_buffer.clear()

        # final flush
        if jsonl_buffer:
            jf.write("\n".join(jsonl_buffer) + "\n")

    out_df = pd.DataFrame(records_for_csv)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("✅ Saved:")
    print(f"- {OUT_JSONL}")
    print(f"- {OUT_CSV}")
    print(out_df.groupby("cancer_type")["paper_id"].count())

if __name__ == "__main__":
    main()