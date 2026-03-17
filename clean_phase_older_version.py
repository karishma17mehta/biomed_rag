# ingest/02_clean_phase1.py

import re
import json
import pandas as pd
import ftfy
from pathlib import Path
from tqdm import tqdm

DATA_PATH = Path("outputs/clean_papers.csv")
OUT_JSONL = Path("outputs/phase1_clean_papers.jsonl")
OUT_CSV   = Path("outputs/phase1_clean_papers.csv")

LABEL_COL = "cancer_type"
TEXT_COL  = "clean_text"

# -----------------------------
# 1) Heading reconstruction
# -----------------------------
HEADINGS = [
    "ABSTRACT", "BACKGROUND", "INTRODUCTION", "OBJECTIVE", "OBJECTIVES",
    "METHODS", "MATERIALS AND METHODS", "PATIENTS AND METHODS",
    "RESULTS", "DISCUSSION", "CONCLUSION", "CONCLUSIONS",
    "LIMITATIONS", "FUNDING", "CONFLICT OF INTEREST", "ACKNOWLEDGEMENTS",
    "REFERENCES", "BIBLIOGRAPHY", "REFERENCE LIST", "LITERATURE CITED",
    "WORKS CITED", "REFERENCES AND NOTES",
]

# Build a regex that catches headings even when glued to text like "BackgroundVitamin D..."
heading_re = re.compile(
    r"(?i)(" + "|".join(re.escape(h) for h in sorted(HEADINGS, key=len, reverse=True)) + r")"
)

def unflatten_headings(text: str) -> str:
    text = heading_re.sub(r"\n\1\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# -----------------------------
# 2) Boilerplate removal
# -----------------------------
BOILERPLATE_LINE_RE = re.compile(
    r"(?im)^\s*(?:"
    r".*\b[\w\.-]+@[\w\.-]+\.\w+\b.*|"                  # email
    r".*\b\d{4}-\d{4}-\d{4}-\d{3}[\dX]\b.*|"            # ORCID
    r".*(?:https?://\S+|www\.\S+).*|"                   # URL
    r".*\bdoi:\s*\S+.*|"                                # doi lines
    r".*\b(vol(?:ume)?\s+\d+|issue\s+\d+|page\s+\d+)\b.*|" # vol/issue/page
    r".*(creative\s+commons|open\s+access|cc[-\s]?by|all\s+rights\s+reserved|copyright).*"
    r")\s*$\n?"
)

def remove_boilerplate_lines(text: str) -> str:
    text = BOILERPLATE_LINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# -----------------------------
# 2.5) Remove publisher / footer / supplement junk lines
# -----------------------------
# Remove junk footer/publisher lines in one pass (multiline)
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

INLINE_AUTHOR_YEAR_CIT_RE = re.compile(
    r"\s*\((?:"
    r"[A-Z][A-Za-z'\-]+(?:\s+(?:and|&)\s+[A-Z][A-Za-z'\-]+)?"
    r"(?:\s+et\s+al\.)?"
    r"(?:,\s*[A-Z]\.)?"
    r",\s*\d{4}"
    r"(?:;\s*[A-Z][A-Za-z'\-]+(?:\s+et\s+al\.)?,\s*\d{4})*"
    r")\)\s*"
)

def remove_junk_lines_and_supplements(text: str) -> str:
    # 1) remove junk lines fast
    text = JUNK_LINES_RE.sub("", text)

    # 2) remove supplementary blocks (fast-ish)
    text = SUPPLEMENT_BLOCK_RE.sub("\n\n", text)

    # 3) remove inline author-year citations
    text = INLINE_AUTHOR_YEAR_CIT_RE.sub(" ", text)

    # cleanup
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



# -----------------------------
# 3) Remove figure/table mentions
# -----------------------------
FIG_TABLE_LABEL_RE = re.compile(r"(?i)\b(figure|fig\.?|table)\s*\d+[a-z]?\b[:\.\)]?")
FIG_TABLE_INLINE_RE = re.compile(r"(?i)\((?:see\s+)?(figure|fig\.?|table)\s*\d+[a-z]?(?:[^)]*)\)")
AS_SHOWN_RE = re.compile(r"(?i)\b(as\s+shown\s+in|shown\s+in|see)\s+(figure|fig\.?|table)\s*\d+[a-z]?\b[^\.]*\.")

def strip_figure_table_mentions(text: str) -> str:
    text = FIG_TABLE_INLINE_RE.sub("", text)
    text = AS_SHOWN_RE.sub("", text)
    text = FIG_TABLE_LABEL_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


# -----------------------------
# 4) Table block removal (conservative)
# -----------------------------
TABLE_DUMP_RE = re.compile(
    r"(?is)\n(?:table\s*\d+[^\n]*\n)?"
    r"(?:[^\n]*\d+[^\n]*\n){6,}"
)

def remove_table_dumps(text: str, enabled: bool = True) -> str:
    if not enabled:
        return text
    text = TABLE_DUMP_RE.sub("\n[TABLE_OMITTED]\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# -----------------------------
# 5) Citation cleanup (NEW)
# -----------------------------
# Numeric citations: [12], [12-14], [12, 15], [12–14]
NUMERIC_BRACKET_CIT_RE = re.compile(r"\[\s*\d+(?:\s*[\-,–]\s*\d+)?(?:\s*,\s*\d+(?:\s*[\-,–]\s*\d+)?)?\s*\]")
# Numeric parentheses: (12), (12, 13)
NUMERIC_PAREN_CIT_RE = re.compile(r"\(\s*\d+(?:\s*,\s*\d+)*\s*\)")

# Author-year: (Smith, 2020) / (Smith et al., 2020) / (van der Waals, 2019)
AUTHOR_YEAR_PAREN_RE = re.compile(
    r"\(\s*[A-Z][A-Za-z'’\-]+(?:\s+(?:[A-Z][A-Za-z'’\-]+|van|von|de|del|da|di|dos|der|den|la|le))*"
    r"(?:\s+et\s+al\.?)?"
    r"\s*,\s*\d{4}[a-z]?\s*\)"
)

# Narrative: Smith (2020) / Smith et al. (2020)
AUTHOR_YEAR_NARR_RE = re.compile(
    r"\b[A-Z][A-Za-z'’\-]+(?:\s+(?:[A-Z][A-Za-z'’\-]+|van|von|de|del|da|di|dos|der|den|la|le))*"
    r"(?:\s+et\s+al\.?)?\s*\(\s*\d{4}[a-z]?\s*\)"
)

# Your specific pattern (kept)
SPECIFIC_AUTHOR_INIT_RE = re.compile(r"\s\([A-Z][a-z]+,\s[A-Z][a-z]?\.[^\)]*,\s\d{4}\)")

# DOI bare forms often leak into body
DOI_RE = re.compile(r"(?i)\bdoi\s*:\s*\S+|\b10\.\d{4,9}/\S+\b")

def remove_inline_citations(text: str) -> str:
    if not text:
        return text


    # apply multiple passes; order matters
    text = NUMERIC_BRACKET_CIT_RE.sub("", text)
    text = NUMERIC_PAREN_CIT_RE.sub("", text)

    text = AUTHOR_YEAR_PAREN_RE.sub("", text)
    text = AUTHOR_YEAR_NARR_RE.sub("", text)

    text = SPECIFIC_AUTHOR_INIT_RE.sub("", text)
    text = DOI_RE.sub("", text)

    # cleanup spacing around removed citations
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\(\s*\)", "", text)  # empty parentheses
    return text.strip()


# -----------------------------
# 6) Cut references section
# -----------------------------
REF_HEADINGS = [
    "REFERENCES", "BIBLIOGRAPHY", "REFERENCE LIST", "LITERATURE CITED",
    "WORKS CITED", "CITED LITERATURE", "REFERENCES AND NOTES",
    "NOTES", "FOOTNOTES"
]

REF_CUT_RE = re.compile(
    r"(?ims)^\s*(?:"
    + "|".join(re.escape(h) for h in sorted(REF_HEADINGS, key=len, reverse=True))
    + r")\b[\s:.\-–—]*$"
    r".*"
)

def cut_references(text: str) -> str:
    m = REF_CUT_RE.search(text)
    if not m:
        return text.strip()
    return text[:m.start()].strip()


# -----------------------------
# 7) Final normalizers
# -----------------------------
def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

MOJIBAKE_HINT_RE = re.compile(r"(Ã|â€™|â€“|â€œ|â€|�)")
def maybe_ftfy(text: str) -> str:
    # only run ftfy if we see real mojibake markers
    if MOJIBAKE_HINT_RE.search(text):
        return ftfy.fix_text(text)
    return text


def basic_clean_v2(text: str, remove_tables: bool = True) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    text = maybe_ftfy(text)
    text = text.replace("ï¬", "")

    # 1) headings separable
    text = unflatten_headings(text)

    # 2) cut references EARLY (shrinks doc a lot)
    text = cut_references(text)

    # 3) boilerplate removal (regex-based)
    text = remove_boilerplate_lines(text)

    # 4) fig/table mentions
    text = strip_figure_table_mentions(text)

    # 5) inline citations only if likely present
    tlow = text.lower()
    if ("[" in text) or ("doi" in tlow) or ("10." in text) or re.search(r"\b\d{4}\b", text):
        text = remove_inline_citations(text)

    # recompute after changes
    tlow = text.lower()

    # 6) junk/supplement blocks only if likely present
    if ("dovepress" in tlow or "plos" in tlow or "journalpone" in tlow or
        "submit your manuscript" in tlow or "supplement" in tlow or
        "additional file" in tlow or "http" in tlow or "www." in tlow):
        text = remove_junk_lines_and_supplements(text)

    # recompute again (optional)
    tlow = text.lower()

    # 7) table dumps only if likely present
    if remove_tables and ("\n" in text) and ("table" in tlow or "\t" in text):
        text = remove_table_dumps(text, enabled=True)

    # 8) normalize
    text = normalize_whitespace(text)
    return text

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin1")

    if LABEL_COL not in df.columns or TEXT_COL not in df.columns:
        raise ValueError(f"Expected columns {LABEL_COL=} and {TEXT_COL=}. Found: {list(df.columns)}")

    df = df[[LABEL_COL, TEXT_COL]].copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df[TEXT_COL] = df[TEXT_COL].fillna("")

    records_for_csv = []

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        rows = df[[LABEL_COL, TEXT_COL]].itertuples(index=False, name=None)

        for i, (cancer_type, raw_text) in enumerate(tqdm(rows, total=len(df))):
            cancer_type = str(cancer_type).strip()
            raw_text = "" if not isinstance(raw_text, str) else raw_text

            clean_text = basic_clean_v2(raw_text, remove_tables=True)

            rec = {
                "paper_id": f"paper_{i:06d}",
                "cancer_type": cancer_type,
                "raw_text_preview": raw_text[:400],
                "clean_text": clean_text,
                "n_chars": len(clean_text),
                "n_words": word_count(clean_text),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            records_for_csv.append({
                "paper_id": rec["paper_id"],
                "cancer_type": cancer_type,
                "clean_text": clean_text,
                "n_chars": rec["n_chars"],
                "n_words": rec["n_words"],
            })

    out_df = pd.DataFrame(records_for_csv)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print("✅ Saved:")
    print(f"- {OUT_JSONL}")
    print(f"- {OUT_CSV}")
    print("\nQuick stats:")
    print(out_df.groupby("cancer_type")["paper_id"].count())
    print("\nArtifacts check:")
    print("Rows containing 'Ã':", (out_df["clean_text"].str.contains("Ã", na=False)).sum())
    print("Rows containing 'creative commons':", (out_df["clean_text"].str.contains("creative commons", case=False, na=False)).sum())
    print("Rows containing 'copyright':", (out_df["clean_text"].str.contains("copyright", case=False, na=False)).sum())
    print("Rows containing '[TABLE_OMITTED]':", (out_df["clean_text"].str.contains(r"\[TABLE_OMITTED\]", na=False)).sum())


if __name__ == "__main__":
    main()
