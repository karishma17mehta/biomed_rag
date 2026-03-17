# ingest/04_chunk_papers.py
import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# -------------------
# Paths
# -------------------
# ✅ Use your high-confidence dataset for best quality / least noise
DATA_PATH = Path("outputs/phase2_highconf_papers_strict.csv")  
OUT_JSONL = Path("outputs/chunks.jsonl")
OUT_STATS = Path("outputs/chunks_stats.json")

LABEL_COL = "cancer_type"
TEXT_COL  = "clean_text"
PAPER_ID_COL = "paper_id"

# -------------------
# Chunking params (tuned for final chunk size after header+overlap)
# -------------------
# Final tokens ≈ TARGET + overlap_used + header_tokens
TARGET_TOKENS   = 320
OVERLAP_TOKENS  = 80
MIN_PARA_TOKENS = 80
MAX_PARA_TOKENS = 450
MIN_CHUNK_TOKENS = 160
MIN_FINAL_TOKENS = 180
MAX_FINAL_TOKENS = 650  # if paragraph exceeds this, sentence-split it

# -------------------
# Section detection
# -------------------
SKIP_SECTIONS = {
    "ACKNOWLEDGEMENTS",
    "ACKNOWLEDGMENTS",
    "FUNDING",
    "CONFLICT OF INTEREST",
    "COMPETING INTERESTS",
    "DATA AVAILABILITY",
    "AVAILABILITY OF DATA AND MATERIALS",
    "CONSENT",
    "CONSENT FOR PUBLICATION",
    "TRIAL REGISTRATION",
    "PROVENANCE AND PEER REVIEW",
    "SUPPLEMENTARY MATERIAL",
    "SUPPLEMENTARY INFORMATION",
}

HEADINGS = [
    "ABSTRACT", "BACKGROUND", "INTRODUCTION", "OBJECTIVE", "OBJECTIVES",
    "METHODS", "MATERIALS AND METHODS", "PATIENTS AND METHODS",
    "RESULTS", "DISCUSSION", "CONCLUSION", "CONCLUSIONS",
    "LIMITATIONS", "FUNDING", "CONFLICT OF INTEREST", "ACKNOWLEDGEMENTS",
    "REFERENCES", "BIBLIOGRAPHY"
]

HEADING_LINE_RE = re.compile(
    r"(?im)^(?:"
    + "|".join(re.escape(h) for h in sorted(HEADINGS, key=len, reverse=True))
    + r")\b[\s:.\-–—]*$"
)

# Sentence splitting (conservative)
SENT_SPLIT_RE = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z0-9])")

# -------------------
# Tokenizer
# -------------------
def get_encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

ENC = get_encoder()

def count_tokens(text: str) -> int:
    if not text:
        return 0
    if ENC is None:
        return max(1, len(text.split()))
    return len(ENC.encode(text))

def take_last_tokens(text: str, n_tokens: int) -> str:
    if not text or n_tokens <= 0:
        return ""
    if ENC is None:
        words = text.split()
        return " ".join(words[-n_tokens:])
    toks = ENC.encode(text)
    return ENC.decode(toks[-n_tokens:])

# -------------------
# Noise / table-ish removal (conservative)
# -------------------
TABLEISH_LINE_RE = re.compile(r"(?i)^\s*(table\s*\d+|fig(?:ure)?\s*\d+)\b")
MANY_NUMS_RE = re.compile(r"(?:\d[\d\.\-\(\)%/]*\s+){8,}")
PIPE_TABLE_RE = re.compile(r"\|(?:[^|\n]+\|){2,}")
REPEATED_SEP_RE = re.compile(r"[-_]{6,}|={6,}")

def drop_tableish_blocks(text: str) -> str:
    """
    Conservative line-level filter to reduce extracted table dumps / numeric soups.
    Keeps normal prose with a few numbers.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    lines = text.splitlines()
    kept: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            kept.append("")
            continue

        # obvious table / figure captions or headings
        if TABLEISH_LINE_RE.search(s):
            continue

        # pipe-style tables or repeated separators
        if PIPE_TABLE_RE.search(ln) or REPEATED_SEP_RE.search(ln):
            continue

        # lines that look like numeric column dumps
        if MANY_NUMS_RE.search(ln):
            continue

        kept.append(ln)

    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

# -------------------
# Helpers
# -------------------
def split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]

def merge_small_paras(paras: List[str], min_tokens: int) -> List[str]:
    merged: List[str] = []
    buf = ""
    for p in paras:
        if not buf:
            buf = p
            continue
        if count_tokens(buf) < min_tokens:
            buf = buf + "\n" + p
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged

def token_window_split(text: str, max_tokens: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if ENC is None:
        words = text.split()
        return [" ".join(words[i:i+max_tokens]) for i in range(0, len(words), max_tokens)]
    toks = ENC.encode(text)
    return [ENC.decode(toks[i:i+max_tokens]) for i in range(0, len(toks), max_tokens)]

def split_long_para_into_sentences(para: str) -> List[str]:
    para = para.strip()
    if not para:
        return []
    sents = [s.strip() for s in SENT_SPLIT_RE.split(para) if s.strip()]
    return sents if sents else [para]

# -------------------
# Section segmentation
# -------------------
def segment_by_sections(text: str) -> Optional[List[Tuple[str, str]]]:
    lines = text.splitlines()
    heading_idxs: List[int] = []

    for idx, ln in enumerate(lines):
        if HEADING_LINE_RE.match(ln.strip()):
            heading_idxs.append(idx)

    if len(heading_idxs) < 2:
        return None

    segments: List[Tuple[str, str]] = []
    for j, start_idx in enumerate(heading_idxs):
        raw_heading = lines[start_idx].strip().upper()
        section_name = re.sub(r"[\s:.\-–—]+$", "", raw_heading)

        end_idx = heading_idxs[j + 1] if j + 1 < len(heading_idxs) else len(lines)
        section_text = "\n".join(lines[start_idx + 1:end_idx]).strip()
        if section_text:
            # drop tiny sections (often acknowledgements / boilerplate)
            if count_tokens(section_text) < 120:
                continue
            segments.append((section_name, section_text))

    return segments if segments else None

def build_units_for_doc(text: str) -> List[Tuple[str, str]]:
    """
    Returns list of (section, unit_text) where unit_text is a paragraph (or merged paragraph).
    Skips REFERENCES/BIBLIOGRAPHY sections if section detection succeeds.
    """
    sec_segments = segment_by_sections(text)
    units: List[Tuple[str, str]] = []



    if sec_segments is not None:
        for sec, sec_text in sec_segments:
            if sec.upper() in SKIP_SECTIONS:
                continue
            paras = split_paragraphs(sec_text)
            paras = merge_small_paras(paras, MIN_PARA_TOKENS)
            for p in paras:
                units.append((sec, p))
        return units

    # fallback: whole text as UNKNOWN
    paras = split_paragraphs(text)
    paras = merge_small_paras(paras, MIN_PARA_TOKENS)
    for p in paras:
        units.append(("UNKNOWN", p))
    return units

# -------------------
# Chunk packing
# -------------------
def pack_units_into_chunks(
    units: List[Tuple[str, str]],
    cancer_type: str,
    paper_id: str,
) -> List[Dict]:
    chunks: List[Dict] = []
    cur_parts: List[str] = []
    cur_section = units[0][0] if units else "UNKNOWN"
    cur_tokens = 0
    chunk_idx = 0

    def emit(raw: str, section: str):
        nonlocal chunk_idx
        raw = raw.strip()
        if not raw:
            return

        # merge tiny chunk into previous chunk instead of creating a new chunk
        if chunks and count_tokens(raw) < MIN_CHUNK_TOKENS:
            prev = chunks[-1]
            # keep section consistent: only merge if same section (recommended)
            if prev["section"] == section:
                prev["raw_text"] = prev["raw_text"].rstrip() + "\n\n" + raw
                prev["n_tokens_raw"] = count_tokens(prev["raw_text"])
                prev["n_chars_raw"] = len(prev["raw_text"])
                return

        chunks.append({
            "chunk_id": f"{paper_id}::c{chunk_idx:04d}",
            "paper_id": paper_id,
            "cancer_type": cancer_type,
            "section": section,
            "chunk_index": chunk_idx,
            "raw_text": raw,
            "n_tokens_raw": count_tokens(raw),
            "n_chars_raw": len(raw),
        })
        chunk_idx += 1

    def flush():
        nonlocal cur_parts, cur_tokens
        if not cur_parts:
            return

        raw = "\n\n".join(cur_parts).strip()

        # If merged raw exceeds TARGET, split into windows
        if count_tokens(raw) > TARGET_TOKENS:
            for piece in token_window_split(raw, TARGET_TOKENS):
                emit(piece, cur_section)
        else:
            emit(raw, cur_section)

        cur_parts = []
        cur_tokens = 0

    for section, unit_text in units:
        unit_text = unit_text.strip()
        if not unit_text:
            continue

        # switch sections => flush
        if section != cur_section and cur_parts:
            flush()
            cur_section = section
        else:
            cur_section = section

        utoks = count_tokens(unit_text)

        # If a single unit exceeds TARGET, split by sentences then windows
        if utoks > TARGET_TOKENS:
            if cur_parts:
                flush()

            sents = split_long_para_into_sentences(unit_text)
            temp_parts: List[str] = []
            temp_tokens = 0

            for s in sents:
                stoks = count_tokens(s)

                if stoks > TARGET_TOKENS:
                    if temp_parts:
                        emit("\n\n".join(temp_parts), cur_section)
                        temp_parts, temp_tokens = [], 0
                    for piece in token_window_split(s, TARGET_TOKENS):
                        emit(piece, cur_section)
                    continue

                if temp_tokens + stoks > TARGET_TOKENS and temp_parts:
                    emit("\n\n".join(temp_parts), cur_section)
                    temp_parts, temp_tokens = [], 0

                temp_parts.append(s)
                temp_tokens += stoks

            if temp_parts:
                emit("\n\n".join(temp_parts), cur_section)

            continue

        # Normal packing
        if cur_tokens + utoks > TARGET_TOKENS and cur_parts:
            flush()

        cur_parts.append(unit_text)
        cur_tokens += utoks

    flush()
    return chunks

# -------------------
# Better key idea + smarter overlap
# -------------------
BAD_SENT_RE = re.compile(r"(?i)\b(preprint|doi|http|https|www|copyright|license)\b")
NUM_SOUP_RE = re.compile(r"^[\d\W_]+$")

def first_sentence(text: str, max_words: int = 25) -> str:
    if not text or not text.strip():
        return ""

    # if a header already exists, remove it
    if "---\n" in text:
        text = text.split("---\n", 1)[-1].strip()

    candidates = [c.strip() for c in SENT_SPLIT_RE.split(text) if c.strip()]

    for c in candidates[:12]:
        if len(c.split()) < 6:
            continue
        if BAD_SENT_RE.search(c):
            continue
        if NUM_SOUP_RE.match(re.sub(r"\s+", "", c)):
            continue
        if re.match(r"^[0-9\(\[\{]", c.strip()):
            continue

        words = c.split()
        return (" ".join(words[:max_words]).rstrip() + "…") if len(words) > max_words else c

    words = text.split()
    return (" ".join(words[:max_words]).rstrip() + "…") if len(words) > max_words else text.strip()

def build_micro_header(cancer_type: str, section: str, paper_id: str, raw_chunk_text: str) -> str:
    key = first_sentence(raw_chunk_text, max_words=25)
    return f"[Cancer: {cancer_type} | Section: {section} | Paper: {paper_id}]\nKey idea: {key}\n---\n"

def looks_like_mid_thought(s: str) -> bool:
    s = s.lstrip()
    return bool(re.match(r"^[,;:)\]]|^[a-z]|^’|^\"", s))

def add_overlap_and_headers(chunks: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    prev_raw = ""

    for ch in chunks:
        raw = ch["raw_text"].strip()

        overlap = take_last_tokens(prev_raw, OVERLAP_TOKENS) if prev_raw else ""
        body = raw

        if overlap:
            candidate = (overlap + "\n\n" + raw).strip()
            # IMPORTANT: candidate has no header, so don't split on --- here
            if not looks_like_mid_thought(candidate):
                body = candidate

        header = build_micro_header(ch["cancer_type"], ch["section"], ch["paper_id"], raw)

        ch2 = dict(ch)

        # ✅ embedding/BM25 field: body only
        ch2["text"] = body

        # ✅ optional display field: header + body
        ch2["text_display"] = header + body

        # recompute stats for body-only text
        ch2["n_tokens"] = count_tokens(ch2["text"])
        ch2["n_chars"] = len(ch2["text"])

        overlap_used = bool(overlap and body != raw)
        ch2["overlap_tokens"] = OVERLAP_TOKENS if overlap_used else 0

        # diagnostics
        ch2["n_tokens_body_only"] = count_tokens(raw)
        ch2["n_tokens_overlap_only"] = count_tokens(overlap) if overlap_used else 0
        ch2["n_tokens_header_only"] = count_tokens(header)

        out.append(ch2)
        prev_raw = raw

    return out

def repair_tiny_chunks(chunks: List[Dict], min_tokens: int = 180, max_tokens: int = 650) -> List[Dict]:
    """
    Merge very small chunks into neighbors to reduce <min_tokens tail.
    Keeps within max_tokens when possible.
    Assumes chunks are ordered by chunk_index.
    """
    if not chunks:
        return chunks

    repaired = []
    i = 0
    while i < len(chunks):
        ch = chunks[i]
        tok = ch.get("n_tokens", count_tokens(ch.get("text", "")))

        if tok >= min_tokens:
            repaired.append(ch)
            i += 1
            continue

        # Try merge into previous if same paper + same section and won't exceed max_tokens too much
        if repaired:
            prev = repaired[-1]
            if prev["paper_id"] == ch["paper_id"] and prev["section"] == ch["section"]:
                prev_tok = prev.get("n_tokens", count_tokens(prev["text"]))
                if prev_tok + tok <= max_tokens + 80:  # allow slight spill
                    # merge: append body (not re-overlap), simplest: concatenate texts
                    prev["text"] = prev["text"].rstrip() + "\n\n" + ch["text"].lstrip()
                    prev["n_tokens"] = count_tokens(prev["text"])
                    prev["n_chars"] = len(prev["text"])
                    i += 1
                    continue

        # Else merge into next if exists and same paper (best effort)
        if i + 1 < len(chunks):
            nxt = chunks[i + 1]
            if nxt["paper_id"] == ch["paper_id"]:
                nxt_tok = nxt.get("n_tokens", count_tokens(nxt["text"]))
                if tok + nxt_tok <= max_tokens + 80:
                    nxt["text"] = ch["text"].rstrip() + "\n\n" + nxt["text"].lstrip()
                    nxt["n_tokens"] = count_tokens(nxt["text"])
                    nxt["n_chars"] = len(nxt["text"])
                    i += 1
                    continue

        # If we couldn't merge safely, keep it (rare)
        repaired.append(ch)
        i += 1

    return repaired

# -------------------
# Main
# -------------------
def main():
    print("RUN PARAMS:", TARGET_TOKENS, OVERLAP_TOKENS, MIN_PARA_TOKENS, MAX_PARA_TOKENS)
    print("INPUT:", DATA_PATH)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

    total_docs = len(df)
    total_chunks = 0
    chunk_sizes: List[int] = []
    chunks_per_doc: List[int] = []

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            paper_id = str(row[PAPER_ID_COL])
            cancer_type = str(row[LABEL_COL])
            text = row[TEXT_COL]

            # ✅ remove table-ish / numeric-dump lines before building units
            text = drop_tableish_blocks(text)

            units = build_units_for_doc(text)
            if not units:
                continue

            base_chunks = pack_units_into_chunks(units, cancer_type, paper_id)
            final_chunks = add_overlap_and_headers(base_chunks)
            final_chunks = repair_tiny_chunks(final_chunks, min_tokens=MIN_FINAL_TOKENS, max_tokens=MAX_FINAL_TOKENS)

            chunks_per_doc.append(len(final_chunks))
            for ch in final_chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")
                total_chunks += 1
                chunk_sizes.append(ch["n_tokens"])

    stats = {
        "input_csv": str(DATA_PATH),
        "total_docs": total_docs,
        "total_chunks": total_chunks,
        "avg_chunks_per_doc": (sum(chunks_per_doc) / max(1, len(chunks_per_doc))),
        "min_chunks_per_doc": int(min(chunks_per_doc)) if chunks_per_doc else 0,
        "max_chunks_per_doc": int(max(chunks_per_doc)) if chunks_per_doc else 0,
        "avg_chunk_tokens": (sum(chunk_sizes) / max(1, len(chunk_sizes))),
        "p50_chunk_tokens": float(pd.Series(chunk_sizes).quantile(0.5)) if chunk_sizes else 0.0,
        "p90_chunk_tokens": float(pd.Series(chunk_sizes).quantile(0.9)) if chunk_sizes else 0.0,
        "p99_chunk_tokens": float(pd.Series(chunk_sizes).quantile(0.99)) if chunk_sizes else 0.0,
        "token_counter": "tiktoken(cl100k_base)" if ENC is not None else "fallback(words)",
        "params": {
            "TARGET_TOKENS": TARGET_TOKENS,
            "OVERLAP_TOKENS": OVERLAP_TOKENS,
            "MIN_PARA_TOKENS": MIN_PARA_TOKENS,
            "MAX_PARA_TOKENS": MAX_PARA_TOKENS,
            "SKIP_SECTIONS": sorted(list(SKIP_SECTIONS)),
            "TABLEISH_FILTERING": True,
        }
    }

    with open(OUT_STATS, "w", encoding="utf-8") as sf:
        json.dump(stats, sf, indent=2)

    print("✅ Saved chunks:", OUT_JSONL)
    print("✅ Saved stats:", OUT_STATS)
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
