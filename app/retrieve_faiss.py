# app/retrieve_faiss.py
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import faiss
from openai import OpenAI
import pickle

from app.query_router import infer_intent, extract_entities, section_bonus, extract_cancer_type

INDEX_DIR = Path("outputs/index_openai")
FAISS_PATH = INDEX_DIR / "faiss.index"
META_PATH  = INDEX_DIR / "meta_tagged_v2.jsonl"
BM25_PATH  = INDEX_DIR / "bm25.pkl"

MODEL = "text-embedding-3-large"

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment.")
client = OpenAI(api_key=api_key)


# -------------------------
# Cancer term dictionaries
# -------------------------

CANCER_TERMS = {
    "Colon_Cancer": {
        "must": [
        "colorectal", "colon", "rectal", "rectum",
        "crc", "mcrc",
        "sigmoid", "cecum", "caecum",
        "large intestine", "bowel",
        ]
        ,
        "anti": [
            "thyroid", "papillary thyroid", "follicular thyroid", "medullary thyroid",
            "nsclc", "sclc", "non-small cell lung", "small cell lung", "pulmonary",
        ],
    },

    "Lung_Cancer": {
        "must": [
        "lung", "pulmonary",
        "nsclc", "sclc",
        "non-small cell", "small cell",
        "lung adenocarcinoma", "squamous cell lung",
        ],
        "anti": [
            "thyroid", "papillary thyroid", "follicular thyroid", "medullary thyroid",
            "colon", "colorectal", "rectal", "crc",
        ],
    },

    "Thyroid_Cancer": {
        "must": [
            "thyroid",
            "papillary thyroid", "follicular thyroid", "medullary thyroid", "anaplastic thyroid",
            "differentiated thyroid",
            "ptc", "ftc", "mtc", "atc", "dtc",
            "thyroid cancer", "thyroid carcinoma",
            "papillary thyroid carcinoma", "follicular thyroid carcinoma",
            "medullary thyroid carcinoma", "anaplastic thyroid carcinoma",
            "calcitonin", "thyroglobulin", "tsh",
            "ret proto-oncogene", "thyroidectomy",
        ],
        "anti": [
            "nsclc", "sclc", "lung", "pulmonary",
            "colon", "colorectal", "rectal", "crc",
        ],
    },
}

# -------------------------
# Helpers
# -------------------------

SEP_RE = re.compile(r"\s*---\s*", re.MULTILINE)

def body_text(full_text: str) -> str:
    """Extract the chunk body (not metadata header)."""
    if not full_text:
        return ""
    parts = SEP_RE.split(full_text, maxsplit=1)
    if len(parts) == 2:
        return parts[1].strip()
    m = re.search(r"\]\s*(.*)$", full_text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return full_text.strip()

CITATION_QUERY_RE = re.compile(
    r"\b(cite|citation|citations|reference|references|bibliograph|pmid|doi|pubmed)\b",
    re.IGNORECASE
)

def wants_citations(query: str, intent: str) -> bool:
    return bool(CITATION_QUERY_RE.search(query)) or intent == "citations"

def embed_query(q: str) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

def load_meta() -> List[Dict[str, Any]]:
    metas = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metas.append(json.loads(line))
    return metas

def entity_bonus_score(entities: List[str], text: str, cap: float = 0.25) -> float:
    if not entities or not text:
        return 0.0
    t = text.lower()
    hits = sum(1 for e in entities if e.lower() in t)
    return min(cap, hits * 0.03)

def dedupe_and_diversify(
    candidates: List[Dict[str, Any]],
    max_per_paper: int = 1,
    max_per_hash: int = 1
) -> List[Dict[str, Any]]:
    seen_hash = {}
    per_paper = {}
    kept = []
    for c in candidates:
        h = c.get("content_hash") or ""
        pid = c.get("paper_id") or ""

        if h:
            seen_hash[h] = seen_hash.get(h, 0) + 1
            if seen_hash[h] > max_per_hash:
                continue

        if pid:
            per_paper[pid] = per_paper.get(pid, 0) + 1
            if per_paper[pid] > max_per_paper:
                continue

        kept.append(c)
    return kept

CANCER_KEYWORDS = {
    "Thyroid_Cancer": ["thyroid", "papillary", "follicular", "medullary", "anaplastic"],
    "Colon_Cancer":   ["colon", "colorectal", "crc", "rectal"],
    "Lung_Cancer":    ["lung", "nsclc", "sclc", "adenocarcinoma", "squamous"],
}

def cancer_keyword_mismatch_penalty(requested: str, body: str) -> float:
    """Light penalty if requested cancer keywords don't appear but another cancer's do."""
    if not requested or not body:
        return 0.0

    t = body.lower()
    if any(k in t for k in CANCER_KEYWORDS.get(requested, [])):
        return 0.0

    best_other = 0
    for ct, kws in CANCER_KEYWORDS.items():
        if ct == requested:
            continue
        hits = sum(1 for k in kws if k in t)
        best_other = max(best_other, hits)

    if best_other >= 4:
        return 0.35
    if best_other >= 2:
        return 0.20
    return 0.0

def load_bm25():
    with BM25_PATH.open("rb") as f:
        payload = pickle.load(f)
    return payload["bm25"]

TOKEN_RE = re.compile(r"[A-Za-z0-9\-_/]+")
def bm25_tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())

def count_terms_unique(text: str, terms: list[str]) -> int:
    t = (text or "").lower()
    t_compact = re.sub(r"\s+", "", t)  # removes all whitespace

    hits = 0
    for term in terms:
        term_l = term.lower()
        term_compact = re.sub(r"\s+", "", term_l)

        # normal match
        if term_l in t:
            hits += 1
            continue

        # compact match for short tokens (nsclc, egfr, kras, braf, msi, etc.)
        if 2 <= len(term_compact) <= 12 and term_compact in t_compact:
            hits += 1

    return hits

def cancer_must_mention_gate(requested: str, body: str) -> Tuple[bool, float, str, int]:
    """
    Soft gate (never drops; only penalizes).
    Returns (keep, penalty, reason, req_hits)
    """
    if not requested:
        return True, 0.0, "no requested cancer", 0

    spec = CANCER_TERMS.get(requested, {})
    must = spec.get("must", [])
    anti = spec.get("anti", [])

    req_hits = count_terms_unique(body, must)
    anti_hits = count_terms_unique(body, anti)

    # Requested not mentioned, strong other-cancer signal
    if req_hits == 0 and anti_hits >= 2:
        return True, 0.60, "no requested; strong other-cancer signal", req_hits

    # Requested not mentioned, weak other-cancer hint
    if req_hits == 0 and anti_hits == 1:
        return True, 0.30, "no requested; weak other-cancer hint", req_hits

    # No cancer evidence at all (generic chunk)
    if req_hits == 0 and anti_hits == 0:
        return True, 0.15, "generic/no cancer mention", req_hits

    # Mentioned requested -> OK
    return True, 0.0, "mentions requested cancer", req_hits

# -------------------------
# High-signal entity reranking (query -> overlap in chunk)
# -------------------------

CHECKPOINT_PAT = re.compile(r"\b(PD-?1|PD-?L1|CTLA-?4|LAG-?3|TIM-?3|TIGIT)\b", re.IGNORECASE)
PATHWAY_PAT = re.compile(r"\b(MAPK|PI3K/?AKT|AKT|mTOR|WNT|TGF-?β|TGF-?B|JAK/?STAT|NF-?κB|NF-?KB)\b", re.IGNORECASE)
DRUG_PAT = re.compile(
    r"\b[a-zA-Z][a-zA-Z\-]{3,40}"
    r"(?:nib|tinib|mab|zumab|ximab|umab|cept|parib|ciclib|platin|taxel|cycline)\b",
    re.IGNORECASE,
)
GENE_PAT = re.compile(r"\b[A-Z]{2,}[A-Z0-9]{0,8}\b")
GENE_STOP = {
    "CI","SD","SE","NS","NA","USA",
    "BMC","AJCC","WHO","SEER",
    "DNA","RNA","MRNA",
    "FIG","TABLE","SUPP","ETAL",
    "I","II","III","IV","V","VI",
    "CT","MRI","PET","FDG","SUV",
    "OS","PFS","DFS","RFS","ORR","DCR","HR","PK","PD","AUC",
}

def extract_highsignal_entities(text: str) -> List[str]:
    if not text:
        return []
    ents: List[str] = []
    ents += [m.upper() for m in CHECKPOINT_PAT.findall(text)]
    ents += [m.upper() for m in PATHWAY_PAT.findall(text)]
    ents += [m.lower() for m in DRUG_PAT.findall(text)]
    for g in GENE_PAT.findall(text):
        if g in GENE_STOP:
            continue
        if re.fullmatch(r"\d+", g):
            continue
        if len(g) < 2:
            continue
        ents.append(g)
    return sorted(set(ents))

def normalize_for_match(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("/", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def entity_overlap_ratio(query_ents: List[str], text: str) -> float:
    """Robust overlap including compact matching (K RAS -> KRAS, PD 1 -> PD-1)."""
    if not query_ents:
        return 0.0

    raw = text or ""
    blob_norm = normalize_for_match(raw)
    blob_upper_nospace = re.sub(r"\s+", "", raw.upper())
    blob_lower_nospace = re.sub(r"\s+", "", raw.lower())

    found = 0
    for e in query_ents:
        e_norm = normalize_for_match(e)

        # 1) boundary match on normalized text
        if re.search(rf"\b{re.escape(e_norm)}\b", blob_norm, flags=re.IGNORECASE):
            found += 1
            continue

        # 2) compact symbol match (handles K RAS, PD 1, PD-1)
        e_compact_upper = re.sub(r"[\s\-_/]+", "", e.upper())
        e_compact_lower = re.sub(r"[\s\-_/]+", "", e.lower())

        if 2 <= len(e_compact_upper) <= 12:
            if e_compact_upper in blob_upper_nospace or e_compact_lower in blob_lower_nospace:
                found += 1
                continue

    return found / max(1, len(query_ents))

# Intent evidence (prevents “random KRAS mention” from ranking for resistance questions)
RESIST_PAT = re.compile(r"\b(resistan|refractor|escape|non[- ]?respond|sensitiz|overcome)\w*", re.I)

def intent_evidence_bonus(intent: str, body: str) -> float:
    if intent == "resistance":
        return 0.35 if RESIST_PAT.search(body or "") else -0.25
    return 0.0

from typing import Dict, Tuple  # make sure Tuple is imported

import re
from typing import Dict, Tuple

def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def count_terms_unique_nonoverlap(text: str, terms: list[str]) -> int:
    """
    Counts UNIQUE term hits with overlap-control:
    - match longest terms first
    - after a term matches, mask that span so substrings won't also count
    """
    t = _normalize(text)
    if not t or not terms:
        return 0

    # sort longest-first so 'colorectal cancer' beats 'colorectal'
    terms_sorted = sorted(set(_normalize(x) for x in terms if x), key=len, reverse=True)

    hits = 0
    for term in terms_sorted:
        # word-ish boundaries: avoid matching inside other words
        # also allow spaces inside term as is
        pat = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
        m = pat.search(t)
        if m:
            hits += 1
            # mask matched span so shorter substrings won't match later
            t = t[:m.start()] + (" " * (m.end() - m.start())) + t[m.end():]
    return hits

def cancer_evidence_scores(body: str) -> Dict[str, int]:
    t = body or ""
    return {
        ct: count_terms_unique_nonoverlap(t, spec.get("must", []))
        for ct, spec in CANCER_TERMS.items()
    }


def cancer_evidence_adjust(requested: str, body: str) -> Tuple[float, Dict[str, int]]:
    if not requested:
        return 0.0, {}

    scores = cancer_evidence_scores(body)
    if not scores:
        return 0.0, {}

    req = scores.get(requested, 0)
    best_ct = max(scores, key=scores.get)
    best = scores[best_ct]

    # Generic
    if best == 0:
        return -0.10, scores

    # Requested present
    if req > 0:
        return min(0.30, 0.10 * req), scores

    # Requested absent, other present
    # Use dominance margin to avoid punishing mixed weak signals
    margin = best - req  # req is 0 here
    if margin >= 2:
        return -0.60, scores
    return -0.25, scores


# -------------------------
# Retrieval (Hybrid: Dense + BM25)
# -------------------------

def retrieve(
    query: str,
    top_n_dense: int = 200,
    top_n_bm25: int = 200,
    top_k: int = 10,
    max_per_paper: int = 1,
    max_per_hash: int = 1,
    bm25_div: float = 20.0,
    bm25_weight: float = 0.15,     # overridden dynamically
    rerank_weight: float = 0.18,
    debug: bool = False,
):
    intent = infer_intent(query)
    entities = extract_entities(query)
    cancer_type = extract_cancer_type(query)
    allow_refs = wants_citations(query, intent)
    q_hi = extract_highsignal_entities(query)

    if debug:
        print("intent:", intent)
        print("entities:", entities)
        print("cancer_type:", cancer_type)
        print("q_hi:", q_hi)

    index = faiss.read_index(str(FAISS_PATH))
    metas = load_meta()

    # -------------------------
    # Dense retrieval
    # -------------------------
    qv = embed_query(query).reshape(1, -1)
    dense_scores, dense_ids = index.search(qv, top_n_dense)
    dense_scores = dense_scores[0].tolist()
    dense_ids = dense_ids[0].tolist()
    dense_map = {i: float(s) for i, s in zip(dense_ids, dense_scores) if i >= 0}

    # -------------------------
    # BM25 retrieval (IMPORTANT: use SAME bm25_query for scores + top ids)
    # Add high-signal entities to help lexical match
    # -------------------------
    bm25 = load_bm25()
    bm25_query = query + (" " + " ".join(q_hi) if q_hi else "")
    q_tokens = bm25_tokenize(bm25_query)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top_ids = list(np.argsort(bm25_scores)[::-1])[:top_n_bm25]

    # dynamic BM25 weight (more when entity-like query)
    bm25_weight = 0.25 if entities else 0.10


    # -------------------------
    # Candidate pool
    # -------------------------
    candidate_ids = set(dense_map.keys()) | set(bm25_top_ids)

    CONTENT_SECTIONS = {
        "ABSTRACT", "INTRODUCTION", "RESULTS", "DISCUSSION",
        "CONCLUSION", "BACKGROUND", "OBJECTIVE"
    }

    candidates = []
    for idx in candidate_ids:
        m = metas[idx]

        full_text = m.get("text", "")
        body = body_text(full_text)
        sec = m.get("section", "UNKNOWN")
        is_ref = bool(m.get("is_reference_list", False))

        # base scores
        dense_s = float(dense_map.get(idx, 0.0))
        bm25_s = float(bm25_scores[idx])
        bm25_scaled = bm25_s / bm25_div

        # -------------------------
        # Chunk-based cancer evidence (IGNORE dataset label)
        # -------------------------
        cancer_delta, cancer_scores = cancer_evidence_adjust(cancer_type, body)

        evidence_ct = ""
        if cancer_scores:
            best_ct = max(cancer_scores, key=cancer_scores.get)
            if cancer_scores.get(best_ct, 0) > 0:
                evidence_ct = best_ct

        # HARD GATE: if query specifies a cancer, and the chunk has 0 evidence for it,
        # and some other cancer has >=1 evidence, drop it.
        if cancer_type and cancer_scores:
            req = cancer_scores.get(cancer_type, 0)
            best_ct = max(cancer_scores, key=cancer_scores.get)
            best = cancer_scores[best_ct]

            # if requested absent AND some other cancer present -> drop
            if req == 0 and best_ct != cancer_type and best >= 1:
                continue

        # -------------------------
        # Soft must-mention gate (only applies in content sections)
        # -------------------------
        mismatch_pen = cancer_keyword_mismatch_penalty(cancer_type, body)

        cancer_pen = 0.0
        req_hits = 0
        if (sec or "").upper() in CONTENT_SECTIONS:
            keep, cancer_pen, reason, req_hits = cancer_must_mention_gate(cancer_type, body)
            if not keep:
                continue

        # -------------------------
        # Entity overlap rerank
        # -------------------------
        overlap = entity_overlap_ratio(q_hi, body)

        # -------------------------
        # Final score
        # -------------------------
        final = dense_s
        final += bm25_weight * bm25_scaled
        final += section_bonus(intent, sec)
        final += entity_bonus_score(entities, body)
        final += cancer_delta
        final -= mismatch_pen
        final -= cancer_pen
        final += min(0.25, 0.06 * req_hits)

        # overlap rerank only when meaningful
        if len(q_hi) >= 2 and overlap > 0.0:
            final += rerank_weight * (overlap ** 1.5)

        # reference penalties
        if not allow_refs:
            if (sec or "").upper() == "REFERENCES":
                final -= 0.12
            if is_ref:
                final -= 0.45

        candidates.append({
            "score": final,
            "dense_score": dense_s,
            "bm25_score": bm25_s,
            "bm25_scaled": bm25_scaled,

            "entity_overlap": overlap,
            "query_hi_entities": q_hi,

            "intent": intent,
            "paper_id": m.get("paper_id"),
            "chunk_id": m.get("chunk_id"),

            # dataset label (may be wrong)
            "label_cancer_type": m.get("cancer_type"),

            "section": sec,
            "is_reference_list": is_ref,
            "content_hash": m.get("content_hash"),
            "text": full_text,

            # debug / analysis fields
            "req_hits": req_hits,
            "cancer_pen": cancer_pen,
            "mismatch_pen": mismatch_pen,
            "cancer_delta": cancer_delta,
            "evidence_ct": evidence_ct,
            "evidence_scores": cancer_scores,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    candidates = dedupe_and_diversify(
        candidates,
        max_per_paper=max_per_paper,
        max_per_hash=max_per_hash
    )
    return candidates[:top_k]

if __name__ == "__main__":
    q = "How do BRAF or KRAS mutations contribute to therapy resistance in Colon Cancer?"
    hits = retrieve(q, top_n_dense=250, top_n_bm25=250, top_k=8, max_per_paper=1, max_per_hash=1, debug=True)

    for h in hits:
        overlap = h.get("entity_overlap", 0.0)
        print(
            f"\n[{h['score']:.3f}] dense={h['dense_score']:.3f} bm25={h['bm25_score']:.2f} "
            f"overlap={overlap:.2f} | label={h['label_cancer_type']} evidence={h['evidence_ct']} "
            f"| {h['section']} | {h['paper_id']}::{h['chunk_id']}"
        )
        print(h["text"][:450].replace("\n", " ") + "...")