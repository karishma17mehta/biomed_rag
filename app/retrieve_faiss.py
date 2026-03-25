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
from app.config import CFG

# ── Config ────────────────────────────────────────────────────────────────────
_R = CFG["retrieval"]
_M = CFG["models"]
_P = CFG["paths"]

INDEX_DIR  = Path(_P["index_dir"])
FAISS_PATH = Path(_P["faiss_index"])
META_PATH  = Path(_P["meta_file"])
BM25_PATH  = Path(_P["bm25_file"])
MODEL      = _M["embedding"]

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment.")
client = OpenAI(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Cancer term dictionaries
# ─────────────────────────────────────────────────────────────────────────────
CANCER_TERMS = {
    "Colon_Cancer": {
        "must": [
            "colorectal", "colon", "rectal", "rectum",
            "crc", "mcrc", "sigmoid", "cecum", "caecum",
            "large intestine", "bowel",
        ],
        "anti": [
            "thyroid", "papillary thyroid", "follicular thyroid", "medullary thyroid",
            "nsclc", "sclc", "non-small cell lung", "small cell lung", "pulmonary",
        ],
    },
    "Lung_Cancer": {
        "must": [
            "lung", "pulmonary", "nsclc", "sclc",
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
            "thyroid", "papillary thyroid", "follicular thyroid",
            "medullary thyroid", "anaplastic thyroid", "differentiated thyroid",
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

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
SEP_RE = re.compile(r"\s*---\s*", re.MULTILINE)

def body_text(full_text: str) -> str:
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

def entity_bonus_score(entities: List[str], text: str) -> float:
    if not entities or not text:
        return 0.0
    t = text.lower()
    hits = sum(1 for e in entities if e.lower() in t)
    return min(_R["entity_bonus_cap"], hits * _R["entity_bonus_per_hit"])

def dedupe_and_diversify(
    candidates: List[Dict[str, Any]],
    max_per_paper: int = 1,
    max_per_hash: int = 1,
) -> List[Dict[str, Any]]:
    seen_hash, per_paper, kept = {}, {}, []
    for c in candidates:
        h   = c.get("content_hash") or ""
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
        return _R["mismatch_strong_penalty"]
    if best_other >= 2:
        return _R["mismatch_weak_penalty"]
    return 0.0

def load_bm25():
    with BM25_PATH.open("rb") as f:
        payload = pickle.load(f)
    return payload["bm25"]

TOKEN_RE = re.compile(r"[A-Za-z0-9\-_/]+")

def bm25_tokenize(text: str):
    return TOKEN_RE.findall((text or "").lower())

def count_terms_unique(text: str, terms: list) -> int:
    t = (text or "").lower()
    t_compact = re.sub(r"\s+", "", t)
    hits = 0
    for term in terms:
        term_l = term.lower()
        term_compact = re.sub(r"\s+", "", term_l)
        if term_l in t:
            hits += 1
            continue
        if 2 <= len(term_compact) <= 12 and term_compact in t_compact:
            hits += 1
    return hits

def cancer_must_mention_gate(requested: str, body: str) -> Tuple[bool, float, str, int]:
    if not requested:
        return True, 0.0, "no requested cancer", 0
    spec     = CANCER_TERMS.get(requested, {})
    must     = spec.get("must", [])
    anti     = spec.get("anti", [])
    req_hits  = count_terms_unique(body, must)
    anti_hits = count_terms_unique(body, anti)
    if req_hits == 0 and anti_hits >= 2:
        return True, _R["must_strong_other_penalty"], "no requested; strong other-cancer signal", req_hits
    if req_hits == 0 and anti_hits == 1:
        return True, _R["must_weak_other_penalty"],   "no requested; weak other-cancer hint",    req_hits
    if req_hits == 0 and anti_hits == 0:
        return True, _R["must_generic_penalty"],      "generic/no cancer mention",               req_hits
    return True, 0.0, "mentions requested cancer", req_hits

# ─────────────────────────────────────────────────────────────────────────────
# High-signal entity extraction
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PAT = re.compile(r"\b(PD-?1|PD-?L1|CTLA-?4|LAG-?3|TIM-?3|TIGIT)\b", re.IGNORECASE)
PATHWAY_PAT    = re.compile(r"\b(MAPK|PI3K/?AKT|AKT|mTOR|WNT|TGF-?β|TGF-?B|JAK/?STAT|NF-?κB|NF-?KB)\b", re.IGNORECASE)
DRUG_PAT       = re.compile(
    r"\b[a-zA-Z][a-zA-Z\-]{3,40}"
    r"(?:nib|tinib|mab|zumab|ximab|umab|cept|parib|ciclib|platin|taxel|cycline)\b",
    re.IGNORECASE,
)
GENE_PAT  = re.compile(r"\b[A-Z]{2,}[A-Z0-9]{0,8}\b")
GENE_STOP = {
    "CI","SD","SE","NS","NA","USA","BMC","AJCC","WHO","SEER",
    "DNA","RNA","MRNA","FIG","TABLE","SUPP","ETAL",
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
        if g in GENE_STOP or re.fullmatch(r"\d+", g) or len(g) < 2:
            continue
        ents.append(g)
    return sorted(set(ents))

def normalize_for_match(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("/", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s)

def entity_overlap_ratio(query_ents: List[str], text: str) -> float:
    if not query_ents:
        return 0.0
    raw = text or ""
    blob_norm          = normalize_for_match(raw)
    blob_upper_nospace = re.sub(r"\s+", "", raw.upper())
    blob_lower_nospace = re.sub(r"\s+", "", raw.lower())
    found = 0
    for e in query_ents:
        e_norm = normalize_for_match(e)
        if re.search(rf"\b{re.escape(e_norm)}\b", blob_norm, flags=re.IGNORECASE):
            found += 1
            continue
        e_compact_upper = re.sub(r"[\s\-_/]+", "", e.upper())
        e_compact_lower = re.sub(r"[\s\-_/]+", "", e.lower())
        if 2 <= len(e_compact_upper) <= 12:
            if e_compact_upper in blob_upper_nospace or e_compact_lower in blob_lower_nospace:
                found += 1
    return found / max(1, len(query_ents))

RESIST_PAT = re.compile(r"\b(resistan|refractor|escape|non[- ]?respond|sensitiz|overcome)\w*", re.I)

def intent_evidence_bonus(intent: str, body: str) -> float:
    if intent == "resistance":
        return 0.35 if RESIST_PAT.search(body or "") else -0.25
    return 0.0

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()

def count_terms_unique_nonoverlap(text: str, terms: list) -> int:
    t = _normalize(text)
    if not t or not terms:
        return 0
    terms_sorted = sorted(set(_normalize(x) for x in terms if x), key=len, reverse=True)
    hits = 0
    for term in terms_sorted:
        pat = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)")
        m = pat.search(t)
        if m:
            hits += 1
            t = t[:m.start()] + (" " * (m.end() - m.start())) + t[m.end():]
    return hits

def cancer_evidence_scores(body: str) -> Dict[str, int]:
    return {
        ct: count_terms_unique_nonoverlap(body or "", spec.get("must", []))
        for ct, spec in CANCER_TERMS.items()
    }

def cancer_evidence_adjust(requested: str, body: str) -> Tuple[float, Dict[str, int]]:
    if not requested:
        return 0.0, {}
    scores = cancer_evidence_scores(body)
    if not scores:
        return 0.0, {}
    req     = scores.get(requested, 0)
    best_ct = max(scores, key=scores.get)
    best    = scores[best_ct]
    if best == 0:
        return _R["cancer_generic_penalty"], scores
    if req > 0:
        return min(_R["cancer_present_cap"], _R["cancer_present_per_hit"] * req), scores
    margin = best - req
    if margin >= 2:
        return _R["cancer_absent_strong_penalty"], scores
    return _R["cancer_absent_weak_penalty"], scores


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    top_n_dense: int  = None,
    top_n_bm25: int   = None,
    top_k: int        = None,
    max_per_paper: int = None,
    max_per_hash: int  = None,
    bm25_div: float   = None,
    bm25_weight: float = None,
    rerank_weight: float = None,
    debug: bool = False,
):
    # Fall back to config defaults
    top_n_dense   = top_n_dense   or _R["top_n_dense"]
    top_n_bm25    = top_n_bm25    or _R["top_n_bm25"]
    top_k         = top_k         or _R["top_k"]
    max_per_paper = max_per_paper or _R["max_per_paper"]
    max_per_hash  = max_per_hash  or _R["max_per_hash"]
    bm25_div      = bm25_div      or _R["bm25_div"]
    rerank_weight = rerank_weight or _R["rerank_weight"]

    intent      = infer_intent(query)
    entities    = extract_entities(query)
    cancer_type = extract_cancer_type(query)
    allow_refs  = wants_citations(query, intent)
    q_hi        = extract_highsignal_entities(query)

    if debug:
        print("intent:", intent)
        print("entities:", entities)
        print("cancer_type:", cancer_type)
        print("q_hi:", q_hi)

    index  = faiss.read_index(str(FAISS_PATH))
    metas  = load_meta()

    # Dense retrieval
    qv = embed_query(query).reshape(1, -1)
    dense_scores, dense_ids = index.search(qv, top_n_dense)
    dense_map = {i: float(s) for i, s in zip(dense_ids[0], dense_scores[0]) if i >= 0}

    # BM25 retrieval
    bm25        = load_bm25()
    bm25_query  = query + (" " + " ".join(q_hi) if q_hi else "")
    q_tokens    = bm25_tokenize(bm25_query)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top_ids = list(np.argsort(bm25_scores)[::-1])[:top_n_bm25]

    # Dynamic BM25 weight
    if len(entities) >= 2:
        bm25_weight = _R["bm25_weight_with_entities"]
    elif entities:
        bm25_weight = _R["bm25_weight_with_entities"]
    else:
        bm25_weight = _R["bm25_weight_no_entities"]

    CONTENT_SECTIONS = {
        "ABSTRACT", "INTRODUCTION", "RESULTS", "DISCUSSION",
        "CONCLUSION", "BACKGROUND", "OBJECTIVE"
    }

    candidates = []
    for idx in set(dense_map.keys()) | set(bm25_top_ids):
        m         = metas[idx]
        full_text = m.get("text", "")
        body      = body_text(full_text)
        sec       = m.get("section", "UNKNOWN")
        is_ref    = bool(m.get("is_reference_list", False))

        dense_s      = float(dense_map.get(idx, 0.0))
        bm25_s       = float(bm25_scores[idx])
        bm25_scaled  = bm25_s / bm25_div

        cancer_delta, cancer_scores = cancer_evidence_adjust(cancer_type, body)

        evidence_ct = ""
        if cancer_scores:
            best_ct = max(cancer_scores, key=cancer_scores.get)
            if cancer_scores.get(best_ct, 0) > 0:
                evidence_ct = best_ct

        # Hard gate
        if cancer_type and cancer_scores:
            req     = cancer_scores.get(cancer_type, 0)
            best_ct = max(cancer_scores, key=cancer_scores.get)
            if req == 0 and best_ct != cancer_type and cancer_scores[best_ct] >= 1:
                continue

        mismatch_pen = cancer_keyword_mismatch_penalty(cancer_type, body)
        cancer_pen   = 0.0
        req_hits     = 0
        if (sec or "").upper() in CONTENT_SECTIONS:
            _, cancer_pen, _, req_hits = cancer_must_mention_gate(cancer_type, body)

        overlap = entity_overlap_ratio(q_hi, body)

        final  = dense_s
        final += bm25_weight * bm25_scaled
        final += section_bonus(intent, sec)
        final += entity_bonus_score(entities, body)
        final += cancer_delta
        final -= mismatch_pen
        final -= cancer_pen
        final += min(_R["req_hits_bonus_cap"], _R["req_hits_bonus_per_hit"] * req_hits)

        if len(q_hi) >= _R["rerank_min_entities"] and overlap > 0.0:
            final += rerank_weight * (overlap ** 1.5)

        if not allow_refs:
            if (sec or "").upper() == "REFERENCES":
                final -= _R["reference_section_penalty"]
            if is_ref:
                final -= _R["reference_flag_penalty"]

        candidates.append({
            "score": final, "dense_score": dense_s,
            "bm25_score": bm25_s, "bm25_scaled": bm25_scaled,
            "entity_overlap": overlap, "query_hi_entities": q_hi,
            "intent": intent, "paper_id": m.get("paper_id"),
            "chunk_id": m.get("chunk_id"),
            "label_cancer_type": m.get("cancer_type"),
            "section": sec, "is_reference_list": is_ref,
            "content_hash": m.get("content_hash"), "text": full_text,
            "req_hits": req_hits, "cancer_pen": cancer_pen,
            "mismatch_pen": mismatch_pen, "cancer_delta": cancer_delta,
            "evidence_ct": evidence_ct, "evidence_scores": cancer_scores,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = dedupe_and_diversify(candidates, max_per_paper=max_per_paper, max_per_hash=max_per_hash)
    return candidates[:top_k]


if __name__ == "__main__":
    q = "How do BRAF or KRAS mutations contribute to therapy resistance in Colon Cancer?"
    hits = retrieve(q, top_n_dense=250, top_n_bm25=250, top_k=8, debug=True)
    for h in hits:
        print(
            f"\n[{h['score']:.3f}] dense={h['dense_score']:.3f} overlap={h['entity_overlap']:.2f} "
            f"| {h['section']} | {h['paper_id']}::{h['chunk_id']}"
        )
        print(h["text"][:400].replace("\n", " ") + "...")