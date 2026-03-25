# app/entities.py
"""
Shared entity patterns, cancer term dictionaries, and extraction functions.
Single source of truth — imported by query_router.py and retrieve_faiss.py.
"""

import re
from typing import List, Dict

# ─────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────

CHECKPOINT_PAT = re.compile(
    r"\b(PD-?1|PD-?L1|CTLA-?4|LAG-?3|TIM-?3|TIGIT)\b",
    re.IGNORECASE
)

PATHWAY_PAT = re.compile(
    r"\b(MAPK|PI3K/?AKT|AKT|mTOR|WNT|TGF-?β|TGF-?B|JAK/?STAT|NF-?κB|NF-?KB)\b",
    re.IGNORECASE
)

DRUG_PAT = re.compile(
    r"\b[a-zA-Z][a-zA-Z\-]{3,40}"
    r"(?:nib|tinib|mab|zumab|ximab|umab|cept|parib|ciclib|platin|taxel|cycline)\b",
    re.IGNORECASE,
)

CELL_LINE_PAT = re.compile(
    r"\b(?:[A-Z]\d{3,5}|H\d{3,5}|HCC\d{3,5}|NCI-?H\d{2,4}|PC-?9)\b",
    re.IGNORECASE
)

DIAG_PAT = re.compile(
    r"\b(IHC|FISH|PCR|qPCR|ELISA|NGS|WES|RNA-?seq|microarray|immunohistochem)\b",
    re.IGNORECASE
)

ENDPOINT_PAT = re.compile(
    r"\b(OS|PFS|DFS|RFS|ORR|DCR|HR|CI|AUC|Cmax|Tmax|t1/2|half-?life|PK|PD|ICER|QALY)\b",
    re.IGNORECASE
)

IMAGING_PAT = re.compile(
    r"\b(PET/?CT|CT|MRI|DWI|FDG|SUV|max)\b",
    re.IGNORECASE
)

GENE_PAT = re.compile(r"\b[A-Z]{2,}[A-Z0-9]{0,8}\b")

GENE_STOP = {
    # stats / common noise
    "CI", "SD", "SE", "NS", "NA", "USA", "BMC", "AJCC", "WHO", "SEER",
    # imaging
    "CT", "MRI", "PET", "FDG", "SUV",
    # section noise
    "FIG", "TABLE", "SUPP", "ETAL",
    # roman numerals
    "I", "II", "III", "IV", "V", "VI",
    # generic bio
    "DNA", "RNA", "MRNA",
    # clinical endpoints (keep out of gene list)
    "OS", "PFS", "DFS", "RFS", "ORR", "DCR", "HR", "PK", "PD", "AUC",
}


# ─────────────────────────────────────────────────────────────
# Cancer dictionaries
# ─────────────────────────────────────────────────────────────

# Full term lists for evidence scoring (retrieve_faiss.py)
CANCER_TERMS: Dict[str, Dict[str, List[str]]] = {
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

# Short keyword lists for fast mismatch detection (retrieve_faiss.py)
CANCER_KEYWORDS: Dict[str, List[str]] = {
    "Thyroid_Cancer": ["thyroid", "papillary", "follicular", "medullary", "anaplastic"],
    "Colon_Cancer":   ["colon", "colorectal", "crc", "rectal"],
    "Lung_Cancer":    ["lung", "nsclc", "sclc", "adenocarcinoma", "squamous"],
}

# Regex patterns for cancer type detection (query_router.py)
CANCER_PATTERNS: Dict[str, List[str]] = {
    "Thyroid_Cancer": [
        r"\bthyroid cancer\b", r"\bthyroid carcinoma\b",
        r"\bpapillary thyroid\b", r"\bfollicular thyroid\b",
        r"\bmedullary thyroid\b", r"\banaplastic thyroid\b",
        r"\bptc\b", r"\bftc\b", r"\bmtc\b", r"\batc\b",
    ],
    "Lung_Cancer": [
        r"\blung cancer\b", r"\bnsclc\b", r"\bsclc\b",
        r"\blung carcinoma\b", r"\badenocarcinoma of the lung\b",
        r"\bsquamous cell lung\b",
    ],
    "Colon_Cancer": [
        r"\bcolon cancer\b", r"\bcolorectal cancer\b",
        r"\brectal cancer\b", r"\bcrc\b", r"\bmcrc\b",
    ],
}


# ─────────────────────────────────────────────────────────────
# Extraction functions
# ─────────────────────────────────────────────────────────────

def extract_entities(query: str) -> List[str]:
    """
    Extract biomedical entities from a query string.
    Used by query_router.py for intent routing and section bias.
    """
    q = query or ""
    ents: List[str] = []

    ents += [m.upper() for m in ENDPOINT_PAT.findall(q)]
    ents += [m.upper() for m in CHECKPOINT_PAT.findall(q)]
    ents += [m.upper() for m in PATHWAY_PAT.findall(q)]
    ents += [m.lower() for m in DRUG_PAT.findall(q)]
    ents += [m.upper() for m in CELL_LINE_PAT.findall(q)]
    ents += [m.upper() for m in DIAG_PAT.findall(q)]
    ents += [m.upper() for m in IMAGING_PAT.findall(q)]

    for g in GENE_PAT.findall(q):
        if g in GENE_STOP or re.fullmatch(r"\d+", g) or len(g) < 2:
            continue
        ents.append(g)

    return sorted(set(ents))


def extract_highsignal_entities(text: str) -> List[str]:
    """
    Extract high-signal biomedical entities from chunk text.
    Used by retrieve_faiss.py for reranking.
    Same logic as extract_entities but operates on chunk body text.
    """
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


def extract_cancer_type(query: str) -> str:
    """
    Detect cancer type from query string using regex patterns.
    Returns one of: 'Thyroid_Cancer', 'Lung_Cancer', 'Colon_Cancer', or ''.
    """
    q = query or ""
    for cancer_type, patterns in CANCER_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q, re.IGNORECASE):
                return cancer_type
    return ""