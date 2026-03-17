import re
from typing import Dict, List

# -----------------------
# ENTITY EXTRACTION
# -----------------------

# High-signal clinical endpoints + stats
ENDPOINT_PAT = re.compile(
    r"\b(OS|PFS|DFS|RFS|ORR|DCR|HR|CI|AUC|Cmax|Tmax|t1/2|half-?life|PK|PD|ICER|QALY)\b",
    re.IGNORECASE
)

# Immune checkpoints
CHECKPOINT_PAT = re.compile(
    r"\b(PD-?1|PD-?L1|CTLA-?4|LAG-?3|TIM-?3|TIGIT)\b",
    re.IGNORECASE
)

# Pathways
PATHWAY_PAT = re.compile(
    r"\b(MAPK|PI3K/?AKT|AKT|mTOR|WNT|TGF-?β|TGF-?B|JAK/?STAT|NF-?κB|NF-?KB)\b",
    re.IGNORECASE
)

# Drugs: tuned to what appears a lot in your corpus (-nib/-mab/-platin/-taxel/-parib/-ciclib, etc.)
DRUG_PAT = re.compile(
    r"\b[a-zA-Z][a-zA-Z\-]{3,40}"
    r"(?:nib|tinib|mab|zumab|ximab|umab|cept|parib|ciclib|platin|taxel|cycline)\b",
    re.IGNORECASE
)

# Cell lines & common model IDs: A549, H1975, HCC827, NCI-H460, etc.
CELL_LINE_PAT = re.compile(
    r"\b(?:[A-Z]\d{3,5}|H\d{3,5}|HCC\d{3,5}|NCI-?H\d{2,4}|PC-?9)\b",
    re.IGNORECASE
)

# Diagnostics / lab methods (very common in your corpus)
DIAG_PAT = re.compile(
    r"\b(IHC|FISH|PCR|qPCR|ELISA|NGS|WES|RNA-?seq|microarray|immunohistochem)\b",
    re.IGNORECASE
)

# Imaging
IMAGING_PAT = re.compile(
    r"\b(PET/?CT|CT|MRI|DWI|FDG|SUV|max)\b",
    re.IGNORECASE
)

# Gene-ish symbols: MUST start with a letter, contain a letter, length 2-10, allow digits
# Excludes pure numbers + roman numerals later via stoplist

GENE_STOP = {
    # stats/common noise
    "CI","SD","SE","NS","NA","USA","BMC","AJCC","WHO","SEER",
    # imaging/common
    "CT","MRI","PET","FDG","SUV",
    # section-ish/common
    "FIG","TABLE","SUPP","ETAL",
    # roman numerals / phases noise
    "I","II","III","IV","V","VI",
    # generic bio noise
    "DNA","RNA","MRNA",
}

GENE_PAT = re.compile(r"\b[A-Z]{2,}[A-Z0-9]{0,8}\b")

def extract_entities(query: str) -> List[str]:
    q = query or ""
    ents = []

    # High precision buckets
    ents += [m.upper() for m in ENDPOINT_PAT.findall(q)]
    ents += [m.upper() for m in CHECKPOINT_PAT.findall(q)]
    ents += [m.upper() for m in PATHWAY_PAT.findall(q)]
    ents += [m.lower() for m in DRUG_PAT.findall(q)]
    ents += [m.upper() for m in CELL_LINE_PAT.findall(q)]
    ents += [m.upper() for m in DIAG_PAT.findall(q)]
    ents += [m.upper() for m in IMAGING_PAT.findall(q)]

    # Gene symbols (do NOT uppercase whole query)
    for g in GENE_PAT.findall(q):
        if g in GENE_STOP:
            continue
        if re.fullmatch(r"\d+", g):
            continue
        ents.append(g)

    return sorted(set(ents))

# -----------------------
# INTENT ROUTING
# -----------------------

INTENT_RULES = [
    ("cost_effectiveness", [r"\b(cost|cost-?effect|icer|qaly)\b"]),
    ("pharmacokinetics",   [r"\b(pharmacokinetic|pharmacodynamic|\bpk\b|\bpd\b|auc|cmax|tmax|half-?life|t1/2)\b"]),
    ("trial_design",       [r"\b(trial|phase\s*(i|ii|iii|iv)|randomi[sz]ed|double-?blind|placebo|cohort|observational|real-?world)\b"]),
    ("efficacy_outcomes",  [r"\b(efficacy|response rate|orr|dcr|os|pfs|dfs|hazard ratio|hr)\b"]),
    ("adverse_events",     [r"\b(adverse event|toxicity|safety|grade\s*[34]|ctcae)\b"]),
    ("immune_therapy",     [r"\b(checkpoint|immunotherapy|pd-?1|pd-?l1|ctla-?4|tigit|lag-?3)\b"]),
    ("resistance",         [r"\b(resistance|resistant|escape|reversal|mechanism)\b"]),
    ("mutations",          [r"\b(mutation|mutations|variant|variants|kras|braf|egfr|alk|tp53|fusion|rearrangement)\b"]),
    ("biomarkers_dx",      [r"\b(biomarker|diagnos|screen|sensitivity|specificity|roc|auroc)\b"]),
    ("prognosis_signatures",[r"\b(signature|signatures|gene-?expression|multigene|prognos)\b"]),
    ("multi_omics",        [r"\b(multi-?omics|proteomic|transcriptomic|genomic|metabolomic|integrat|rna-?seq)\b"]),
    ("models_cell_lines",  [r"\b(xenograft|mouse|mice|in vivo|animal model|cell line|in vitro|a549|h1975|hcc827)\b"]),
    ("surgery",            [r"\b(surger|resection|laparoscop|tme|lymph node dissection)\b"]),
    ("radiation",          [r"\b(radiotherap|sabr|sb rt|stereotactic|irradiat)\b"]),
    ("epigenetics",        [r"\b(methylation|epigenet|histone|chromatin)\b"]),
    ("microbiome",         [r"\b(microbiome|microbiota)\b"]),
    ("prevention_risk",    [r"\b(risk factor|lifestyle|smoking|diet|prevention)\b"]),
    ("general",            [r".*"]),
]

SECTION_BIAS = {
    "pharmacokinetics":     {"METHODS": 0.25, "MATERIALS AND METHODS": 0.25, "RESULTS": 0.15, "DISCUSSION": 0.05},
    "cost_effectiveness":   {"RESULTS": 0.20, "DISCUSSION": 0.15, "METHODS": 0.10},
    "trial_design":         {"METHODS": 0.25, "RESULTS": 0.15, "DISCUSSION": 0.05},
    "efficacy_outcomes":    {"RESULTS": 0.25, "DISCUSSION": 0.10, "METHODS": 0.05},
    "adverse_events":       {"RESULTS": 0.20, "METHODS": 0.10, "DISCUSSION": 0.10},
    "immune_therapy":       {"RESULTS": 0.20, "DISCUSSION": 0.15, "INTRODUCTION": 0.05},
    "resistance":           {"RESULTS": 0.20, "DISCUSSION": 0.20, "METHODS": 0.10},
    "mutations":            {"RESULTS": 0.20, "METHODS": 0.15, "DISCUSSION": 0.10},
    "biomarkers_dx":        {"RESULTS": 0.20, "METHODS": 0.15, "DISCUSSION": 0.10},
    "prognosis_signatures": {"RESULTS": 0.20, "DISCUSSION": 0.20, "METHODS": 0.10},
    "multi_omics":          {"METHODS": 0.20, "RESULTS": 0.20, "DISCUSSION": 0.10},
    "models_cell_lines":    {"METHODS": 0.25, "MATERIALS AND METHODS": 0.25, "RESULTS": 0.10},
    "surgery":              {"METHODS": 0.20, "RESULTS": 0.20, "DISCUSSION": 0.10},
    "radiation":            {"METHODS": 0.15, "RESULTS": 0.20, "DISCUSSION": 0.15},
    "epigenetics":          {"RESULTS": 0.20, "DISCUSSION": 0.15, "METHODS": 0.10},
    "microbiome":           {"RESULTS": 0.20, "DISCUSSION": 0.15, "INTRODUCTION": 0.05},
    "prevention_risk":      {"RESULTS": 0.15, "DISCUSSION": 0.20, "INTRODUCTION": 0.10},
    "general":              {"RESULTS": 0.10, "DISCUSSION": 0.10},
}


SECTION_PENALTY = {
    "REFERENCES": -0.40,
    "ACKNOWLEDGEMENTS": -0.50,
    "ACKNOWLEDGMENTS": -0.50,
    "FUNDING": -0.50,
    "CONFLICT OF INTEREST": -0.50,
    "COMPETING INTERESTS": -0.50,
    "DATA AVAILABILITY": -0.30,
    "SUPPLEMENTARY INFORMATION": -0.30,
    "SUPPLEMENTARY MATERIAL": -0.30,
    "TRIAL REGISTRATION": -0.30,
    "UNKNOWN": -0.05,
    "ABSTRACT": -0.05,
}

CANCER_PATTERNS = {
    "Thyroid_Cancer": [
        r"\bthyroid cancer\b",
        r"\bthyroid carcinoma\b",
        r"\bpapillary thyroid\b",
        r"\bfollicular thyroid\b",
        r"\bmedullary thyroid\b",
        r"\banaplastic thyroid\b",
        r"\bptc\b", r"\bftc\b", r"\bmtc\b", r"\batc\b",
    ],
    "Lung_Cancer": [
        r"\blung cancer\b",
        r"\bnsclc\b",
        r"\bsclc\b",
        r"\blung carcinoma\b",
        r"\badenocarcinoma of the lung\b",
        r"\bsquamous cell lung\b",
    ],
    "Colon_Cancer": [
        r"\bcolon cancer\b",
        r"\bcolorectal cancer\b",
        r"\brectal cancer\b",
        r"\bcrc\b",
        r"\bmcrc\b",
    ],
}

def infer_intent(query: str) -> str:
    q = (query or "").lower()
    for name, pats in INTENT_RULES:
        if any(re.search(p, q, re.IGNORECASE) for p in pats):
            return name
    return "general"

def section_bonus(intent: str, section: str) -> float:
    sec = (section or "UNKNOWN").upper()
    bonus = SECTION_BIAS.get(intent, {}).get(sec, 0.0)
    penalty = SECTION_PENALTY.get(sec, 0.0)
    return bonus + penalty

def extract_cancer_type(query: str) -> str:
    q = query or ""
    for cancer_type, patterns in CANCER_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q, re.IGNORECASE):
                return cancer_type
    return ""