import re
from typing import Dict, List

from app.entities import (
    CHECKPOINT_PAT, PATHWAY_PAT, DRUG_PAT, GENE_PAT, GENE_STOP,
    CANCER_PATTERNS, extract_entities, extract_cancer_type,
)

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