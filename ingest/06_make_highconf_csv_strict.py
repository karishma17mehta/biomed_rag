#!/usr/bin/env python3
import re
import json
import pandas as pd
from pathlib import Path

IN_PATH  = Path("outputs/phase2_filtered_papers.csv")
OUT_PATH = Path("outputs/phase2_highconf_papers_strict.csv")
REPORT   = Path("outputs/label_conf_report_strict.json")
MISSES   = Path("outputs/label_conf_misses_strict.csv")

LABEL_COL    = "cancer_type"
TEXT_COL     = "clean_text"
PAPER_ID_COL = "paper_id"

MAX_MISSES_PER_LABEL = 80

# ----------------------------
# Core signals
# ----------------------------
CANCER_RE = re.compile(
    r"\b(cancer|carcinoma|tumou?r|neoplasm|malignan|metastas(?:is|e|es|ed|ing)?|oncolog|adenocarcinoma)\w*\b",
    re.I
)

# A small window where we want site+ cancer to appear together (prevents false positives)
PROX_WINDOW_CHARS = 320

# ----------------------------
# Site keyword sets (CORE vs SECONDARY)
# ----------------------------
KEYWORDS_CORE = {
    "Lung_Cancer": [
        r"\blung(s)?\b",
        r"\bpulmonar(y|ies)\b",
        r"\bnsclc\b",
        r"\bsclc\b",
        r"\bnon[-\s]?small\s+cell\b",
        r"\bsmall\s+cell\b",
        r"\bbronch(us|ial|ogenic)?\b",
        r"\bpleura(l)?\b",
        r"\bthorac(ic|ax)\b",
        r"\bmesothelioma\b",
    ],
    "Colon_Cancer": [
        r"\bneoplasia\b",
        r"\bcrn\b",
        r"\bcolon\b",
        r"\bcolorectal\b",
        r"\bcrc\b",
        r"\brectal\b",
        r"\bcolonic\b",
        r"\b(sigmoid|cecum|caecum|ileocecal|appendix)\b",
        r"\blarge\s+intestine\b",
        r"\bcolonoscopy\b",
        r"\bpolyp(s)?\b",
        r"\badenoma(s)?\b",
        r"\b(hnpcc|lynch)\b",
        r"\bfap\b",
        r"\bcolorect(al)?\s+carcinoma\b",
        r"\bcolorectal\s+adenocarcinoma\b",
        r"\brectum\b",
        r"\bcecal\b",
        r"\bcolectomy\b",
        r"\bhemicolectomy\b",
        r"\bmetastatic\s+colorectal\b",
    ],
    "Thyroid_Cancer": [
        r"\bthyroid\b",
        r"\bthyroidectomy\b",
        r"\bthyroglobulin\b",
        r"\btsh\b",
        r"\bpapillary\b",
        r"\bfollicular\b",
        r"\bmedullary\b",
        r"\banaplastic\b",
        r"\b(ptc|ftc|mtc|atc)\b",
        r"\bhashimoto\b",
        r"\bgraves\b",
        r"\bdifferentiated\s+thyroid\s+cancer\b",
        r"\bthyroid\s+carcinoma\b",
        r"\bmetastatic\s+thyroid\b",
        r"\brai\b", 
        r"\bradioiodine\b", 
        r"\bi[-\s]?131\b",
        r"\bbethesda\b", 
        r"\bfna\b", 
        r"\bfine\s+needle\s+aspiration\b",
        r"\bbraf\b", 
        r"\bv600e\b", 
        r"\btert\b", 
        r"\bret\b",
        r"\bnif?tp\b",  # NIFTP appears a lot in thyroid pathology
        r"\bthy\s*\d\b",  # Thy1/Thy2/Thy3... classification sometimes appears
        r"\bthyroid\s+carcinoma\b",
        r"\bdifferentiated\s+thyroid\s+cancer\b",
        r"\bpoorly\s+differentiated\s+thyroid\b",
        r"\bradioiodine\b", r"\brai\b", r"\bi[-\s]?131\b",
        r"\bfine\s+needle\s+aspiration\b", r"\bfna\b", r"\bbethesda\b",
        r"\bbraf\b", r"\bv600e\b", r"\btert\b", r"\bret\b",
        r"\bnif?tp\b",
    ],
}

# Secondary signals help rescue true papers that use clinical shorthand,
# BUT we should not allow secondary-only matches.
KEYWORDS_SECONDARY = {
    "Colon_Cancer": [
        r"\bbowel\b",
        r"\bibd\b",
        r"\bulcerative\s+colitis\b",
        r"\bcrohn(?:'s)?\b",
        r"\bdiverticul(itis|osis)\b",
        r"\b(colon|colorectal)\s+adenocarcinoma\b",
        r"\bmetastatic\s+colorectal\b",
        r"\brectum\b",
        r"\bcecal\b", 
        r"\bcaecal\b",
        r"\bileum\b", 
        r"\bileal\b",               # appears in CRC surgical contexts
        r"\bcolectomy\b", 
        r"\bproctectomy\b",
        r"\bcrc\s+cell\b",                        # “CRC cells”
        r"\bapc\b", 
        r"\bkras\b", 
        r"\bbraf\b",     # CRC genetics common
    ],
    "Thyroid_Cancer": [
        r"\bnodule(s)?\b",
        r"\bgoit(re|er)\b",
        r"\bdtc\b",                               # differentiated thyroid cancer
        r"\bradioiodine\b", 
        r"\brai\b",           # RAI / radioiodine
        r"\bi[-\s]?131\b",                        # I-131
        r"\bfna\b", 
        r"\bfine\s+needle\s+aspiration\b",
        r"\bbethesda\b",                          # Bethesda thyroid cytology
        r"\bbraf\b", 
        r"\bv600e\b",                # BRAF V600E
        r"\btert\b", 
        r"\bret\b",                  # TERT / RET (common in thyroid ca)
        r"\blevothyroxine\b",                     # thyroid suppression context
        r"\bthyrotoxic\b",                        # sometimes appears in thyroid-related oncology/endocrine papers
    
    ],
    "Lung_Cancer": [
        r"\balveol(ar|i)\b",
        r"\btrachea(l)?\b",
        r"\b(emphysema|chronic\s+bronchitis)\b",
    ],
}

SITE_CORE_RE = {k: re.compile("|".join(v), re.I) for k, v in KEYWORDS_CORE.items()}
SITE_SEC_RE  = {k: re.compile("|".join(v), re.I) for k, v in KEYWORDS_SECONDARY.items()}

THYROID_CANCER_RE = re.compile(
    r"\b(papillary|follicular|medullary|anaplastic)\s+thyroid\s+(carcinoma|cancer)\b",
    re.I
)

# ----------------------------
# Wrong-topic blacklists (lightweight, high payoff)
# ----------------------------
NEG_GLOBAL_NEURO = re.compile(
    r"\b(alzheimer|parkinson|dementia|amyotrophic|als\b|neuron|neuronal|brain|hippocamp|olfactory|glial)\b",
    re.I
)

NEG_BY_LABEL = {
    "Thyroid_Cancer": re.compile(r"\b(covid|sars[-\s]?cov[-\s]?2|disinfectant|steriliz|pandemic)\b", re.I),
}

CROSS_SITE_STRONG = {
  "Colon_Cancer": re.compile("|".join(KEYWORDS_CORE["Lung_Cancer"]), re.I),
  "Lung_Cancer": re.compile("|".join(KEYWORDS_CORE["Colon_Cancer"]), re.I),
  "Thyroid_Cancer": re.compile("|".join(KEYWORDS_CORE["Lung_Cancer"] + KEYWORDS_CORE["Colon_Cancer"]), re.I),
}

OTHER_SITE_TERMS = {
    "brain": r"\b(glioma|glioblastoma|brain|astrocytoma|meningioma)\b",
    "breast": r"\b(breast cancer|mammary)\b",
    "prostate": r"\bprostate\b",
}

# ----------------------------
# Helper: site+cancer proximity check
# ----------------------------

OTHER_CANCER_SITES_RE = re.compile(
    r"\b(breast|mammary|prostate|ovarian|cervical|endometrial|pancrea|gastric|stomach|hepatic|liver|renal|kidney|bladder|melanoma|glioma|leukemia|lymphoma|myeloma)\b",
    re.I
)



ONCO_CONTEXT_RE = re.compile(
    r"\b(tumou?r|carcinoma|metasta|oncolog|chemotherap|radiotherap|immunotherap|survival|prognos|stage\s+[ivx]+|hazard\s+ratio|overall\s+survival|disease[-\s]?free)\b",
    re.I
)

# Neuro words that are truly "wrong topic"
NEG_GLOBAL_NEURO = re.compile(
    r"\b(alzheimer|parkinson|dementia|amyotrophic|als\b)\b",
    re.I
)

# Softer: allow “brain”, “neuronal”, etc. (too common in cancer biology).
NEURO_SOFT_RE = re.compile(
    r"\b(neuron|neuronal|brain|hippocamp|olfactory|glial)\b", re.I
)

def neuro_is_probably_wrong_topic(text: str) -> bool:
    # hard neuro disease terms => wrong topic unless very oncology-heavy
    if NEG_GLOBAL_NEURO.search(text) and not ONCO_CONTEXT_RE.search(text):
        return True
    return False


def has_site_cancer_proximity(text: str, site_re: re.Pattern, cancer_re: re.Pattern, window: int) -> bool:
    """
    Returns True if we can find at least one occurrence where site term and cancer term
    appear within 'window' characters of each other (either order).
    """
    # Fast path: if either missing, skip
    if not site_re.search(text) or not cancer_re.search(text):
        return False

    # Find a few site matches; don’t iterate thousands
    site_iter = list(site_re.finditer(text))
    if not site_iter:
        return False

    # Limit to first N site matches to keep it fast
    for m in site_iter[:30]:
        start = max(0, m.start() - window)
        end   = min(len(text), m.end() + window)
        snippet = text[start:end]
        if cancer_re.search(snippet):
            return True
    return False

# ----------------------------
# Extra compiled helpers (ADD ONCE)
# ----------------------------
THYROID_STRONG_RE = re.compile(
    r"\b("
    r"thyroid\s+(?:cancer|carcinoma|microcarcinoma|neoplasm|malignan(?:cy|t)|tumou?r)|"
    r"(?:papillary|follicular|medullary|anaplastic|poorly\s+differentiated)\s+thyroid\s+(?:carcinoma|cancer|microcarcinoma)|"
    r"\bDTC\b|\bPTC\b|\bPTMC\b|\bMTC\b|\bATC\b|\bFTC\b"
    r")\b",
    re.I
)

COLON_STRONG_RE = re.compile(
    r"\b("
    r"colorectal\s+(?:cancer|carcinoma|adenocarcinoma)|"
    r"colon\s+(?:cancer|carcinoma|adenocarcinoma)|"
    r"\bCRC\b|\bHNPCC\b|\bLynch\b|\bFAP\b|"
    r"colectomy|hemicolectomy|colonoscopy"
    r")\b",
    re.I
)

LUNG_STRONG_RE = re.compile(
    r"\b("
    r"lung\s+(?:cancer|carcinoma)|"
    r"\bNSCLC\b|\bSCLC\b|"
    r"non[-\s]?small\s+cell|small\s+cell|"
    r"pulmonar(?:y|ies)|bronch(?:us|ial)|mesothelioma"
    r")\b",
    re.I
)

# strong OTHER primary-site indicators (avoid generic "brain")
OTHER_PRIMARY_SITE_RE = re.compile(
    r"\b("
    r"glioblastoma|astrocytoma|meningioma|"
    r"breast\s+cancer|mammary\s+carcinoma|"
    r"prostate\s+cancer|"
    r"ovarian\s+cancer|"
    r"pancreatic\s+cancer|"
    r"gastric\s+cancer|stomach\s+cancer|"
    r"hepatocellular|liver\s+cancer|"
    r"renal\s+cell|kidney\s+cancer|"
    r"bladder\s+cancer|"
    r"leukemia|lymphoma|myeloma"
    r"breast cancer|mammary|hepatocellular carcinoma|hcc\b|liver cancer|"
    r"acute myeloid leukemia|aml\b|leukemia|lymphoma|myeloma|"
    r"prostate cancer|ovarian cancer|cervical cancer|endometrial cancer|"
    r"glioblastoma|glioma|astrocytoma|"
    r"pancreatic cancer|gastric cancer|stomach cancer|renal cancer|kidney cancer|"
    r"melanoma"
    r")\b",
    
    re.I
)

# hard neuro disease terms only (keep this strict)
NEURO_DISEASE_RE = re.compile(r"\b(alzheimer|parkinson|dementia|amyotrophic\s+lateral\s+sclerosis|\bALS\b)\b", re.I)

def count_hits(rx: re.Pattern, text: str, cap: int = 200) -> int:
    # count occurrences but cap to avoid pathological docs slowing you down
    c = 0
    for _ in rx.finditer(text):
        c += 1
        if c >= cap:
            break
    return c

def compute_is_highconf(label: str, text: str):
    if not isinstance(text, str) or not text.strip():
        return False, "empty"
    if label not in SITE_CORE_RE:
        return False, "unknown_label"

    # 1) Must have cancer-ish language somewhere
    if not CANCER_RE.search(text):
        return False, "no_cancer_signal_anywhere"

    # 2) label-specific negatives
    neg_label = NEG_BY_LABEL.get(label)
    if neg_label and neg_label.search(text):
        return False, "negative_filter_hit"

    # 3) hard wrong-topic neuro disease papers (keep strict)
    if NEURO_DISEASE_RE.search(text) and not ONCO_CONTEXT_RE.search(text):
        return False, "negative_neuro_wrong_topic"

    core_re = SITE_CORE_RE[label]
    sec_re  = SITE_SEC_RE.get(label)

    core_hit = bool(core_re.search(text))
    sec_hit  = bool(sec_re.search(text)) if sec_re else False
    if not core_hit and not sec_hit:
        return False, "no_site_signal_anywhere"

    # 4) Strong cross-site mismatch (only if other site appears AND label site is weak)
    cross = CROSS_SITE_STRONG.get(label)
    if cross and cross.search(text) and not core_hit:
        return False, "cross_site_mismatch"

    # 5) count site hits properly
    site_hits = 0
    for _ in core_re.finditer(text):
        site_hits += 1
        if site_hits >= 20:
            break

    if site_hits < 1:
        return False, "weak_site_presence"

    # 6) Proximity check (nice to have)
    prox_ok = False
    if core_hit and has_site_cancer_proximity(text, core_re, CANCER_RE, PROX_WINDOW_CHARS):
        prox_ok = True
    elif core_hit and sec_re and sec_hit and has_site_cancer_proximity(text, sec_re, CANCER_RE, PROX_WINDOW_CHARS):
        prox_ok = True

    # 7) Strong "other primary site" dominance check (only triggers when label is weak)
    # If doc clearly about another cancer site AND our site signal isn't strong -> drop.
    if OTHER_PRIMARY_SITE_RE.search(text) and site_hits <= 2:
        return False, "dominant_other_site"

    # 8) Label-specific strong signals (acts like a "pass" even if proximity missing)
    if label == "Thyroid_Cancer":
        strong = bool(THYROID_STRONG_RE.search(text))
        # keep if strong thyroid cancer OR thyroid present + oncology context
        if not (strong or (core_hit and ONCO_CONTEXT_RE.search(text))):
            return False, "weak_thyroid_context"
    elif label == "Colon_Cancer":
        strong = bool(COLON_STRONG_RE.search(text))
        if not (strong or (core_hit and ONCO_CONTEXT_RE.search(text))):
            return False, "weak_colon_context"
    elif label == "Lung_Cancer":
        strong = bool(LUNG_STRONG_RE.search(text))
        if not (strong or (core_hit and ONCO_CONTEXT_RE.search(text))):
            return False, "weak_lung_context"

    # 9) If proximity is missing and onco context is missing, drop
    # (filters diabetes/iodine, QoL norms, general methods papers)
    if not prox_ok and not ONCO_CONTEXT_RE.search(text):
        return False, "weak_onco_context_no_proximity"

    return True, "ok"


def main():
    df = pd.read_csv(IN_PATH)
    for col in (LABEL_COL, TEXT_COL, PAPER_ID_COL):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}'. Found: {list(df.columns)}")

    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()
    df[TEXT_COL]  = df[TEXT_COL].fillna("").astype(str)

    is_highconf = []
    reasons = []
    for lbl, txt in zip(df[LABEL_COL].tolist(), df[TEXT_COL].tolist()):
        ok, reason = compute_is_highconf(lbl, txt)
        is_highconf.append(ok)
        reasons.append(reason)

    df["is_highconf"] = is_highconf
    df["conf_reason"] = reasons

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    kept = df[df["is_highconf"]].copy()
    kept.to_csv(OUT_PATH, index=False, encoding="utf-8")

    report = {}
    for ct, sub in df.groupby(LABEL_COL):
        report[ct] = {
            "total": int(len(sub)),
            "kept_highconf": int(sub["is_highconf"].sum()),
            "dropped_lowconf": int((~sub["is_highconf"]).sum()),
            "kept_pct": float(100.0 * sub["is_highconf"].mean()),
            "top_drop_reasons": sub.loc[~sub["is_highconf"], "conf_reason"]
                .value_counts().head(10).to_dict(),
        }

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    misses = []
    for ct, sub in df.groupby(LABEL_COL):
        m = sub[~sub["is_highconf"]].copy()
        if m.empty:
            continue
        m["snippet_500"] = m[TEXT_COL].str.slice(0, 500)
        misses.append(m.head(MAX_MISSES_PER_LABEL)[[LABEL_COL, PAPER_ID_COL, "conf_reason", "snippet_500"]])

    if misses:
        pd.concat(misses, ignore_index=True).to_csv(MISSES, index=False, encoding="utf-8")

    print(f"✅ Saved misses sample: {MISSES}")
    print(f"✅ Saved strict high-confidence dataset: {OUT_PATH}")
    print(f"✅ Saved report: {REPORT}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()