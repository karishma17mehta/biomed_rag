import pandas as pd
import re
from pathlib import Path

IN_PATH  = Path("outputs/phase2_filtered_papers.csv")
OUT_PATH = Path("outputs/phase2_filtered_papers_highconf.csv")
OUT_REPORT = Path("outputs/label_conf_report.json")

df = pd.read_csv(IN_PATH)
df["clean_text"] = df["clean_text"].fillna("").astype(str)

KEYWORDS = {

    "Lung_Cancer": [
        # Core
        "lung", "pulmonary", "respiratory", "bronch", "bronchus", "alveolar",

        # Disease names
        "non small cell lung cancer", "non-small cell lung cancer", "nsclc",
        "small cell lung cancer", "sclc",
        "lung carcinoma", "lung tumor", "lung tumour",
        "bronchogenic carcinoma",

        # Histology
        "adenocarcinoma", "squamous cell carcinoma", "large cell carcinoma",
        "pleural", "pleura", "mesothelioma",

        # Mutations / biomarkers
        "egfr", "alk", "ros1", "kras", "braf", "pd-l1", "her2",

        # Staging / screening
        "tnm stage", "stage iv lung", "stage iii lung",
        "low-dose ct", "ldct screening",

        # Procedures
        "lobectomy", "pneumonectomy", "thoracic surgery",
        "bronchoscopy"
    ],


    "Colon_Cancer": [
        # Core
        "colon", "colorectal", "rectal", "colonic",
        "large intestine", "bowel",

        # Disease names
        "colorectal cancer", "crc",
        "colon carcinoma", "rectal carcinoma",
        "sigmoid colon", "cecum", "caecum",

        # Histology
        "adenoma", "adenomatous polyp", "polyp",
        "mucinous carcinoma", "signet ring",

        # Genetics / biomarkers
        "apc mutation", "kras mutation", "braf mutation",
        "microsatellite instability", "msi",
        "lynch syndrome", "mlh1", "msh2",

        # Inflammatory conditions (strong risk factors)
        "ulcerative colitis", "crohn", "ibd",

        # Screening / procedures
        "colonoscopy", "fecal occult blood", "fit test",
        "hemicolectomy", "colectomy"
    ],


    "Thyroid_Cancer": [
        # Core
        "thyroid", "thyroid gland",

        # Disease names
        "thyroid cancer", "thyroid carcinoma",
        "papillary thyroid carcinoma", "ptc",
        "follicular thyroid carcinoma", "ftc",
        "medullary thyroid carcinoma", "mtc",
        "anaplastic thyroid carcinoma", "atc",
        "differentiated thyroid cancer",

        # Histology
        "papillary", "follicular", "medullary", "anaplastic",

        # Biomarkers / mutations
        "braf v600e", "ret mutation", "ras mutation",
        "thyroglobulin", "calcitonin", "tsh receptor",

        # Clinical terms
        "thyroid nodule", "goiter", "goitre",
        "thyroidectomy", "radioiodine", "iodine-131",
        "neck dissection"
    ],
}


def compile_pattern(words):
    escaped = []
    for w in words:
        if " " in w:
            parts = [re.escape(p) for p in w.split()]
            escaped.append(r"\b" + r"\s+".join(parts) + r"\b")
        else:
            escaped.append(r"\b" + re.escape(w) + r"\b")
    return re.compile(r"(?:%s)" % "|".join(escaped), flags=re.IGNORECASE)

PAT = {k: compile_pattern(v) for k, v in KEYWORDS.items()}

# Label-hit flag
df["label_kw_hit"] = df.apply(
    lambda r: bool(PAT.get(r["cancer_type"], re.compile("$^")).search(r["clean_text"])),
    axis=1
)

# Strict high-confidence subset
high = df[df["label_kw_hit"]].copy()

# Report
report = {}
for ct in sorted(df["cancer_type"].unique()):
    sub = df[df["cancer_type"] == ct]
    report[ct] = {
        "total": int(len(sub)),
        "kept_highconf": int(sub["label_kw_hit"].sum()),
        "dropped_lowconf": int((~sub["label_kw_hit"]).sum()),
        "kept_pct": float(sub["label_kw_hit"].mean() * 100.0) if len(sub) else 0.0
    }

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
high.to_csv(OUT_PATH, index=False)

import json
with OUT_REPORT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("✅ Saved:", OUT_PATH)
print("✅ Report:", OUT_REPORT)
print(json.dumps(report, indent=2))
