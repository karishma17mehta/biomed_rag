import pandas as pd
import re
from pathlib import Path

PATH = Path("outputs/phase2_filtered_papers.csv")
OUT_MISSES = Path("outputs/label_sanity_misses_sample.csv")

df = pd.read_csv(PATH)
df["clean_text"] = df["clean_text"].fillna("").astype(str)

# Broader lexicons (still simple, but much more realistic)
KEYWORDS = {
    "Lung_Cancer": [
        "lung", "pulmonary", "nsclc", "sclc", "adenocarcinoma", "bronch", "bronchus",
        "alveolar", "bronchiolo", "pleura", "pleural", "respiratory"
    ],
    "Colon_Cancer": [
        "colon", "colorectal", "rectal", "crc", "colonic", "sigmoid", "cecum", "caecum",
        "bowel", "large intestine", "adenoma", "polyp", "ibd", "ulcerative colitis", "crohn"
    ],
    "Thyroid_Cancer": [
        "thyroid", "papillary", "follicular", "ptc", "ftc", "differentiated thyroid",
        "thyroidectomy", "thyroglobulin", "tsh", "nodule", "goiter", "goitre",
        "anaplastic", "atc", "medullary", "mtc"
    ],
}

def compile_pattern(words):
    # non-capturing group -> avoids pandas "match groups" warning
    # also allow phrases (e.g. "large intestine") with \s+
    escaped = []
    for w in words:
        if " " in w:
            parts = [re.escape(p) for p in w.split()]
            escaped.append(r"\b" + r"\s+".join(parts) + r"\b")
        else:
            escaped.append(r"\b" + re.escape(w) + r"\b")
    return re.compile(r"(?:%s)" % "|".join(escaped), flags=re.IGNORECASE)

PATTERNS = {k: compile_pattern(v) for k, v in KEYWORDS.items()}

print("\n=== Label Sanity Check (Deep) ===\n")

miss_rows = []

for cancer_type, pattern in PATTERNS.items():
    subset = df[df["cancer_type"] == cancer_type].copy()
    total = len(subset)

    hits = subset["clean_text"].str.contains(pattern, na=False).sum()
    pct = (hits / total * 100) if total else 0.0

    print(f"{cancer_type}")
    print(f"  Total docs: {total}")
    print(f"  Contain expected keywords: {hits} ({pct:.2f}%)")
    print(f"  NO expected keywords: {total - hits} ({100 - pct:.2f}%)")

    # per-keyword hit rates (helps you see which terms matter)
    print("  Top keyword hits:")
    kw_hits = []
    for kw in KEYWORDS[cancer_type]:
        kw_pat = compile_pattern([kw])
        c = subset["clean_text"].str.contains(kw_pat, na=False).sum()
        kw_hits.append((kw, c))
    kw_hits = sorted(kw_hits, key=lambda x: x[1], reverse=True)[:8]
    for kw, c in kw_hits:
        print(f"    {kw:<22} {c} ({(c/total*100 if total else 0):.2f}%)")

    # sample misses (so we can judge if label leakage exists)
    misses = subset[~subset["clean_text"].str.contains(pattern, na=False)]
    sample = misses.sample(n=min(25, len(misses)), random_state=7)

    for _, r in sample.iterrows():
        snippet = r["clean_text"][:500].replace("\n", " ")
        miss_rows.append({
            "cancer_type": cancer_type,
            "paper_id": r.get("paper_id", ""),
            "n_words": r.get("n_words", ""),
            "snippet_500": snippet
        })

    print()

# Save misses for manual review
if miss_rows:
    out = pd.DataFrame(miss_rows)
    OUT_MISSES.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_MISSES, index=False)
    print(f"✅ Saved miss samples to: {OUT_MISSES}")
else:
    print("✅ No misses found; nothing saved.")
                                    