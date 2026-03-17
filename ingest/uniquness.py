import pandas as pd
import hashlib
import re
from pathlib import Path

# 🔴 CHANGE THIS to your real original CSV path
CSV_PATH = Path("outputs/clean_papers.csv")   # or phase1_clean_papers.csv

TEXT_COL = "clean_text"   # change if different
ID_COL = "paper_id"       # change if different

def fingerprint(text: str) -> str:
    t = (text or "").lower().strip()
    t = re.sub(r"\s+", " ", t)
    return hashlib.md5(t.encode("utf-8")).hexdigest()

df = pd.read_csv(CSV_PATH)

print("Total rows:", len(df))

df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

# Fingerprint the full document text
df["fp"] = df[TEXT_COL].map(fingerprint)

unique_texts = df["fp"].nunique()

print("Unique document texts:", unique_texts)
print("Duplicate ratio:", 1 - (unique_texts / max(len(df), 1)))

# Show most duplicated texts
print("\nTop 10 most duplicated document texts:")
print(df["fp"].value_counts().head(10))

# Show example paper_ids sharing the worst duplicate
most_common_fp = df["fp"].value_counts().index[0]
dupes = df[df["fp"] == most_common_fp]

print("\nExample duplicated paper_ids:")
print(dupes[ID_COL].head(20).tolist())