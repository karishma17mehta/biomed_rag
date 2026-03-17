import pandas as pd
from pathlib import Path

IN_PATH  = Path("outputs/phase1_clean_papers.csv")
OUT_PATH = Path("outputs/phase2_filtered_papers.csv")

MIN_WORDS = 200

def main():
    df = pd.read_csv(IN_PATH)

    # Ensure clean_text is string
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["n_words"] = df.get("n_words", df["clean_text"].str.split().str.len())

    before = len(df)

    df = df[df["clean_text"].str.strip().str.len() > 0]
    df = df[df["n_words"] >= MIN_WORDS]

    after = len(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print("✅ Filtered papers saved:", OUT_PATH)
    print(f"Before: {before} | After: {after} | Dropped: {before - after}")

if __name__ == "__main__":
    main()