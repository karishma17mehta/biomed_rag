import json
import re
from pathlib import Path

IN_PATH  = Path("outputs/index_openai/meta.jsonl")
OUT_PATH = Path("outputs/index_openai/meta_clean.jsonl")

# Regex to strip:
# 1) [Cancer: ...]
# 2) "Key idea: ..."
# 3) leading whitespace
META_PREFIX_RE = re.compile(
    r"^\[Cancer:.*?\]\s*\n?\s*Key idea:.*?\n?\s*",
    re.IGNORECASE | re.DOTALL
)

def extract_body(text: str) -> str:
    """
    Remove header and key idea.
    Keep only the true body after the '---' separator.
    """
    if not text:
        return ""

    txt = text.strip()

    # Remove header + key idea block
    txt = META_PREFIX_RE.sub("", txt).strip()

    # Split on --- separator (supports both styles)
    if "\n---\n" in txt:
        parts = txt.split("\n---\n", 1)
        return parts[1].strip() if len(parts) == 2 else parts[0].strip()

    if " --- " in txt:
        parts = txt.split(" --- ", 1)
        return parts[1].strip() if len(parts) == 2 else parts[0].strip()

    return txt.strip()


def main():
    removed = 0
    total = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, \
         OUT_PATH.open("w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            total += 1
            row = json.loads(line)

            original = row.get("text", "")
            cleaned  = extract_body(original)

            if cleaned != original:
                removed += 1

            row["text"] = cleaned
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Processed: {total}")
    print(f"Cleaned:   {removed}")
    print(f"Saved to:  {OUT_PATH}")


if __name__ == "__main__":
    main()