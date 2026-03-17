import json, re, hashlib
from pathlib import Path

META_IN = Path("outputs/index_openai/meta.jsonl")
META_OUT = Path("outputs/index_openai/meta_hashed.jsonl")

def normalize_for_hash(text: str) -> str:
    t = text or ""
    # remove leading bracket header: [Cancer: ...]
    t = re.sub(r"^\[Cancer:.*?\]\s*", "", t, flags=re.DOTALL)
    # remove "Key idea:" wrapper if present
    t = re.sub(r"\bKey idea:\s*", "", t, flags=re.IGNORECASE)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def main():
    n = 0
    with META_IN.open("r", encoding="utf-8") as fin, META_OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            norm = normalize_for_hash(obj.get("text",""))
            h = hashlib.md5(norm.encode("utf-8")).hexdigest()
            obj["content_hash"] = h
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print("Wrote:", META_OUT, "rows:", n)

if __name__ == "__main__":
    main()

