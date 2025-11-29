"""
Small helpers for preprocessing dataset JSONL to a canonical form used by eval scripts.
"""

import json
from pathlib import Path

def normalize_jsonl(src_path, dst_path, fields=("id","question","context","answer")):
    out = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            normalized = {k: ex.get(k, None) for k in fields}
            out.append(normalized)
    with open(dst_path, "w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preprocess.py input.jsonl output.jsonl")
    else:
        normalize_jsonl(sys.argv[1], sys.argv[2])
