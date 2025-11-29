"""
Collect queries from data/<domain> and write classifier dataset CSV: text,label
"""
import csv
from pathlib import Path
import json

DATA_ROOT = Path("data")
OUT = Path("classifier/dataset.csv")

domains = ["biomedical","legal","code"]

rows = []
for d in domains:
    p = DATA_ROOT / d
    if not p.exists():
        print(f"Warning: {p} not found, skipping")
        continue
    for fn in p.glob("*.jsonl"):
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                q = ex.get("question","")
                if not q:
                    continue
                rows.append((q, d))

OUT.parent.mkdir(exist_ok=True, parents=True)
with open(OUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["text","label"])
    for r in rows:
        writer.writerow(r)
print("Wrote classifier dataset:", OUT, "n=", len(rows))
