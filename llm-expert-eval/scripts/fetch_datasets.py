"""
Download / prepare datasets for three domains and save them as JSONL:
Each line: {"id": str, "question": str, "context": str | null, "answer": str}
"""

import os
import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data")

def save_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def fetch_bioasq():
    out = OUT_DIR / "biomedical"
    out.mkdir(parents=True, exist_ok=True)

    # BioASQ often requires registration and manual fetching. Try HF dataset then fallback.
    try:
        ds = load_dataset("bioasq")
        items = []
        split = next(iter(ds.keys()))
        for ex in ds[split]:
            items.append({
                "id": ex.get("id", ""),
                "question": ex.get("question", ""),
                "context": ex.get("snippets", None),
                "answer": ex.get("answers", None)
            })
        save_jsonl(items, out / "bioasq_test.jsonl")
        print("Saved BioASQ ->", out / "bioasq_test.jsonl")
    except Exception as e:
        print("BioASQ may require manual download. See: https://bioasq.org/ for instructions.")
        print("Error:", e)

def fetch_pubmedqa():
    out = OUT_DIR / "biomedical"
    out.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset("pubmed_qa", "pqa_labeled")
        items = []
        for ex in ds["test"]:
            items.append({
                "id": ex.get("pmid", ""),
                "question": ex.get("question", ""),
                "context": ex.get("abstract", ""),
                "answer": ex.get("label", "")
            })
        save_jsonl(items, out / "pubmedqa_test.jsonl")
        print("Saved PubMedQA ->", out / "pubmedqa_test.jsonl")
    except Exception as e:
        print("PubMedQA not available via HF in this environment. See https://github.com/ for manual instructions.")
        print("Error:", e)

def fetch_legal():
    out = OUT_DIR / "legal"
    out.mkdir(parents=True, exist_ok=True)

    # LEGALBENCH / JEC-QA often require manual fetch. Attempt to load JEC-QA if available.
    try:
        ds = load_dataset("jec_qa", split="test")
        items = [{
            "id": ex.get("id", ""),
            "question": ex.get("question", ""),
            "context": ex.get("context", ""),
            "answer": ex.get("answer", "")
        } for ex in ds]
        save_jsonl(items, out / "jecqa_test.jsonl")
        print("Saved JEC-QA ->", out / "jecqa_test.jsonl")
    except Exception as e:
        print("LEGALBENCH/JEC-QA may require manual download. See LEGALBENCH paper and supplemental materials for links.")
        print("Error:", e)

def fetch_code():
    out = OUT_DIR / "code"
    out.mkdir(parents=True, exist_ok=True)
    # Try MBPP, HumanEval often require special handling. We'll try MBPP from HF.
    try:
        ds = load_dataset("mbpp", "main")
        items = []
        # use test split if present; HF's 'mbpp' might have an evaluation split
        split_name = "test" if "test" in ds else "validation"
        for ex in ds[split_name]:
            items.append({
                "id": ex.get("baseline_id", ""),
                "question": ex.get("text", ""),
                "context": None,
                "answer": ex.get("code", "")
            })
        save_jsonl(items, out / "mbpp_test.jsonl")
        print("Saved MBPP ->", out / "mbpp_test.jsonl")
    except Exception as e:
        print("MBPP not available via HF in this environment. For HumanEval see OpenAI or GitHub mirrors.")
        print("Error:", e)

if __name__ == "__main__":
    fetch_bioasq()
    fetch_pubmedqa()
    fetch_legal()
    fetch_code()
