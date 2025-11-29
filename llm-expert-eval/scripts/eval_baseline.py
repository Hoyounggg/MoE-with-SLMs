"""
Run baseline 7B model inference on dataset and compute simple metrics.
Assumes datasets are in data/<domain>/*.jsonl
"""

import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import re
import numpy as np
import argparse

def normalize_text(s):
    if s is None:
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def compute_f1_em(pred, gold):
    p = normalize_text(pred)
    g = normalize_text(gold)
    em = 1 if p == g and p != "" else 0
    p_tokens = p.split()
    g_tokens = g.split()
    common = set(p_tokens) & set(g_tokens)
    if len(common) == 0:
        f1 = 0.0
    else:
        prec = len(common) / (len(p_tokens) + 1e-12)
        rec = len(common) / (len(g_tokens) + 1e-12)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return f1, em

def run_inference(model_id, dataset_path, out_path, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    results = []
    for ex in tqdm(load_jsonl(dataset_path)):
        q = ex.get("question", "")
        prompt = q
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256)
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        results.append({"id": ex.get("id",""), "question": q, "pred": pred, "gold": ex.get("answer","")})
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results

def eval_results(results):
    f1s = []
    ems = []
    for r in results:
        f1, em = compute_f1_em(r["pred"], r["gold"])
        f1s.append(f1); ems.append(em)
    return {"f1_mean": float(np.mean(f1s)), "em": float(np.mean(ems)), "n": len(results)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="baseline model id or local path")
    parser.add_argument("--dataset", required=True, help="path to dataset jsonl")
    parser.add_argument("--out", required=True, help="output jsonl path")
    args = parser.parse_args()

    res = run_inference(args.model, args.dataset, args.out)
    metrics = eval_results(res)
    print("Metrics:", metrics)
