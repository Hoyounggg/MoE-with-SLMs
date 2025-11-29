"""
Simulate sequential expert evaluation: for each domain in order, feed all test queries,
measure latency, ensure model load/unload as domain changes, and collect metrics.
"""

import time
import json
from pathlib import Path
from router.model_router import ExpertRouter
import argparse
import numpy as np

DOMAIN_ORDER = ["biomedical", "legal", "code"]

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def compute_metrics(results):
    import re
    f1s = []
    ems = []
    for r in results:
        p = (r.get("pred") or "").strip().lower()
        g = (r.get("gold") or "").strip().lower()
        p = re.sub(r"\s+", " ", p)
        g = re.sub(r"\s+", " ", g)
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
        f1s.append(f1); ems.append(em)
    return {"f1_mean": float(np.mean(f1s)), "em": float(np.mean(ems)), "n": len(results)}

def run_sequence(data_root="data", domains=DOMAIN_ORDER):
    router = ExpertRouter()
    all_results = []
    total_time = 0.0
    for domain in domains:
        dataset_path = Path(data_root) / domain
        test_files = list(dataset_path.glob("*.jsonl"))
        if not test_files:
            print(f"No dataset found for {domain} in {dataset_path}")
            continue
        items = load_jsonl(test_files[0])
        # load expert
        t0 = time.time()
        router.load_expert(domain)
        load_time = time.time() - t0
        print(f"Loaded {domain} expert in {load_time:.2f}s")
        # run each query
        latencies = []
        results = []
        for ex in items:
            q = ex.get("question", "")
            start = time.time()
            pred = router.generate(q, max_tokens=256)
            latency = time.time() - start
            latencies.append(latency)
            results.append({
                "id": ex.get("id",""), "domain": domain, "question": q,
                "pred": pred, "gold": ex.get("answer",""), "latency": latency
            })
        # unload model to simulate switching
        router.unload_current_model()
        all_results.extend(results)
        total_time += sum(latencies) + load_time
    metrics = compute_metrics(all_results)
    return all_results, metrics, total_time

if __name__ == "__main__":
    res, metrics, tot_time = run_sequence()
    print("Sequence metrics:", metrics, "total_time:", tot_time)
    Path("runs").mkdir(exist_ok=True)
    with open("runs/expert_sequence_results.jsonl", "w", encoding="utf-8") as f:
        for r in res:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
