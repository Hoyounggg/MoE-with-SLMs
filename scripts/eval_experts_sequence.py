import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.eval.data_utils import build_prompt, get_example_id, load_domain_dataset
from scripts.eval.evaluate import evaluate_domain

EXPERT_MODEL_PATHS = {
    "biomedical": os.path.join("experts", "biomedical", "MediPhi-Instruct"),
    "legal": os.path.join("experts", "legal", "Saul-7B-Instruct-v1"),
    "code": os.path.join("experts", "coding", "Qwen2.5-Coder-3B"),
}


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def load_expert(domain: str):
    model_path = EXPERT_MODEL_PATHS.get(domain)
    if model_path is None:
        raise ValueError(f"Unknown domain: {domain}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, model


def unload_expert(model, tokenizer):
    try:
        del model
        del tokenizer
    except Exception:
        pass
    torch.cuda.empty_cache()


def evaluate_expert_domain(domain: str, results_dir: str, max_new_tokens: int):
    dataset = load_domain_dataset(domain)
    tokenizer, model = load_expert(domain)
    preds, times = {}, []
    for ex in dataset:
        ex_id = get_example_id(domain, ex)
        prompt = build_prompt(domain, ex)
        start = time.time()
        preds[ex_id] = generate_response(model, tokenizer, prompt, max_new_tokens)
        times.append(time.time() - start)

    unload_expert(model, tokenizer)

    pred_path = os.path.join(results_dir, f"expert_preds_{domain}.json")
    with open(pred_path, "w") as f:
        json.dump(preds, f, indent=2)

    metrics = evaluate_domain(domain, preds, dataset)
    metrics["avg_latency"] = sum(times) / len(times) if times else 0.0
    return metrics


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    metrics = {}
    for domain in ["biomedical", "legal", "code"]:
        print(f"Evaluating expert model for domain: {domain}")
        metrics[domain] = evaluate_expert_domain(domain, results_dir, max_new_tokens=256)

    metrics_path = os.path.join(results_dir, "experts_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved expert metrics to {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    main()
