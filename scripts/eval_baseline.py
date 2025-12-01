import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.eval.data_utils import build_prompt, get_example_id, load_domain_dataset
from scripts.eval.evaluate import evaluate_domain

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


def evaluate_baseline_for_domain(domain, model, tokenizer, results_dir: str, max_new_tokens: int):
    dataset = load_domain_dataset(domain)
    preds = {}
    for ex in dataset:
        ex_id = get_example_id(domain, ex)
        prompt = build_prompt(domain, ex)
        preds[ex_id] = generate_response(model, tokenizer, prompt, max_new_tokens)

    pred_path = os.path.join(results_dir, f"baseline_preds_{domain}.json")
    with open(pred_path, "w") as f:
        json.dump(preds, f, indent=2)

    metrics = evaluate_domain(domain, preds, dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single baseline model on all domains.")
    parser.add_argument(
        "--model",
        "-m",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HF model ID or local path for the baseline model (≤8B).",
    )
    parser.add_argument(
        "--results_dir",
        "-o",
        default="results",
        help="Directory to store predictions and metrics.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per example.",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    metrics = {}
    for domain in ["biomedical", "legal", "code"]:
        print(f"Evaluating baseline on domain: {domain}")
        metrics[domain] = evaluate_baseline_for_domain(
            domain, model, tokenizer, args.results_dir, args.max_new_tokens
        )

    metrics_path = os.path.join(args.results_dir, "baseline_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved baseline metrics to {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    main()
