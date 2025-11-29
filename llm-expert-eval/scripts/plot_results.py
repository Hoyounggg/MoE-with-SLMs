"""
Plot comparison: baseline vs experts (bar chart for F1/EM and latency)
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_bar(baseline_metrics, expert_metrics, out="runs/compare.png"):
    domains = ["biomedical", "legal", "code"]
    baseline_f1 = [baseline_metrics[d]["f1_mean"] for d in domains]
    expert_f1 = [expert_metrics[d]["f1_mean"] for d in domains]

    x = np.arange(len(domains))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, baseline_f1, width, label='baseline (7B)')
    ax.bar(x + width/2, expert_f1, width, label='expert-system')
    ax.set_ylabel('F1')
    ax.set_title('Baseline vs Expert-system by domain')
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    print("Saved plot:", out)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python plot_results.py baseline_metrics.json expert_metrics.json")
    else:
        baseline_metrics = json.load(open(sys.argv[1], "r", encoding="utf-8"))
        expert_metrics = json.load(open(sys.argv[2], "r", encoding="utf-8"))
        plot_bar(baseline_metrics, expert_metrics)
