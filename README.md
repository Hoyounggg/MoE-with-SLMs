# llm-expert-eval

Evaluate domain-specific expert LLMs (Biomedical, Legal, Code) vs a single baseline model (≤8B). This repo includes dataset fetching scripts, evaluation code for a baseline model and a sequential expert router (load/unload models per domain), plotting utilities, and a small DistilBERT-based 3-class domain classifier training pipeline.

**Summary**
- Domains: `biomedical`, `legal`, `code`
- Experts (pre-downloaded, not trained here):
  - Biomedical: `MediPhi-3.5-mini` (Phi-3.5 family, 3.8B)
  - Legal: `Legal-Llama-3.2-3B` (Llama-3.2 family, 3.21B)
  - Code: `Qwen2.5-Coder-3B` (Qwen2.5 family, 3.09B)
- Baseline: example `Mistral-7B-Instruct` (or other ≤8B model)

**Goal**
1. Evaluate a single baseline model on datasets for each domain.
2. Evaluate the sequential expert router that loads the appropriate expert for each domain (simulate user queries fed domain-by-domain), measure metrics and latency.
3. Build a classifier dataset from the queries and fine-tune a DistilBERT 3-class classifier for routing.

---
## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Fetch / prepare datasets:

```bash
python scripts/fetch_datasets.py
```

Note: some datasets require registration or manual download. The script will print instructions when manual action is needed.

3. Run baseline evaluation (example):

```bash
python scripts/eval_baseline.py --model mistralai/Mistral-7B-Instruct-v0.3 --dataset data/biomedical/bioasq_test.jsonl --out runs/baseline_bioasq.jsonl
```

4. Run sequential expert evaluation:

```bash
python scripts/eval_experts_sequence.py
```

5. Build classifier dataset and train classifier:

```bash
python classifier/build_dataset_from_experts.py
python classifier/train_classifier.py
```

6. Plot comparison (after running baseline & expert experiments and collecting metrics):

```bash
python scripts/plot_results.py baseline_metrics.json expert_metrics.json
```

---
## Notes / Requirements
- You must pre-download or be able to access the three expert models locally under the `experts/` folder (paths used in `router/model_router.py`). This repository does not ship those models.
- Some datasets (BioASQ, LEGALBENCH) require registration or manual download — see dataset scripts for pointers.
- For multi-GPU inference you may use `device_map="auto"` (HuggingFace) or DeepSpeed inference. See comments in `router/model_router.py`.

---
## Evaluation metrics
- Biomedical & Legal: token-level F1, Exact Match (EM)
- Code: pass@k (if unit-tests available) or exact-match for short snippets
- Also measure: model load time, per-query latency, and VRAM usage (via `nvidia-smi` logging)

---
## Contact
If you need DeepSpeed-optimized router code or Dockerfile / conda environment files, open an issue or request in the conversation.
