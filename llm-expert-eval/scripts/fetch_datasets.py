# scripts/fetch_datasets.py
# ---------------------------------------------------------
# Downloads all datasets used for evaluating the domain experts.
# Biomedical → PubMedQA
# Legal → Law-StackExchange
# Code → MBPP
# ---------------------------------------------------------

import os
from datasets import load_dataset

DATA_ROOT = "dataset"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def fetch_pubmedqa():
    """Download the PubMedQA dataset (biomedical domain)."""
    print("[Biomedical] Downloading PubMedQA...")
    ds = load_dataset("pubmed_qa", "pqa_labeled")
    save_path = os.path.join(DATA_ROOT, "pubmedqa")
    ensure_dir(save_path)
    ds.save_to_disk(save_path)
    print(f"[Biomedical] Saved to {save_path}")


def fetch_law_stackexchange():
    """Download the Law-StackExchange dataset (legal domain)."""
    print("[Legal] Downloading Law-StackExchange...")
    ds = load_dataset("ymoslem/Law-StackExchange")
    save_path = os.path.join(DATA_ROOT, "law_stackexchange")
    ensure_dir(save_path)
    ds.save_to_disk(save_path)
    print(f"[Legal] Saved to {save_path}")


def fetch_mbpp():
    """Download the MBPP dataset (code domain)."""
    print("[Coding] Downloading MBPP...")
    ds = load_dataset("mbpp")
    save_path = os.path.join(DATA_ROOT, "mbpp")
    ensure_dir(save_path)
    ds.save_to_disk(save_path)
    print(f"[Coding] Saved to {save_path}")


if __name__ == "__main__":
    ensure_dir(DATA_ROOT)
    fetch_pubmedqa()
    fetch_law_stackexchange()
    fetch_mbpp()
    print("[DONE] All datasets downloaded.")
