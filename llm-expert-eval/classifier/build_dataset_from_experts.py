# classifier/build_dataset_from_experts.py
# ----------------------------------------------------------------------
# Converts each domain dataset into:
# 1) Expert evaluation format (id, question, context, gold_answer)
# 2) Classifier dataset: (question, label) with automatic train/test split
# ----------------------------------------------------------------------

import os
import json
import random # For random sampling
from datasets import load_from_disk
from sklearn.model_selection import train_test_split

DATA_ROOT = "dataset"
SAVE_ROOT = "dataset/classifier"

DOMAINS = ["biomedical", "legal", "code"]
# Set the maximum number of samples per domain to ensure class balance
# MBPP (Code) is the smallest (~974), so we aim for 1000 per domain.
MAX_SAMPLES_PER_DOMAIN = 1000


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ----------------------------------------------------------------------
# Biomedical: PubMedQA
# ----------------------------------------------------------------------
def process_pubmedqa():
    ds = load_from_disk(os.path.join(DATA_ROOT, "pubmedqa"))

    items = []
    # PubMedQA (pqa_labeled) has 1000 examples, so we just take all of them.
    for ex in ds["train"]:
        q = ex["question"][:-1]
        items.append({
            "id": ex["pubid"],
            "question": q,
            "context": None,
            "answer": ex["long_answer"]
        })
    
    # Apply truncation just in case the size changes (should be 1000)
    return items[:MAX_SAMPLES_PER_DOMAIN]


# ----------------------------------------------------------------------
# Legal: Law-StackExchange
# ----------------------------------------------------------------------
def process_law_stackexchange():
    ds = load_from_disk(os.path.join(DATA_ROOT, "law_stackexchange"))

    all_legal_items = []
    
    # 1. Collect all legal items
    for ex in ds["train"]:
        q = ex["question_title"][:-1].replace("&quot;", "").replace("&#39;", "") # + "\n" + ex.get("question_body", "") Remove main contents for classifier.
        best_answer = None

        # Find the top answer if available
        if isinstance(ex["answers"], list) and len(ex["answers"]) > 0:
            # Sort by score if available
            sorted_ans = sorted(ex["answers"], key=lambda a: a.get("score", 0), reverse=True)
            best_answer = sorted_ans[0]["body"]

        all_legal_items.append({
            "id": ex["question_id"],
            "question": q,
            "context": None,
            "answer": best_answer
        })

    # 2. Randomly DOWN-SAMPLE the large legal dataset to MAX_SAMPLES_PER_DOMAIN
    print(f"Original Legal size: {len(all_legal_items)}. Downsampling to {MAX_SAMPLES_PER_DOMAIN}")
    random.seed(42) # Ensure reproducible sampling
    
    # Sample only MAX_SAMPLES_PER_DOMAIN items
    # If the list is smaller than the max, sample will return the whole list.
    sampled_items = random.sample(all_legal_items, min(len(all_legal_items), MAX_SAMPLES_PER_DOMAIN))
    
    return sampled_items


# ----------------------------------------------------------------------
# Code: MBPP
# ----------------------------------------------------------------------
def process_mbpp():
    ds = load_from_disk(os.path.join(DATA_ROOT, "mbpp"))

    # Use all splits (train, test, validation, prompt) as discussed for a larger 'code' set
    items = []
    for split_name in ds.keys():
        print(f"Processing split: {split_name}")
        
        # Iterate over all examples in the current split
        for ex in ds[split_name]:
            q = ex["text"][:-1]
            q = ' '.join(q.split()[4:]).capitalize()
            items.append({
                "id": ex["task_id"],
                "question": q,    # Natural language prompt
                "context": None,           # MBPP typically doesn't use context
                "answer": ex["code"]       # Ground truth code
            })

    # MBPP total is ~974, which is less than MAX_SAMPLES_PER_DOMAIN (1000).
    return items


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def build_classifier_dataset(all_items):
    """Extract classifier dataset = (question, label) and perform stratified split."""
    X = []
    for domain, items in all_items.items():
        for ex in items:
            X.append({
                "question": ex["question"],
                "label": domain
            })

    # Automatic stratified split (80% train, 20% test)
    labels = [x["label"] for x in X]
    # Use random_state=42 for reproducibility and stratify to maintain class balance
    train, test = train_test_split(X, test_size=0.2, random_state=42, stratify=labels)

    ensure_dir(SAVE_ROOT)
    save_json(os.path.join(SAVE_ROOT, "classifier_train.json"), train)
    save_json(os.path.join(SAVE_ROOT, "classifier_test.json"), test)

    print(f"\n[Classifier] Total data size: {len(X)}")
    print(f"[Classifier] Train size: {len(train)} (80%), Test size: {len(test)} (20%)")


if __name__ == "__main__":
    print("[START] Building expert evaluation & classifier datasets...")

    biomedical = process_pubmedqa()
    legal = process_law_stackexchange()
    code = process_mbpp()
    
    # Report final sizes after balancing
    print(f"Final domain sizes:")
    print(f"  Biomedical: {len(biomedical)}")
    print(f"  Legal:      {len(legal)}")
    print(f"  Code:       {len(code)}")

    # Save expert evaluation sets
    EXPERT_SAVE_DIR = "dataset/expert_eval"
    ensure_dir(EXPERT_SAVE_DIR)

    save_json(os.path.join(EXPERT_SAVE_DIR, "biomedical.json"), biomedical)
    save_json(os.path.join(EXPERT_SAVE_DIR, "legal.json"), legal)
    save_json(os.path.join(EXPERT_SAVE_DIR, "code.json"), code)

    all_data = {
        "biomedical": biomedical,
        "legal": legal,
        "code": code
    }

    # Build classifier dataset
    build_classifier_dataset(all_data)

    print("[DONE] All datasets processed.")