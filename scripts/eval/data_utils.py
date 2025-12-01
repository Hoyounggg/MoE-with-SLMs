import os
from datasets import load_from_disk

# Domain configuration for dataset paths and identifiers
DOMAIN_CONFIG = {
    "biomedical": {
        "path": os.path.join("dataset", "pubmedqa"),
        "split": "train",
        "id_field": "pubid",
    },
    "legal": {
        "path": os.path.join("dataset", "law_stackexchange"),
        "split": "train",
        "id_field": "question_id",
    },
    "code": {
        "path": os.path.join("dataset", "mbpp"),
        "split": "train",
        "id_field": "task_id",
    },
}


def load_domain_dataset(domain: str):
    """Load the train split for the given domain directly from disk."""
    cfg = DOMAIN_CONFIG.get(domain)
    if cfg is None:
        raise ValueError(f"Unknown domain: {domain}")
    ds = load_from_disk(cfg["path"])
    split = cfg.get("split", "train")
    if split not in ds:
        raise ValueError(f"Split '{split}' not found in dataset at {cfg['path']}")
    return ds[split]


def get_example_id(domain: str, example) -> str:
    """Return the identifier for an example, used as key for predictions."""
    cfg = DOMAIN_CONFIG.get(domain)
    if cfg is None:
        raise ValueError(f"Unknown domain: {domain}")
    return str(example[cfg["id_field"]])


def build_prompt(domain: str, example) -> str:
    """Create a domain-specific prompt from a dataset row."""
    if domain == "biomedical":
        context_field = example.get("context") if isinstance(example, dict) else example["context"]
        context_chunks = []
        if isinstance(context_field, dict):
            contexts = context_field.get("contexts") or []
            if contexts:
                context_chunks.append(" ".join(contexts))
        context_text = f"\nContext: {' '.join(context_chunks)}" if context_chunks else ""
        return (
            "You are a biomedical expert. Provide a clear answer and a final decision (yes/no/maybe).\n"
            f"Question: {example['question']}{context_text}\nAnswer:"
        )

    if domain == "legal":
        title = example.get("question_title", "") if isinstance(example, dict) else example["question_title"]
        body = example.get("question_body", "") if isinstance(example, dict) else example["question_body"]
        question = f"{title}\n{body}".strip()
        return (
            "You are a legal expert. Answer the question concisely and provide the best possible legal guidance.\n"
            f"Question:\n{question}\nAnswer:"
        )

    if domain == "code":
        prompt = example["text"]
        return (
            "You are a coding assistant. Write a correct, runnable Python solution that satisfies the tests. "
            "Return only code.\n"
            f"Problem: {prompt}\nSolution:"
        )

    raise ValueError(f"Unknown domain: {domain}")
