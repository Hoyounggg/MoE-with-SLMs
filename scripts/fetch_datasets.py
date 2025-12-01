
from datasets import load_dataset
import os

ROOT = "dataset"
os.makedirs(ROOT, exist_ok=True)

def save(name, path):
    ds = load_dataset(name) if isinstance(name, str) else load_dataset(*name)
    ds.save_to_disk(path)

print("Downloading PubMedQA")
save(("pubmed_qa", "pqa_labeled"), f"{ROOT}/pubmedqa")

print("Downloading Law-StackExchange")
save("ymoslem/Law-StackExchange", f"{ROOT}/law_stackexchange")

print("Downloading MBPP")
save("mbpp", f"{ROOT}/mbpp")

print("Done")
