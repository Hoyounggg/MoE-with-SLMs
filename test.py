# 1_build_dataset.py
import json
import random
from datasets import load_dataset
from tqdm import tqdm

# Configuration
OUTPUT_FILE = "routing_dataset.jsonl"
SAMPLES_PER_DOMAIN = 500

def create_entry(raw_query, refined_query, domain):
    # Determine the target output JSON
    target_json = json.dumps({
        "refined_query": refined_query,
        "domain": domain
    })
    
    return {
        "instruction": "Analyze the query, refine it, and route to Biomedical, Legal, or Programming. Output JSON.",
        "input": raw_query,
        "output": target_json
    }

def build_data():
    data = []
    print("Building Dataset...")

    # --- 1. Biomedical Data (PubMedQA) ---
    print("Loading Biomedical data...")
    ds_bio = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)
    # PubMedQA has questions that act as 'refined' queries. We treat them as raw for now.
    for row in ds_bio.select(range(SAMPLES_PER_DOMAIN)):
        # Simulate a "lazy" user query by lowercasing and removing punctuation
        clean_q = row['question']
        lazy_q = clean_q.lower().replace("?", "")
        data.append(create_entry(lazy_q, clean_q, "Biomedical"))

    # --- 2. Legal Data (LegalBench / Pile of Law subset) ---
    # Using a subset of StackExchange Law for variety
    print("Loading Legal data...")
    ds_legal = load_dataset("ymoslem/Law-StackExchange", split="train") 
    for row in ds_legal.select(range(SAMPLES_PER_DOMAIN)):
        clean_q = row['question_title']
        lazy_q = clean_q.lower()
        data.append(create_entry(lazy_q, clean_q, "Legal"))

    # --- 3. Programming Data (MBPP) ---
    print("Loading Programming data...")
    ds_code = load_dataset("mbpp", split="test")
    for row in ds_code.select(range(SAMPLES_PER_DOMAIN)):
        clean_q = row['text'] # "Write a python function to..."
        lazy_q = clean_q.replace("Write a python function to ", "code for ").lower()
        data.append(create_entry(lazy_q, clean_q, "Programming"))

    # Shuffle and Save
    random.shuffle(data)
    with open(OUTPUT_FILE, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Saved {len(data)} training examples to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_data()