"""
Train a DistilBERT 3-class classifier using the generated JSON files.
Output model saved to classifier/model/
"""

import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --- Configuration ---
MODEL_OUTPUT_DIR = "./classifier/model"
# The directory where build_dataset_from_experts.py saved the JSON files
SAVE_ROOT = "dataset/classifier"
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 3
TEST_RESULT_FILE = os.path.join(MODEL_OUTPUT_DIR, "test_results.json") # Path to save test results

# Define the mapping from domain string to integer ID
LABEL2ID = {"biomedical": 0, "legal": 1, "code": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# --- 1. Data Loading and Splitting ---

def load_classifier_datasets(save_root):
    """Load train and test data from JSON files and create a validation split."""
    
    train_path = os.path.join(save_root, "classifier_train.json")
    test_path = os.path.join(save_root, "classifier_test.json")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Required dataset files not found. Ensure build_dataset_from_experts.py has been run. "
            f"Missing files in: {os.path.abspath(save_root)}"
        )

    # Load data from the files created by the preprocessing script
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
        
    # Convert list of dicts to DataFrame for easier stratified splitting
    df_train = pd.DataFrame(train_data)
    
    # Stratified split to create a validation set: 
    # The train_data (80% of total) is split into ~87.5% final_train and ~12.5% validation.
    # This results in approx. 70% total for training, 10% for validation, and 20% for testing.
    df_train_final, df_val = train_test_split(
        df_train, 
        test_size=0.125, # 0.125 * 0.8 = 0.1 (10% of total data)
        random_state=42, 
        stratify=df_train["label"] 
    )
    
    # Convert DataFrames to Hugging Face Datasets
    # Added preserve_index=False to prevent creation of __index_level_0__ column
    ds = DatasetDict({
        "train": Dataset.from_pandas(df_train_final, preserve_index=False),
        "validation": Dataset.from_pandas(df_val, preserve_index=False),
        "test": Dataset.from_pandas(pd.DataFrame(test_data), preserve_index=False)
    })
    
    print(f"\n[Data Load] Successfully loaded and split data:")
    print(f"  Train Size: {len(ds['train'])}")
    print(f"  Validation Size: {len(ds['validation'])}")
    print(f"  Test Size: {len(ds['test'])}")
    
    return ds

# Load the datasets from the generated JSON files located in 'dataset/classifier'
raw_datasets = load_classifier_datasets(SAVE_ROOT)

# --- 2. Tokenization and Encoding ---

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    """Tokenize the 'question' text."""
    # Max length 128 is suitable for query classification
    return tokenizer(examples["question"], truncation=True, padding="max_length", max_length=128)

def encode_labels(examples):
    """Convert string labels to integer IDs."""
    examples["labels"] = LABEL2ID[examples["label"]]
    return examples

# Apply preprocessing and label encoding
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
encoded_datasets = tokenized_datasets.map(encode_labels)

# Check which columns actually exist in the dataset and remove only those.
cols_to_remove = ["question", "label", "__index_level_0__"]
existing_cols = encoded_datasets["train"].column_names
cols_to_remove = [c for c in cols_to_remove if c in existing_cols]

final_datasets = encoded_datasets.remove_columns(cols_to_remove) 
final_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# --- 3. Model and Trainer Setup ---

# Load model with defined labels
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute accuracy and F1 scores
    accuracy = accuracy_score(labels, predictions)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    
    return {
        "accuracy": accuracy,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro
    }

# [FIXED] Renamed evaluation_strategy to eval_strategy
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    eval_strategy="epoch",  # Updated parameter name for recent transformers versions
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    save_strategy="epoch",
    fp16=True, # Enable mixed precision
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=final_datasets["train"],
    eval_dataset=final_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
)

# --- 4. Training and Saving ---
print("\n[START] Training DistilBERT Classifier...")
trainer.train()

# Save final model and tokenizer
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
print(f"[DONE] Model saved to {MODEL_OUTPUT_DIR}")


# --- 5. Final Test Set Evaluation and Saving Results ---
print("\n[START] Evaluating on Test Set...")
test_results = trainer.evaluate(eval_dataset=final_datasets["test"])

print("\n--- Final Test Set Results ---")
print(test_results)

# Save test results to JSON file
with open(TEST_RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(test_results, f, indent=4)

print(f"[DONE] Test results saved to {TEST_RESULT_FILE}")