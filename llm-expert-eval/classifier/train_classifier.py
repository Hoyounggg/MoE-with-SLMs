"""
Train a DistilBERT 3-class classifier using the classifier/dataset.csv.
Output saved to classifier/model/
"""

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("classifier/dataset.csv")
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
train_ds = Dataset.from_pandas(train_df)
eval_ds = Dataset.from_pandas(eval_df)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(preprocess, batched=True)
eval_ds = eval_ds.map(preprocess, batched=True)

label2id = {"biomedical":0, "legal":1, "code":2}
def encode(batch):
    batch["labels"] = label2id[batch["label"]]
    return batch

train_ds = train_ds.map(encode)
eval_ds = eval_ds.map(encode)

train_ds = train_ds.remove_columns(["text","label"])
eval_ds = eval_ds.remove_columns(["text","label"])

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir="./classifier/model",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-5,
    load_best_model_at_end=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds)
trainer.train()
trainer.save_model("./classifier/model")
tokenizer.save_pretrained("./classifier/model")
