# infer_classifier.py
# --------------------------------------
# Purpose: Run inference using the 3-class DistilBERT classifier.
# Output: "biomedical", "legal", or "code".
# --------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = ["biomedical", "legal", "code"]

class DomainClassifier:
    def __init__(self, model_path="./classifier/model"):
        # Load classifier
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text: str) -> str:
        """Predict the domain of the input query."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        return LABELS[pred_id]

if __name__ == "__main__":
    clf = DomainClassifier()
    print(clf.predict("How do I fix a Python bug?"))
