# infer_classifier.py
# --------------------------------------
# Purpose: Run inference using the 3-class DistilBERT classifier.
# Output: "biomedical", "legal", or "code".
# --------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the explicit mapping used during training
LABEL2ID = {"biomedical": 0, "legal": 1, "code": 2}
# Use ID2LABEL for easy conversion from predicted ID back to string
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

class DomainClassifier:
    """A wrapper class for running inference on the fine-tuned DistilBERT domain classifier."""
    
    def __init__(self, model_path: str = "./classifier/model"):
        # Determine the device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        ).to(self.device) # Move model to the selected device
        
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()

    def predict(self, text: str) -> str:
        """
        Predict the domain of the input query.
        
        Args:
            text: The input natural language query (e.g., "How do I fix a Python bug?").
            
        Returns:
            The predicted domain label ("biomedical", "legal", or "code").
        """
        
        # 1. Tokenize the input text
        # Set return_tensors="pt" for PyTorch tensors
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128, # Match max_length used during training
            padding=True
        ).to(self.device) # Move input tensors to the selected device

        # 2. Run inference without calculating gradients
        with torch.no_grad():
            # Get logits from the model
            logits = self.model(**inputs).logits
            
        # 3. Determine the predicted class ID
        # argmax finds the index (ID) of the highest logit
        pred_id = torch.argmax(logits, dim=-1).item()
        
        # 4. Convert ID back to the domain label string
        return ID2LABEL[pred_id]

if __name__ == "__main__":
    # Create an instance of the classifier
    clf = DomainClassifier()

    # Example queries for testing
    queries = [
        "How do I fix a Python bug?",
        "What is the best treatment for hyperkalemia?",
        "Can I be sued for intellectual property infringement?",
        "Write a function to reverse a string in Python."
    ]

    print("\n--- Running Inference Examples ---")
    for query in queries:
        prediction = clf.predict(query)
        print(f"Query: '{query}'")
        print(f"Prediction: {prediction}\n")