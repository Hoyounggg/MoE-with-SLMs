# model_router.py
# ---------------------------------------------------
# Purpose:
#   - Load and unload domain-specific expert LLMs
#   - Cache currently active model to avoid repeated loads
#   - Automatically unload model when switching domains
#
# Assumptions:
#   - All models are pre-downloaded under experts/<domain>/
#   - Using HuggingFace Transformers
# ---------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Local paths that match download_experts.py targets
DOMAIN_TO_PATH = {
    "biomedical": "./experts/biomedical/MediPhi-Instruct",
    "legal": "./experts/legal/Saul-7B-Instruct-v1",
    "code": "./experts/coding/Qwen2.5-Coder-3B",
}

class ExpertRouter:
    def __init__(self):
        self.current_domain = None
        self.model = None
        self.tokenizer = None

    def unload_current_model(self):
        """Unload current model from GPU to free VRAM."""
        if self.model is not None:
            try:
                # delete and free memory
                del self.model
                del self.tokenizer
            except Exception:
                pass
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()

    def load_expert(self, domain: str):
        """Load expert model for given domain. Unload if switching."""
        if domain == self.current_domain and self.model is not None:
            # Already loaded -> do nothing
            return

        # Unload old model
        self.unload_current_model()

        model_path = DOMAIN_TO_PATH.get(domain)
        if model_path is None:
            raise ValueError(f"Unknown domain: {domain}")

        print(f"[Router] Loading expert model for domain: {domain} from {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model using multi-GPU device_map auto for efficiency
        # device_map='auto' will split across available GPUs (works well for large models)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.current_domain = domain

    def generate(self, query: str, max_tokens=256) -> str:
        """Generate output using the currently loaded expert."""
        if self.model is None:
            raise RuntimeError("No expert model loaded.")

        inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
