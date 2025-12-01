# download_experts.py

import os
from huggingface_hub import snapshot_download
from pathlib import Path

# --- 1. Model Mapping ---
HF_MODEL_MAPPING = {
    "biomedical": "microsoft/MediPhi-Instruct",
    "legal": "Equall/Saul-7B-Instruct-v1",
    "code": "Qwen/Qwen2.5-Coder-3B",
}

# Define the target local structure based on model_router.py
TARGET_LOCAL_PATHS = {
    "biomedical": "./experts/biomedical/MediPhi-Instruct",
    "legal": "./experts/legal/Saul-7B-Instruct-v1",
    "code": "./experts/coding/Qwen2.5-Coder-3B",
}

def download_expert_models():
    """
    Downloads all expert LLMs from the Hugging Face Hub to their specified local paths.
    """
    print("--- Starting Expert Model Download ---")
    
    for domain, hf_repo_id in HF_MODEL_MAPPING.items():
        local_dir = TARGET_LOCAL_PATHS[domain]
        
        # Ensure the target directory structure is correctly defined
        local_path = Path(local_dir).resolve() 
        
        # Check if the model directory already exists and seems complete
        if (local_path / "config.json").exists():
            print(f"✅ {domain} model already exists at {local_dir}. Skipping download.")
            continue
            
        print(f"\n[DOWNLOADING] Domain: {domain}")
        print(f"  Repo: {hf_repo_id}")
        print(f"  Target Path: {local_dir}")
        
        try:
            # Use snapshot_download to get all files in the repository
            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False, # Copy files directly for robustness
                allow_patterns=["*.json", "*.bin", "*.safetensors", "tokenizer*", "*.py"],
                # tqdm=True (optional, for progress bar)
            )
            print(f"--- Successfully downloaded {domain} model. ---")
            
        except Exception as e:
            print(f"❌ Failed to download {domain} model ({hf_repo_id}). Error: {e}")
            print("Please ensure the Hugging Face Repo ID is correct and accessible.")

if __name__ == "__main__":
    download_expert_models()