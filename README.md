# MoE-with-SLMs

This project implements a Mixture-of-Experts (MoE) system using Small Language Models (SLMs) that dynamically routes queries to domain-specific expert models. This project evaluates domain-specific expert LLMs (Biomedical, Legal, Code) against a single baseline model (≤8B parameters) and implements an intelligent routing mechanism using a fine-tuned DistilBERT classifier.

## Overview

**Key Features:**
- 🎯 **Domain-Specific Experts**: Specialized models optimized for biomedical, legal, and coding tasks
- 🔀 **Smart Routing**: DistilBERT-based classifier for automatic domain detection
- 📊 **Comprehensive Evaluation**: Performance metrics, latency measurements, and VRAM usage tracking
- 💾 **Memory Efficient**: Sequential loading/unloading of expert models to optimize resource usage

**Supported Domains:**
- `biomedical` - Medical and healthcare queries
- `legal` - Legal questions and guidance
- `code` - Programming problems and solutions

**Expert Models:**
- **Biomedical**: `microsoft/MediPhi-Instruct` (Phi-3.5 family, ~3.8B parameters)
- **Legal**: `Equall/Saul-7B-Instruct-v1` (Llama-3.2 family, ~7B parameters)
- **Code**: `Qwen/Qwen2.5-Coder-3B` (Qwen2.5 family, ~3B parameters)

**Baseline Model**: `mistralai/Mistral-7B-Instruct-v0.3` (or any other ≤8B model)

## Project Goals

1. **Baseline Evaluation**: Assess a single general-purpose model on all domain-specific datasets
2. **Expert Evaluation**: Measure performance of domain-specific experts with sequential routing
3. **Classifier Training**: Build and train a DistilBERT classifier for automatic domain routing
4. **Performance Analysis**: Compare metrics, latency, and resource usage between baseline and expert systems

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for model inference)
- At least 16GB RAM
- 50GB+ free disk space for models and datasets

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/moe-with-slms.git
cd moe-with-slms
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download expert models:**
```bash
python download_experts.py
```
This will download all three expert models from Hugging Face Hub to the `experts/` directory.

## Quick Start

### Step 1: Fetch Datasets

Download and prepare the evaluation datasets:

```bash
python scripts/fetch_datasets.py
```

**Note**: Some datasets (BioASQ, LEGALBENCH) may require registration or manual download. The script will provide instructions when manual action is needed.

**Datasets Used:**
- **Biomedical**: PubMedQA (`pubmed_qa/pqa_labeled`)
- **Legal**: Law-StackExchange (`ymoslem/Law-StackExchange`)
- **Code**: MBPP (Mostly Basic Programming Problems)

### Step 2: Build Classifier Dataset

Process the raw datasets into expert evaluation and classifier training formats:

```bash
python classifier/build_dataset_from_experts.py
```

This creates:
- `dataset/expert_eval/*.json` - Expert evaluation datasets (1000 samples per domain)
- `dataset/classifier/classifier_train.json` - Classifier training data (80% split)
- `dataset/classifier/classifier_test.json` - Classifier test data (20% split)

### Step 3: Train Domain Classifier

Train the DistilBERT-based 3-class domain classifier:

```bash
python classifier/train_classifier.py
```

The trained model will be saved to `classifier/model/` with test metrics in `classifier/model/test_results.json`.

### Step 4: Run Evaluations

**Baseline Evaluation** (single model on all domains):

```bash
python scripts/eval_baseline.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --results_dir results/baseline \
  --max_new_tokens 256 \
  --max_examples 100
```

**Expert Evaluation** (domain-specific models):

```bash
python scripts/eval_experts_sequence.py \
  --results_dir results/experts \
  --max_new_tokens 256 \
  --max_examples 100
```

### Step 5: Interactive Demo

Try the interactive expert router:

```bash
python main.py
```

Type your queries and see the system automatically detect the domain and route to the appropriate expert. Type `exit` to quit.

## Project Structure

```
moe-with-slms/
├── classifier/                    # Domain classification module
│   ├── build_dataset_from_experts.py  # Dataset preprocessing
│   ├── train_classifier.py        # DistilBERT training script
│   ├── infer_classifier.py        # Inference wrapper
│   └── model/                     # Trained classifier checkpoints
├── router/                        # Expert routing system
│   ├── model_router.py            # Expert model loader/unloader
│   └── prompts.py                 # Domain-specific prompt templates
├── scripts/                       # Evaluation scripts
│   ├── fetch_datasets.py          # Dataset download utility
│   ├── eval_baseline.py           # Baseline model evaluation
│   ├── eval_experts_sequence.py   # Expert models evaluation
│   └── eval/                      # Evaluation utilities
│       ├── data_utils.py          # Dataset loading helpers
│       ├── evaluate.py            # Main evaluation dispatcher
│       ├── metrics_biomedical.py  # Biomedical metrics
│       ├── metrics_legal.py       # Legal metrics
│       └── metrics_code.py        # Code generation metrics
├── experts/                       # Downloaded expert models (created by download_experts.py)
│   ├── biomedical/
│   ├── legal/
│   └── coding/
├── dataset/                       # Downloaded and processed datasets
│   ├── pubmedqa/
│   ├── law_stackexchange/
│   ├── mbpp/
│   ├── expert_eval/               # Evaluation datasets
│   └── classifier/                # Classifier training data
├── results/                       # Evaluation results and metrics
├── download_experts.py            # Expert model download script
├── main.py                        # Interactive demo
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Evaluation Metrics

### Domain-Specific Metrics

**Biomedical (PubMedQA):**
- Decision Accuracy: Correct yes/no/maybe classification
- ROUGE-L F1: Semantic similarity of long-form answers

**Legal (Law-StackExchange):**
- ROUGE-L F1: Quality of legal advice
- Answer completeness and relevance

**Code (MBPP):**
- Exact Match: Syntactic code similarity
- Pass@k: Functional correctness (when unit tests available)

### System Metrics
- **Latency**: Per-query inference time
- **VRAM Usage**: Peak GPU memory consumption
- **Model Load Time**: Time to load/unload expert models

## Advanced Usage

### Custom Expert Models

To use different expert models, modify `DOMAIN_TO_PATH` in `router/model_router.py` and update `download_experts.py`:

```python
DOMAIN_TO_PATH = {
    "biomedical": "./experts/biomedical/your-model",
    "legal": "./experts/legal/your-model",
    "code": "./experts/coding/your-model",
}
```

### Multi-GPU Inference

The router automatically uses `device_map="auto"` for efficient multi-GPU distribution. For manual control:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map={"": 0}  # Force specific GPU
)
```

### DeepSpeed Integration

For optimized inference with DeepSpeed, uncomment the DeepSpeed dependency in `requirements.txt` and modify the model loading in `router/model_router.py`.

### Custom Domains

To add a new domain:

1. Add the domain to `DOMAINS` list in `classifier/build_dataset_from_experts.py`
2. Create a processing function for your dataset
3. Update `LABEL2ID` in `classifier/train_classifier.py`
4. Add evaluation metrics in `scripts/eval/metrics_<domain>.py`
5. Update the router's `DOMAIN_TO_PATH` mapping

## Configuration

### Command-Line Arguments

**eval_baseline.py:**
- `--model, -m`: HuggingFace model ID or local path (default: `mistralai/Mistral-7B-Instruct-v0.3`)
- `--results_dir, -o`: Output directory for results (default: `results`)
- `--max_new_tokens`: Maximum tokens to generate (default: 256)
- `--max_examples`: Limit examples per domain (default: 47)

**eval_experts_sequence.py:**
- `--results_dir, -o`: Output directory for results (default: `results`)
- `--max_new_tokens`: Maximum tokens to generate (default: 256)
- `--max_examples`: Limit examples per domain (default: 47)

### Training Configuration

Edit `classifier/train_classifier.py` to adjust:
- `MODEL_NAME`: Base classifier model (default: `distilbert-base-uncased`)
- `NUM_LABELS`: Number of domain classes (default: 3)
- Training hyperparameters in `TrainingArguments`

## Troubleshooting

### Out of Memory Errors
- Reduce `per_device_train_batch_size` in classifier training
- Lower `max_new_tokens` during inference
- Use smaller expert models or quantization

### Dataset Download Issues
- Some datasets require manual registration (BioASQ, LEGALBENCH)
- Check Hugging Face Hub access for gated models
- Verify internet connection and disk space

### Model Loading Errors
- Ensure expert models are downloaded with `download_experts.py`
- Check CUDA compatibility with PyTorch version
- Verify sufficient disk space in `experts/` directory

## Performance Tips

1. **Use FP16**: Models automatically use `torch.float16` for faster inference
2. **Batch Processing**: Modify evaluation scripts for batch inference
3. **Model Quantization**: Use 8-bit or 4-bit quantization with `bitsandbytes`
4. **Cache Models**: Set `HF_HOME` environment variable to cache downloaded models

## Citation

If you use this project in your research, please cite:

```bibtex
@software{moe_with_slms,
  title={MoE-with-SLMs: Mixture of Experts using Small Language Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/moe-with-slms}
}
```

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- **Expert Models**: Thanks to Microsoft (MediPhi), Equall (Saul), and Qwen team
- **Datasets**: PubMedQA, Law-StackExchange, MBPP communities
- **Framework**: Built with 🤗 Transformers and PyTorch

## Contact

For questions, issues, or feature requests, please open an issue on GitHub or contact the maintainers.
