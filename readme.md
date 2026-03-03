# RuPromptriever

Implementation of **Promptriever** (Instruction-Trained Retrievers) adapted for the Russian language
## Overview

Modern dense retrieval models typically rely on semantic similarity between a query and a document. This project aims to replicate the Promptriever architecture, which enables the retrieval model to follow complex, instance-level natural language instructions (e.g., "retrieve documents about X but exclude Y").

The goal is to train a bi-encoder that can be controlled via prompts, effectively transferring the instruction-following capabilities of LLMs to dense retrieval tasks in Russian.

## Methodology

The pipeline consists of the following stages:

1.  **Data Curation:** Using the Russian split of the mMARCO dataset as the source.
2.  **Synthetic Data Generation:**
    *   Generating specific instructions for existing queries using LLMs.
    *   Mining "Instruction Negatives" — synthetic documents that are relevant to the query but irrelevant to the specific instruction.
3.  **Filtration:** Validating synthetic triplets using a multilingual Cross-Encoder.
4.  **Training:** Fine-tuning a backbone LLM (Qwen3 8b) as a bi-encoder using the curated dataset.

## References

*   **Original Paper:** [Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models](https://arxiv.org/abs/2409.11136) (Weller et al., 2024).
*   **Base Dataset:** [mMARCO](https://github.com/unicamp-dl/mMARCO).

## Installation

To run the training pipeline, you must install the dependencies. **Note for FlashAttention 2**: It cannot be installed via standard `requirements.txt` because it requires PyTorch to be present during its `pip` build process (build isolation issue). 

```bash
# 1. Clone the repository
git clone https://github.com/Vdmrl/ru-promptriever.git
cd ru-promptriever/training_pipeline

# 2. Install base requirements (this includes PyTorch if not present, and all HuggingFace libs)
pip install -r requirements.txt

# 3. Manually install FlashAttention 2 (Required for RTX 3090/4090/5090 etc.)
# The --no-build-isolation flag forces pip to use the PyTorch we just installed to compile the C++ CUDA kernels.
pip install flash-attn>=2.5.0 --no-build-isolation
```

```bash
# 4. Authenticate to Hugging Face (to download the model and datasets):
huggingface-cli login

# 5. Export WandB API key (from https://wandb.ai/authorize)
export WANDB_API_KEY="your_api_key_here"

```bash
# 6. Run training script
torchrun --nproc_per_node=2 train.py --config configs/v0.2_optimized4b_2_5090.yaml
```

## Post-Training

After the training finishes, only the lightweight LoRA adapter weights are saved. To merge these weights with the base model and push the final, standalone model to Hugging Face, run the `merge_lora.py` script:

```bash
python merge_lora.py \
    --base_model_name_or_path "Qwen/Qwen3-4B" \
    --lora_model_path "./output_v0.2_optimized4b" \
    --output_dir "./merged_ru_promptriever" \
    --push_to_hub "Vladimirlv/ru-promptriever-qwen3-4b"
```

*   `--lora_model_path`: Can be a local path or a Hugging Face repo ID.
*   `--push_to_hub`: (Optional) Automatically pushes the final merged model and its tokenizer to your Hugging Face account.

## Evaluation Pipeline

The `evaluation_pipeline/` directory contains a full benchmarking suite to compare Russian retrieval models on **Synthetic Test (ru-promptriever)**, **mFollowIR-RU**, and **ruMTEB** datasets. It supports `bm25s`, `sentence-transformers`, causal LLMs (last-token pooling), and the custom instruction-following models.

### Quick Start (Deployment on Cloud GPUs like Vast.ai)

1. **Clone and setup base environment**:
   ```bash
   git clone https://github.com/Vdmrl/ru-promptriever.git
   cd ru-promptriever/pythonProject
   pip install --upgrade pip wheel packaging ninja psutil
   ```

2. **Install Flash Attention safely** (requires `ninja` and limits workers to prevent Out-Of-Memory during compilation):
   ```bash
   MAX_JOBS=4 pip install flash-attn --no-build-isolation
   ```

3. **Install evaluation requirements**:
   ```bash
   cd evaluation_pipeline
   pip install -r requirements.txt
   ```

4. **Login to Hugging Face** (required to pull private models/datasets and upload results):
   ```bash
   huggingface-cli login
   ```

### Running the Evaluation

All settings (models, datasets, batch sizes) are configured in `configs/baseline_qwen3-4b.yaml`.

```bash
# 1. Smoke test (5 queries per model) to ensure no memory errors:
python evaluate.py --config configs/baseline_qwen3-4b.yaml --max-queries 5

# 2. Full evaluation with automatic intermediate uploads to Hugging Face:
python evaluate.py --config configs/baseline_qwen3-4b.yaml --hf-repo "Vladimirlv/ru-promptriever-benchmark-results"

# 3. Skip existing results if resuming an interrupted run:
python evaluate.py --config configs/baseline_qwen3-4b.yaml --skip-existing
```

### Uploading Results to Hugging Face

Results are automatically saved as JSON files in `evaluation_pipeline/results/`. 

If you pass the `--hf-repo "YourName/repo-name"` argument, the script will **automatically upload** the `results/` folder to your Hugging Face Dataset after every successful evaluation step. This prevents data loss if a cloud instance is interrupted.

If you didn't use the flag and want to upload them manually later, you can use:

```python
from huggingface_hub import HfApi

repo_id = "Vladimirlv/ru-promptriever-benchmark-results"
HfApi().create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
HfApi().upload_folder(
    folder_path="./results",
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="run_1" 
)
```