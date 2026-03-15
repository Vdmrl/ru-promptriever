# RuPromptriever

Implementation of **Promptriever** (Instruction-Trained Retrievers) adapted for the Russian language

## Overview

Modern dense retrieval models typically rely on semantic similarity between a query and a document. This project aims to replicate the Promptriever architecture, which enables the retrieval model to follow complex, instance-level natural language instructions (e.g., "retrieve documents about X but exclude Y").

The goal is to train a bi-encoder that can be controlled via prompts, effectively transferring the instruction-following capabilities of LLMs to dense retrieval tasks in Russian.

## Methodology

The pipeline consists of the following stages:

1. **Data Curation:** Using the Russian split of the mMARCO dataset as the source.
2. **Synthetic Data Generation:**
   * Generating specific instructions for existing queries using LLMs (GigaChat-2-Max).
   * Mining "Instruction Negatives" — synthetic documents that are relevant to the query but irrelevant to the specific instruction.
3. **Filtration:** 
   * Validating synthetic triplets using an LLM-based pipeline (`GigaChat-2-Lite`) to filter out bad positive matches and weak negatives.
   * Ensuring negatives genuinely violate the generated instructions.
4. **Training:** Fine-tuning a backbone LLM (Qwen3 8b/4b) as a bi-encoder using the curated dataset.

## References

* **Original Paper:** [Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models](https://arxiv.org/abs/2409.11136) (Weller et al., 2024).
* **Base Dataset:** [mMARCO](https://github.com/unicamp-dl/mMARCO).

## Installation

To run the training pipeline, you must install the dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/Vdmrl/ru-promptriever.git
cd ru-promptriever/training_pipeline

# 2. Install base requirements
pip install -r requirements.txt
```

```bash
# 4. Authenticate to Hugging Face (to download the model and datasets):
huggingface-cli login

# 5. Export WandB API key (from https://wandb.ai/authorize)
export WANDB_API_KEY="your_api_key_here"

```bash
# 6. Run training script
torchrun --nproc_per_node=2 train.py --config configs/v0.2_optimized4b_2_5090.yaml

# 7. (Optional) Train with duplicated queries included
torchrun --nproc_per_node=2 train.py --config configs/v0.2_optimized4b_2_5090.yaml --use-repeated
```

## Data Preprocessing Pipeline

To run the data generation and filtering pipeline locally (e.g., preparing the dataset before training):

```bash
cd ru-promptriever/data_preprocessing

# 1. Install specific requirements
pip install -r requirements.txt

# 2. Filter synthetic data using LLM
# Be sure to set your GigaChat credentials in configs/config.yaml
python filter_data.py --input_dir data/input --output_dir data/output_filtered

# 3. Recover missing triplets (optional)
# Use this if you have filtered out queries and want to re-run them
python extract_missing_triplets.py

# 4. Build the final training/eval dataset (creates parquet files)
python build_dataset.py --filtered_dir data/output_filtered --output_dir data/output_final_dataset
```

## Post-Training

After the training finishes, only the lightweight LoRA adapter weights are saved. To merge these weights with the base model and push the final, standalone model to Hugging Face, run the `merge_lora.py` script:

```bash
python3 merge_lora.py \
    --base_model_name_or_path "Qwen/Qwen3-4B" \
    --lora_model_path "./output_v0.2_optimized4b" \
    --output_dir "./merged_ru_promptriever" \
    --push_to_hub "Vladimirlv/ru-promptriever-qwen3-4b"
```

* `--lora_model_path`: Can be a local path or a Hugging Face repo ID.
* `--push_to_hub`: (Optional) Automatically pushes the final merged model and its tokenizer to your Hugging Face account.

## Evaluation Pipeline

The `evaluation_pipeline/` directory contains a full benchmarking suite to compare Russian retrieval models on **Synthetic Test (ru-promptriever)**, **mFollowIR-RU**, and **ruMTEB** datasets. It supports `bm25s`, `sentence-transformers`, causal LLMs (last-token pooling), and the custom instruction-following models.

### Quick Start (Deployment on Cloud GPUs like Vast.ai)

1. **Clone and setup base environment**:

   ```bash
   git clone https://github.com/Vdmrl/ru-promptriever.git
   cd ru-promptriever/evaluation_pipeline
   ```
2. **Install evaluation requirements**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Login to Hugging Face** (required to pull private models/datasets and upload results):

   ```bash
   huggingface-cli login
   ```
   *Alternative: If `huggingface-cli` is not found, you can login via Python:*
   ```bash
   python3 -c "from huggingface_hub import login; login('ВАШ_ТОКЕН_ЗДЕСЬ')"
   ```

### Running the Evaluation

All settings (models, datasets, batch sizes) are configured in `configs/baseline_qwen3-4b.yaml`.

```bash
# 1. Smoke test (5 queries per model) to ensure no memory errors:
python3 evaluate.py --config configs/baseline_qwen3-4b.yaml --max-queries 5

# 2. Full evaluation with automatic intermediate uploads to Hugging Face:
python3 evaluate.py --config configs/baseline_qwen3-4b.yaml --hf-repo "Vladimirlv/ru-promptriever-benchmark-results"

# 3. Skip existing results if resuming an interrupted run:
python3 evaluate.py --config configs/baseline_qwen3-4b.yaml --skip-existing
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
