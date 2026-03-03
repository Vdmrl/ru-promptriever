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