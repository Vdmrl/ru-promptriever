# RuPromptriever

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2409.11136-b31b1b.svg)](https://arxiv.org/abs/2409.11136)
[![HF Model](https://img.shields.io/badge/🤗%20Model-ru--promptriever--qwen3--4b-yellow)](https://huggingface.co/Vladimirlv/ru-promptriever-qwen3-4b)
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-ru--promptriever--dataset-blue)](https://huggingface.co/datasets/Vladimirlv/ru-promptriever-dataset-v0.1)
[![HF Results](https://img.shields.io/badge/🤗%20Results-benchmark--results-green)](https://huggingface.co/datasets/Vladimirlv/ru-promptriever-benchmark-results)
[![Code License](https://img.shields.io/badge/Code%20License-MIT-lightgrey)](LICENSE)
[![Model License](https://img.shields.io/badge/Model%20%26%20Data%20License-CC%20BY--NC%204.0-orange)](https://creativecommons.org/licenses/by-nc/4.0/)

**Instruction-following dense retrieval for Russian** — a Russian-language adaptation of [Promptriever](https://arxiv.org/abs/2409.11136) (Weller et al., 2024).

</div>

---

## Overview

Standard dense retrieval models score query–passage pairs using a single semantic similarity signal, giving users little control over what "relevant" means beyond keyword choice. **Promptriever** (Weller et al., 2024) introduces *per-instance* natural language instructions that redefine relevance on a query-by-query basis — a capability previously limited to generative language models.

**RuPromptriever** extends this approach to Russian by:

1. Building a synthetic instruction dataset on top of the Russian split of [mMARCO](https://github.com/unicamp-dl/mMARCO).
2. Training a Qwen3-4B bi-encoder with QLoRA + GradCache on the curated data.
3. Evaluating instruction-following retrieval quality on **mFollowIR-RU**, a **synthetic test split**, and **ruMTEB** benchmarks.

The core insight from the original paper (replicated here in Russian): instruction-following capacity is *not* retained after standard IR fine-tuning. Two additions to the training data are required — **instructions that redefine per-query relevance** and **instruction-negative passages** (documents that are topically relevant but violate the instruction constraint).

---

## Repository Structure

```
.
├── data_generation/          # Stage 1: LLM-based instruction + negative synthesis
│   ├── main.py               # Entry point; thread-pool orchestration
│   └── utils/
│       ├── data_loader.py    # Streams mMARCO triples from disk
│       ├── llm_init.py       # GigaChat / OpenAI client factory
│       ├── processor.py      # Two-stage generation pipeline per sample
│       ├── prompts.py        # Prompt templates and Pydantic output schemas
│       └── scheduler.py      # Time-based thread-count scheduler (MSK)
│
├── data_preprocessing/       # Stage 2: LLM-based filtering + dataset assembly
│   ├── filter_data.py        # Validates positives and negatives with an LLM
│   ├── build_dataset.py      # Assembles parquet shards with BM25 hard negatives
│   ├── reformat_parquet.py   # Re-schemas existing parquet for HF Viewer compat.
│   ├── extract_missing_triplets.py  # Identifies and re-queues filtered-out queries
│   └── utils/
│       ├── bm25.py           # BM25 index wrapper (bm25s + Snowball stemming)
│       ├── io.py             # JSONL read/write helpers
│       ├── llm_init.py       # Same client factory as data_generation
│       ├── processor.py      # Filter logic: positive check + negative validation
│       ├── prompts.py        # Filter prompt templates
│       └── scheduler.py      # Shared MSK-based thread scheduler
│
├── training_pipeline/        # Stage 3: QLoRA fine-tuning with GradCache
│   ├── train.py              # Main training script (single-GPU + DeepSpeed)
│   ├── merge_lora.py         # Merges LoRA adapter into base model
│   ├── configs/              # YAML training configs per experiment
│   └── utils/
│       ├── data.py           # RetrieverDataset + RetrieverCollator
│       └── trainer.py        # EncoderWrapper, ContrastiveLoss, RetrieverTrainer
│
└── evaluation_pipeline/      # Stage 4: Benchmarking suite
    ├── evaluate.py           # Main evaluation script
    ├── configs/              # YAML evaluation configs
    ├── models/               # Retriever wrappers (BM25, E5, BGE, Qwen3, etc.)
    ├── tasks/                # Custom MTEB tasks (synthetic test, mFollowIR-RU)
    └── utils/                # Data loading and metric helpers
```

---

## Data Generation Pipeline

The pipeline mirrors the two-stage process from Weller et al. (2024) with adaptations for Russian.

### Source Data

Triples are sourced from two complementary datasets:
- **mMARCO-RU** ([unicamp-dl/mmarco](https://huggingface.co/datasets/unicamp-dl/mmarco)) — Russian split of MS MARCO passage ranking.
- **Tevatron MS MARCO aug** ([Tevatron/msmarco-passage-aug](https://huggingface.co/datasets/Tevatron/msmarco-passage-aug)) — the hard-negative augmented version used by RepLLaMA, used to supplement missing triples from the mMARCO split.

### Stage 1 — Instruction & Negative Synthesis (`data_generation/`)

For each `(query, positive, negative)` triple from mMARCO-RU:

1. **Instruction generation** (`GigaChat-2-Max`): The model rewrites the machine-translated query into natural Russian and generates a retrieval instruction that keeps the original positive relevant while excluding the negative. Instructions vary in length (short / medium / long / very long) and style (negation / persona / background / feature).
2. **Instruction-negative mining** (`GigaChat-2-Max`): Using the rewritten query and its instruction, the model synthesizes three new passages — one query-positive/instruction-positive (backup positive) and two query-positive/instruction-negative candidates.

```bash
# Install dependencies
pip install -r data_generation/requirements.txt

# Configure your LLM credentials
# → data_generation/configs/config.yaml

# Run generation (adjust --limit and --offset for parallel workers)
python data_generation/main.py \
    --config data_generation/configs/config.yaml \
    --input data_generation/data/input/triples.train.ids.small.tsv \
    --output data_generation/data/output/ \
    --limit 50000 \
    --offset 0
```

### Stage 2 — Filtering & Dataset Assembly (`data_preprocessing/`)

The generated data is validated by a cheaper LLM (`GigaChat-2-Lite`) before being assembled into the final dataset.

```bash
pip install -r data_preprocessing/requirements.txt

# 1. Validate positives and instruction-negatives with an LLM
python data_preprocessing/filter_data.py \
    --input_dir  data_preprocessing/data/input \
    --output_dir data_preprocessing/data/output_filtered

# 2. (Optional) Re-queue queries that were discarded during filtering
python data_preprocessing/extract_missing_triplets.py

# 3. Assemble train / val / test parquet shards with BM25 hard negatives
python data_preprocessing/build_dataset.py \
    --filtered_dir data_preprocessing/data/output_filtered \
    --output_dir   data_preprocessing/data/output_final_dataset \
    --push_to_hub  "Vladimirlv/ru-promptriever-dataset"

# 4. (Optional) Upload an already-built local dataset manually
huggingface-cli upload Vladimirlv/ru-promptriever-dataset \
    data_preprocessing/data/output_final_dataset \
    --repo-type dataset
```

**Filtering semantics:**
- A record is **kept** if: (a) the original positive is judged relevant to `(query + instruction)` by the LLM, **or** (b) the backup generated positive passes the same check.
- Instruction-negative candidates that are judged relevant to the instruction are discarded.
- Discarded records are logged to `deleted_queries.jsonl` / `deleted_negatives.jsonl` for post-hoc analysis.

---

## Training

Fine-tuning uses **QLoRA** (4-bit NF4 quantization + LoRA rank-32) with **GradCache** for large effective batch sizes on limited GPU memory. The model is trained with an InfoNCE contrastive loss using last-token pooling (EOS pooling), matching the RepLLaMA / original Promptriever convention.

### Installation

```bash
cd training_pipeline
pip install -r requirements.txt

# Authenticate to Hugging Face (downloads Qwen3-4B and the dataset)
huggingface-cli login

# (Optional) Set WandB API key
export WANDB_API_KEY="your_key_here"
```

### Running Training

```bash
# Single GPU
python train.py --config configs/exp3_qwen3-4b_fast.yaml

# Multi-GPU with DeepSpeed (recommended: 2× RTX 5090, ~30–40 h/epoch)
torchrun --nproc_per_node=2 train.py \
    --config configs/exp3_qwen3-4b_fast.yaml

# Include duplicate query variants (disabled by default)
torchrun --nproc_per_node=2 train.py \
    --config configs/exp3_qwen3-4b_fast.yaml \
    --use-repeated
```

**Key config parameters** (`configs/exp3_qwen3-4b_fast.yaml`):

| Parameter | Value | Description |
|---|---|---|
| `model_name_or_path` | `Qwen/Qwen3-4B` | Base causal LM |
| `lora_r` / `lora_alpha` | 32 / 64 | LoRA rank and scaling factor |
| `num_negatives` | 7 | Hard negatives per query (3 instruction + 4 BM25) |
| `gc_chunk_size` | 16 | GradCache sub-batch size |
| `per_device_train_batch_size` | 8 | Physical batch size per GPU |
| `gradient_accumulation_steps` | 8 | → Effective batch = 8 × 8 × 2 GPUs = 128 |
| `temperature` | 0.01 | InfoNCE temperature |
| `instruct_only` | `true` | Train only on instruction-augmented rows (~500 k) |

Training checkpoints are automatically pushed to HuggingFace Hub every 500 steps. If interrupted, the script resumes from the latest local or remote checkpoint automatically.

### Post-Training: Merging LoRA

After training, merge the LoRA adapter into the base model for standalone inference:

```bash
python merge_lora.py \
    --base_model_name_or_path "Qwen/Qwen3-4B" \
    --lora_model_path          "./output_v0.2_optimized4b" \
    --output_dir               "./merged_ru_promptriever" \
    --push_to_hub              "Vladimirlv/ru-promptriever-qwen3-4b"
```

`--push_to_hub` is optional. Omit it to save the merged model locally only.

---

## Evaluation

The evaluation pipeline benchmarks models across four task categories:

| Task | Dataset key | Metric(s) | Description |
|---|---|---|---|
| **Synthetic test** | `synthetic_test` | nDCG@20, p-MRR | Test split of our dataset; paired standard + instructed queries |
| **mFollowIR-RU** | `mfollowir_ru` | nDCG@20, p-MRR | Russian split of mFollowIR (Weller et al., 2025); TREC NeuCLIR narratives as instructions |
| **ruMTEB Retrieval** | `rumteb_retrieval` | nDCG@10 | Standard Russian retrieval benchmarks (RuBQRetrieval, etc.) via MTEB |
| **EN MTEB (sanity check)** | `en_mteb_retrieval` | nDCG@10 | SciFact + NFCorpus; verifies scores match the published Promptriever paper |

**p-MRR** (Pairwise Mean Reciprocal Rank) is the primary instruction-following metric: it measures how much a model adjusts rankings for documents whose relevance *changes* between the original and modified instructions. A score of 0 means the model ignores instructions; positive scores indicate correct ranking adjustments.

### Supported Model Types

| Type key | Examples |
|---|---|
| `bm25` | Sparse baseline (bm25s + Snowball stemming) |
| `encoder` | `intfloat/multilingual-e5-large`, `BAAI/bge-m3` |
| `giga_embedding` | `ai-sage/Giga-Embeddings-instruct` |
| `qwen3_embedding` | `Qwen/Qwen3-Embedding-4B` |
| `causal_lm` | `samaya-ai/promptriever-llama3.1-8b-v1`, `Vladimirlv/ru-promptriever-qwen3-4b` |

### Quick Start

```bash
cd evaluation_pipeline
pip install -r requirements.txt
huggingface-cli login
```

```bash
# Smoke test (5 queries per model) — verifies no OOM errors before a full run
python evaluate.py --config configs/baseline_qwen3-4b.yaml --max-queries 5

# Full evaluation with automatic intermediate uploads to HF Hub
python evaluate.py \
    --config   configs/baseline_qwen3-4b.yaml \
    --hf-repo  "Vladimirlv/ru-promptriever-benchmark-results"

# Resume an interrupted run (skips already-computed model×dataset pairs)
python evaluate.py --config configs/baseline_qwen3-4b.yaml --skip-existing

# Targeted evaluation: one model on specific tasks
HF_HUB_HTTP_TIMEOUT=300 python evaluate.py \
    --config   configs/baseline_qwen3-4b.yaml \
    --models   ru-promptriever-qwen3-4b \
    --datasets mfollowir_ru synthetic_test
```

Results are saved as JSON files under `evaluation_pipeline/results/`. If `--hf-repo` is set, the results folder is uploaded to your HuggingFace Dataset repository after every successful model×dataset evaluation, preventing data loss on preemptible cloud instances.

To upload results manually after the fact:

```python
from huggingface_hub import HfApi

repo_id = "Vladimirlv/ru-promptriever-benchmark-results"
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
api.upload_folder(
    folder_path="./results",
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="run_1",
)
```

---

## Datasets & Models

| Artifact | Link |
|---|---|
| Training dataset v0.1 | [![HF](https://img.shields.io/badge/🤗-ru--promptriever--dataset--v0.1-blue)](https://huggingface.co/datasets/Vladimirlv/ru-promptriever-dataset-v0.1) |
| Trained model (merged) | [![HF](https://img.shields.io/badge/🤗-ru--promptriever--qwen3--4b-yellow)](https://huggingface.co/Vladimirlv/ru-promptriever-qwen3-4b) |
| Benchmark results | [![HF](https://img.shields.io/badge/🤗-benchmark--results-green)](https://huggingface.co/datasets/Vladimirlv/ru-promptriever-benchmark-results) |
| mFollowIR (eval) | [![HF](https://img.shields.io/badge/🤗-mFollowIR-orange)](https://huggingface.co/datasets/jhu-clsp/mFollowIR) |
| Source triples (RepLLaMA aug) | [![HF](https://img.shields.io/badge/🤗-msmarco--passage--aug-lightgrey)](https://huggingface.co/datasets/Tevatron/msmarco-passage-aug) |
| Source mMARCO dataset | [![HF](https://img.shields.io/badge/🤗-mmarco-lightgrey)](https://huggingface.co/datasets/unicamp-dl/mmarco) |

---

## License

This repository uses a dual-license structure:

| Component | License | Reason |
|---|---|---|
| **Source code** (`*.py`, `*.yaml`, `*.json`) | [MIT](LICENSE) | Original authorship, no third-party data restrictions |
| **Trained model** ([Vladimirlv/ru-promptriever-qwen3-4b](https://huggingface.co/Vladimirlv/ru-promptriever-qwen3-4b)) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | Derived from MS MARCO (Microsoft Research License — non-commercial) |
| **Dataset** ([Vladimirlv/ru-promptriever-dataset](https://huggingface.co/datasets/Vladimirlv/ru-promptriever-dataset-v0.1)) | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | Contains synthetically transformed MS MARCO content |

The non-commercial restriction on the model and dataset originates from the upstream [MS MARCO license](https://microsoft.github.io/msmarco/) and cannot be lifted by downstream authors. The source code itself is freely usable under MIT.

---

## References

- **Promptriever** — Weller et al., 2024. [*Promptriever: Instruction-Trained Retrievers Can Be Prompted Like Language Models*](https://arxiv.org/abs/2409.11136). arXiv:2409.11136.
- **mFollowIR** — Weller et al., 2025. [*mFollowIR: Multilingual Following of Repository Relevance Instructions*](https://arxiv.org/abs/2501.03516). arXiv:2501.03516.
- **RepLLaMA** — Ma et al., 2023. *Fine-Tuning LLaMA for Multi-Stage Text Retrieval*.
- **mMARCO** — Bonifacio et al., 2022. [mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset](https://github.com/unicamp-dl/mMARCO).
- **Tevatron MS MARCO aug** — Gao et al., 2022. [Tevatron: An Efficient and Flexible Toolkit for Dense Retrieval](https://github.com/texttron/tevatron).
- **GradCache** — Gao & Callan, 2021. [*Scaling Deep Contrastive Learning Batch Size with Almost Free Memory Cost*](https://aclanthology.org/2021.repl4nlp-1.31/).
