"""
Evaluation pipeline entry point for ru-promptriever benchmarking.

Compares 6 models on 3 dataset groups:
  1. Synthetic Test (ru-promptriever-dataset test split)
  2. mFollowIR-RU (Russian instruction-following retrieval)
  3. ruMTEB Retrieval (RiaNewsRetrieval, RuBQRetrieval, MIRACLRetrieval)

Usage:
    # Full evaluation
    python evaluate.py --config configs/eval_config.yaml

    # Single model on a single dataset
    python evaluate.py --config configs/eval_config.yaml \
        --models ru-promptriever-qwen3-4b --datasets synthetic_test

    # Smoke test with limited queries
    python evaluate.py --config configs/eval_config.yaml \
        --models bm25 --datasets synthetic_test --max-queries 10
"""

import argparse
import logging
import os
from typing import Dict, List, Optional

from huggingface_hub import HfApi
import mteb
import numpy as np

from models.bm25_retriever import BM25Retriever
from models.promptriever_retriever import CausalLMRetriever
from models.prompt_utils import materialize_texts, resolve_prompt_name
from models.encoder_retriever import EncoderRetriever
from models.giga_embedding_retriever import GigaEmbeddingRetriever
from models.qwen3_embedding_retriever import Qwen3EmbeddingRetriever
from tasks.pmrr import compute_pmrr
from tasks.synthetic_test_task import RuPrompTrieverTestRetrieval
from utils.data_utils import load_config, print_summary_table, print_intermediate_result, save_results
from utils.run_manifest import (
    find_matching_result,
    git_revision,
    protocol_fingerprint,
    protocol_payload,
    write_run_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hugging Face integration
# ---------------------------------------------------------------------------


def upload_to_huggingface(output_dir: str, repo_id: str):
    """Upload the results directory to a Hugging Face Dataset repository."""
    try:
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=".",
        )
        logger.info(f"Successfully uploaded intermediate results to HF: {repo_id}")
    except Exception as e:
        logger.error(f"Failed to upload results to Hugging Face: {e}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def resolve_latest_checkpoint(path: str) -> str:
    """If path is a directory containing checkpoint-XXX, return the latest one."""
    if os.path.isdir(path):
        import glob
        checkpoints = glob.glob(os.path.join(path, "checkpoint-*"))
        if checkpoints:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            latest = checkpoints[-1]
            logger.info(f"Auto-resolved path {path} to latest checkpoint: {latest}")
            return latest
    return path


def load_model(model_cfg: dict, global_cfg: dict):
    """Instantiate a model wrapper based on config type."""
    model_type = model_cfg["type"]
    device = global_cfg.get("device", "cuda:0")
    dtype = global_cfg.get("dtype", "bfloat16")
    generic_instruction = global_cfg.get(
        "generic_instruction", "Найди релевантный документ."
    )
    
    # Resolve the path to the latest checkpoint if it's an output directory
    resolved_path = resolve_latest_checkpoint(model_cfg.get("model_name_or_path", ""))

    if model_type == "bm25":
        return BM25Retriever()

    elif model_type == "encoder":
        return EncoderRetriever(
            model_name_or_path=resolved_path,
            device=device,
            query_prefix=model_cfg.get("query_prefix", ""),
            passage_prefix=model_cfg.get("passage_prefix", ""),
            max_length=model_cfg.get("max_length", 512),
        )

    elif model_type == "causal_lm":
        return CausalLMRetriever(
            model_name_or_path=resolved_path,
            device=device,
            revision=model_cfg.get("revision"),
            dtype=dtype,
            max_length=model_cfg.get("max_length", 512),
            generic_instruction=generic_instruction,
            query_prefix=model_cfg.get("query_prefix", ""),
            passage_prefix=model_cfg.get("passage_prefix", ""),
            append_eos=model_cfg.get("append_eos"),
            mteb_document_title_separator=model_cfg.get(
                "mteb_document_title_separator"
            ),
            base_revision=model_cfg.get("base_revision"),
        )

    elif model_type == "qwen3_embedding":
        return Qwen3EmbeddingRetriever(
            model_name_or_path=resolved_path,
            device=device,
            max_length=model_cfg.get("max_length", 8192),
        )

    elif model_type == "giga_embedding":
        return GigaEmbeddingRetriever(
            model_name_or_path=resolved_path,
            device=device,
            query_prompt=model_cfg.get(
                "query_prompt",
                "Instruct: Дан вопрос, необходимо найти абзац текста с ответом\nQuery: ",
            ),
            max_length=model_cfg.get("max_length", 4096),
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------


def load_tasks(dataset_cfg: dict) -> List:
    """Load MTEB tasks for a dataset config entry."""
    ds_type = dataset_cfg["type"]

    if ds_type == "synthetic_test":
        required = ("data_path", "revision", "instruction_negative_field")
        missing = [key for key in required if not dataset_cfg.get(key)]
        if missing:
            raise ValueError(
                "Synthetic-test config must explicitly set " + ", ".join(missing)
            )
        return [
            RuPrompTrieverTestRetrieval(
                dataset_path=dataset_cfg["data_path"],
                revision=dataset_cfg["revision"],
                instruction_negative_field=dataset_cfg["instruction_negative_field"],
            )
        ]

    elif ds_type == "mfollowir":
        # Use MTEB's pinned InstructionReranking implementation.  It reranks
        # each query's official 1,000 candidates and computes nDCG on the
        # original instruction plus p-MRR across the original/changed pair.
        tasks = mteb.get_tasks(tasks=["mFollowIR"], languages=["rus"])
        loaded = list(tasks)
        if len(loaded) != 1:
            raise RuntimeError(f"Expected one Russian mFollowIR task, got {len(loaded)}")
        task = loaded[0]
        expected_revision = dataset_cfg.get("revision")
        actual_revision = task.metadata.dataset.get("revision")
        if expected_revision and expected_revision != actual_revision:
            raise ValueError(
                f"Configured mFollowIR revision {expected_revision} does not match "
                f"MTEB 2.10.5 revision {actual_revision}"
            )
        return loaded

    elif ds_type == "rumteb":
        task_names = dataset_cfg.get("task_names", [])
        # Limit to Russian only — without this, multilingual tasks like
        # MIRACLRetrieval download ALL 18 language subsets (many GB of data).
        tasks = mteb.get_tasks(tasks=task_names, languages=["rus"])
        return list(tasks)

    elif ds_type == "en_mteb":
        task_names = dataset_cfg.get("task_names", [])
        tasks = mteb.get_tasks(tasks=task_names, languages=["eng"])
        return list(tasks)

    else:
        raise ValueError(f"Unknown dataset type: {ds_type}")


# ---------------------------------------------------------------------------
# Query manipulation for model-specific instruction handling
# ---------------------------------------------------------------------------


def prepare_queries_for_model(
    queries: Dict[str, str],
    model_type: str,
    dataset_type: str,
    generic_instruction: str,
) -> Dict[str, str]:
    """Adjust queries based on model type and dataset type.

    Rules:
      - synthetic_test: queries already contain instructions
        (baked into the query text). ALL models receive the same queries
        (this is the standard FollowIR protocol — p-MRR measures sensitivity).

      - rumteb: queries have NO instructions. For CausalLM (Promptriever)
        models, append the generic instruction to avoid OOD. For encoders
        and BM25, leave queries as-is.
    """
    if dataset_type in ("rumteb", "en_mteb") and model_type == "causal_lm":
        return {qid: f"{text} {generic_instruction}" for qid, text in queries.items()}

    return queries


# ---------------------------------------------------------------------------
# Metric computation with pytrec_eval
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int] = (10, 20, 100),
) -> Dict[str, float]:
    """Compute nDCG, MAP, Recall, Precision using pytrec_eval.

    Args:
        qrels: {query_id: {doc_id: relevance_score}}
        results: {query_id: {doc_id: retrieval_score}}
        k_values: List of k values for metrics.

    Returns:
        Dict of metric_name -> score.
    """
    import pytrec_eval

    metrics_str = set()
    for k in k_values:
        metrics_str.add(f"ndcg_cut_{k}")
        metrics_str.add(f"map_cut_{k}")
        metrics_str.add(f"recall_{k}")
        metrics_str.add(f"P_{k}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, list(metrics_str))
    scores = evaluator.evaluate(results)

    # Average across queries
    avg_metrics = {}
    if scores:
        all_metric_names = list(next(iter(scores.values())).keys())
        for metric_name in all_metric_names:
            values = [q_scores[metric_name] for q_scores in scores.values()]
            avg_metrics[metric_name] = float(np.mean(values))

    return avg_metrics


# ---------------------------------------------------------------------------
# BM25 evaluation
# ---------------------------------------------------------------------------


def _hf_dataset_to_corpus(raw) -> Dict[str, dict]:
    """Convert a HuggingFace Dataset or plain dict to {doc_id: {text, title}} format."""
    if isinstance(raw, dict):
        # Already in expected format: {doc_id: {text, title, ...}}
        return raw
    # HuggingFace Dataset: iterate rows, each row has _id/text/title columns
    return {
        str(row.get("_id", row.get("id", ""))): {
            "text": row.get("text", ""),
            "title": row.get("title", ""),
        }
        for row in raw
    }


def _hf_dataset_to_queries(raw) -> Dict[str, str]:
    """Convert a HuggingFace Dataset or plain dict to {query_id: text} format."""
    if isinstance(raw, dict):
        return raw
    return {str(row.get("_id", row.get("id", ""))): row.get("text", "") for row in raw}


def _hf_dataset_to_qrels(raw) -> Dict[str, Dict[str, int]]:
    """Convert a HuggingFace Dataset or plain dict to {q_id: {doc_id: score}} format."""
    if isinstance(raw, dict):
        return raw
    result = {}
    for row in raw:
        q_id = str(row.get("query-id", row.get("query_id", "")))
        c_id = str(row.get("corpus-id", row.get("corpus_id", "")))
        score = int(row.get("score", 1))
        if q_id not in result:
            result[q_id] = {}
        result[q_id][c_id] = score
    return result


def _extract_task_data(task, split="test"):
    """Extract corpus/queries/relevant_docs from both custom and MTEB built-in tasks.

    The custom synthetic task sets task.corpus/queries/relevant_docs
    as plain Python dicts keyed by split (e.g. task.corpus["test"]).
    MTEB 1.14+ built-in tasks store them in task.dataset[subset][split].
    """
    if hasattr(task, "corpus") and task.corpus:
        # Check if keyed by split directly (our custom tasks)
        if split in task.corpus and getattr(task.corpus, "get", lambda x: None)(split):
            corpus_raw = task.corpus[split]
            queries_raw = (
                task.queries.get(split, {}) if hasattr(task, "queries") else {}
            )
            qrels_raw = (
                task.relevant_docs.get(split, {})
                if hasattr(task, "relevant_docs")
                else {}
            )
            return (
                _hf_dataset_to_corpus(corpus_raw),
                _hf_dataset_to_queries(queries_raw),
                _hf_dataset_to_qrels(qrels_raw),
            )
        else:
            # Older MTEB format
            subset = (
                list(task.corpus.keys())[0] if isinstance(task.corpus, dict) else None
            )
            if subset and split in task.corpus[subset]:
                corpus_raw = task.corpus[subset][split]
                queries_raw = (
                    task.queries.get(subset, {}).get(split, {})
                    if hasattr(task, "queries")
                    else {}
                )
                qrels_raw = (
                    task.relevant_docs.get(subset, {}).get(split, {})
                    if hasattr(task, "relevant_docs")
                    else {}
                )
                return (
                    _hf_dataset_to_corpus(corpus_raw),
                    _hf_dataset_to_queries(queries_raw),
                    _hf_dataset_to_qrels(qrels_raw),
                )

    # MTEB 1.14+ fallback: data stored in task.dataset
    if hasattr(task, "dataset") and task.dataset:
        # Get first subset (e.g. 'default', 'rus-rus')
        subset = list(task.dataset.keys())[0]
        subset_data = task.dataset[subset]

        if split in subset_data:
            split_data = subset_data[split]
            corpus_raw = split_data.get("corpus", [])
            queries_raw = split_data.get("queries", [])
            qrels_raw = split_data.get("relevant_docs", {})
            return (
                _hf_dataset_to_corpus(corpus_raw),
                _hf_dataset_to_queries(queries_raw),
                _hf_dataset_to_qrels(qrels_raw),
            )

    raise ValueError(
        f"Cannot extract data from task {type(task).__name__}: "
        "Task format not recognized after load_data()."
    )


def _trim_corpus_for_smoke_test(corpus, relevant_docs, max_noise=500):
    """Keep only relevant docs + a small random sample for smoke testing."""
    import random

    relevant_doc_ids = set()
    for qid, docs in relevant_docs.items():
        relevant_doc_ids.update(docs.keys())

    # Keep all relevant docs
    trimmed = {did: corpus[did] for did in relevant_doc_ids if did in corpus}

    # Add some random noise docs so retrieval isn't trivial
    remaining = [did for did in corpus if did not in relevant_doc_ids]
    noise_ids = random.sample(remaining, min(max_noise, len(remaining)))
    for did in noise_ids:
        trimmed[did] = corpus[did]

    logger.info(
        f"Smoke test: trimmed corpus from {len(corpus)} to {len(trimmed)} docs "
        f"({len(relevant_doc_ids)} relevant + {len(noise_ids)} noise)"
    )
    return trimmed


def evaluate_bm25(
    model: BM25Retriever,
    task,
    top_k: int = 1000,
    max_queries: int = None,
) -> Dict:
    """Run BM25 retrieval and compute metrics."""
    task.load_data()

    split = "test"
    corpus, queries, relevant_docs = _extract_task_data(task, split)

    if max_queries:
        query_ids = list(queries.keys())[:max_queries]
        queries = {qid: queries[qid] for qid in query_ids}
        relevant_docs = {
            qid: relevant_docs[qid] for qid in query_ids if qid in relevant_docs
        }
        corpus = _trim_corpus_for_smoke_test(corpus, relevant_docs)

    model.index_corpus(corpus)
    results = model.retrieve(queries, top_k=top_k)

    metrics = compute_retrieval_metrics(relevant_docs, results)
    return metrics


# ---------------------------------------------------------------------------
# Dense model evaluation (custom path for synthetic_test)
# ---------------------------------------------------------------------------


def evaluate_dense_custom(
    model,
    task,
    model_type: str,
    dataset_type: str,
    generic_instruction: str,
    batch_size: int = 32,
    top_k: int = 1000,
    max_queries: int = None,
) -> Dict:
    """Evaluate a dense model on the versioned synthetic test.

    Handles query preparation and metric computation manually.
    """
    task.load_data()

    split = "test"
    corpus, queries, relevant_docs = _extract_task_data(task, split)

    # Adjust queries for model type
    queries = prepare_queries_for_model(
        queries, model_type, dataset_type, generic_instruction
    )

    if max_queries:
        query_ids = list(queries.keys())[:max_queries]
        queries = {qid: queries[qid] for qid in query_ids}
        relevant_docs = {
            qid: relevant_docs[qid] for qid in query_ids if qid in relevant_docs
        }
        corpus = _trim_corpus_for_smoke_test(corpus, relevant_docs)

    # GigaEmbeddings wraps queries with a task-description prefix via prompt_name="query".
    # On the synthetic instruction-following dataset queries already embed
    # the full task instruction, so suppress this extra wrapping only for giga_embedding.
    # CausalLM models (Promptriever) use prompt_name="query" for a format token ("query: ")
    # that must always be applied regardless of dataset type.
    if dataset_type == "synthetic_test" and model_type == "giga_embedding":
        query_prompt_name = None
    else:
        query_prompt_name = "query"

    results = _dense_retrieve(model, queries, corpus, batch_size, top_k, query_prompt_name)

    metrics = compute_retrieval_metrics(relevant_docs, results)
    return results, metrics


# ---------------------------------------------------------------------------
# Dense model evaluation via MTEB (for ruMTEB built-in tasks)
# ---------------------------------------------------------------------------


class CausalLMRetrieverWithInstruction(mteb.EncoderProtocol):
    """Wrapper that injects a generic instruction into queries for MTEB.

    MTEB 2.10 identifies queries with ``prompt_type``.  Custom evaluation
    paths use ``prompt_name``.  This wrapper accepts both conventions.
    """

    def __init__(self, base_model: CausalLMRetriever):
        self.base_model = base_model
        self.generic_instruction = base_model.generic_instruction

    @property
    def mteb_model_meta(self):
        return self.base_model.mteb_model_meta

    @mteb_model_meta.setter
    def mteb_model_meta(self, value):
        self.base_model.mteb_model_meta = value

    def encode(self, sentences, batch_size=32, prompt_name=None, **kwargs):
        resolved_prompt_name = resolve_prompt_name(
            prompt_name, kwargs.get("prompt_type")
        )
        if resolved_prompt_name == "query":
            sentences = [
                f"{text.rstrip()} {self.generic_instruction}".strip()
                for text in materialize_texts(sentences)
            ]
        return self.base_model.encode(
            sentences,
            batch_size=batch_size,
            prompt_name=resolved_prompt_name,
            **kwargs,
        )

    def similarity(self, e1, e2):
        """Cosine similarity via dot product (embeddings are L2-normalized)."""
        import torch

        if not isinstance(e1, torch.Tensor):
            e1 = torch.as_tensor(e1)
        if not isinstance(e2, torch.Tensor):
            e2 = torch.as_tensor(e2)
        return e1 @ e2.T

    def similarity_pairwise(self, e1, e2):
        import torch

        if not isinstance(e1, torch.Tensor):
            e1 = torch.as_tensor(e1)
        if not isinstance(e2, torch.Tensor):
            e2 = torch.as_tensor(e2)
        return (e1 * e2).sum(dim=1)


def evaluate_with_mteb(
    model,
    tasks: List,
    model_name: str,
    model_type: str,
    dataset_type: str,
    generic_instruction: str,
    output_dir: str,
    batch_size: int = 32,
    save_predictions: bool = False,
) -> List[Dict]:
    """Run MTEB evaluation for ruMTEB/en_mteb built-in tasks.

    For CausalLM models on plain retrieval tasks (rumteb, en_mteb), wraps
    the model to inject the generic instruction into queries.

    InstructionRetrieval / InstructionReranking tasks (e.g. FollowIR) are
    excluded from wrapping because MTEB already concatenates the instruction
    field into the query text.
    """
    eval_model = model
    # InstructionRetrieval tasks already embed instructions in query text —
    # wrapping would double-inject (and in the wrong language for cross-lingual).
    instruction_flags = [
        getattr(t.metadata, "type", "")
        in ("InstructionRetrieval", "InstructionReranking")
        for t in tasks
    ]
    if any(instruction_flags) and not all(instruction_flags):
        raise ValueError(
            "Do not mix instruction and plain retrieval tasks in one dataset config"
        )
    is_instruction_task = bool(instruction_flags and all(instruction_flags))
    if (
        model_type == "causal_lm"
        and dataset_type in ("rumteb", "en_mteb")
        and not is_instruction_task
        and generic_instruction.strip()
    ):
        logger.info(
            f'Wrapping CausalLM with generic instruction: "{generic_instruction}"'
        )
        eval_model = CausalLMRetrieverWithInstruction(model)
    elif model_type == "causal_lm" and not is_instruction_task:
        logger.info("Generic instruction is empty; using plain queries unchanged.")
    elif is_instruction_task:
        logger.info(
            "Skipping generic instruction wrapper — InstructionRetrieval tasks "
            "already have instructions embedded in queries by MTEB."
        )

    evaluation = mteb.MTEB(tasks=tasks)
    # Custom encoder wrappers do not ship MTEB metadata.  Ordinary evaluation
    # tolerates that, but MTEB's prediction writer dereferences
    # ``model.mteb_model_meta``.  Attach the same fallback metadata that the
    # deprecated evaluator creates internally so raw rankings can be saved.
    if getattr(eval_model, "mteb_model_meta", None) is None:
        model_meta = evaluation.create_model_meta(eval_model)
        model_meta.name = model_name
        eval_model.mteb_model_meta = model_meta
    prediction_folder = None
    if save_predictions:
        prediction_folder = os.path.join(output_dir, "predictions", model_name)
        os.makedirs(prediction_folder, exist_ok=True)
        logger.info("Saving MTEB predictions to %s", prediction_folder)
    results = evaluation.run(
        eval_model,
        output_folder=os.path.join(output_dir, model_name),
        batch_size=batch_size,
        prediction_folder=prediction_folder,
        overwrite_results=True,
    )
    return results



# ---------------------------------------------------------------------------
# p-MRR evaluation
# ---------------------------------------------------------------------------


def evaluate_pmrr_synthetic(
    task,
    all_results: Dict[str, Dict[str, float]],
) -> float:
    """Compute p-MRR from pre-computed retrieval results.

    The versioned synthetic task does not expose qrel_diff, so changed
    documents are derived from its paired qrels.
    """
    pairs = task.get_query_pairs()
    if not pairs:
        logger.warning("No query pairs found for p-MRR computation.")
        return 0.0

    # Derive qrel_diff from the paired synthetic qrels.
    # For synthetic test, changed docs = instruction negatives
    # (docs relevant to standard query but not to instructed query)
    split = "test"
    relevant_docs = task.relevant_docs[split]

    # Build a synthetic qrel_diff from qrels
    qrel_diff = {}
    og_results = {}
    changed_results = {}

    for std_qid, inst_qid in pairs:
        if std_qid in all_results:
            og_results[f"{std_qid}-og"] = all_results[std_qid]
        if inst_qid in all_results:
            changed_results[f"{std_qid}-changed"] = all_results[inst_qid]

        std_relevant = {
            k for k, v in relevant_docs.get(std_qid, {}).items() if v > 0
        }
        inst_relevant = {
            k for k, v in relevant_docs.get(inst_qid, {}).items() if v > 0
        }
        should_decrease = std_relevant - inst_relevant
        if should_decrease:
            qrel_diff[std_qid] = list(should_decrease)

    return compute_pmrr(
        results_original=og_results,
        results_changed=changed_results,
        qrel_diff=qrel_diff,
    )


# ---------------------------------------------------------------------------
# Dense retrieval helper
# ---------------------------------------------------------------------------


def _dense_retrieve(
    model,
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    batch_size: int = 32,
    top_k: int = 1000,
    query_prompt_name: Optional[str] = "query",
) -> Dict[str, Dict[str, float]]:
    """Retrieve using dense model: encode queries and corpus, compute cosine.

    Args:
        query_prompt_name: Passed as prompt_name when encoding queries. Set to
            None for custom instruction-following datasets such as synthetic_test
            where task-specific prompts are already embedded in the query text.
    """
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    doc_ids = list(corpus.keys())
    doc_texts = []
    for did in doc_ids:
        doc = corpus[did]
        title = doc.get("title", "")
        text = doc.get("text", "")
        doc_texts.append(f"{title}. {text}" if title else text)

    logger.info(f"Encoding {len(query_texts)} queries (prompt_name={query_prompt_name!r})...")
    q_embs = model.encode(query_texts, batch_size=batch_size, prompt_name=query_prompt_name)

    logger.info(f"Encoding {len(doc_texts)} documents...")
    d_embs = model.encode(doc_texts, batch_size=batch_size, prompt_name="passage")

    # Cosine similarity (embeddings are L2-normalized)
    logger.info("Computing similarities...")
    scores = np.dot(q_embs, d_embs.T)

    results = {}
    for i, qid in enumerate(query_ids):
        top_indices = np.argsort(scores[i])[::-1][:top_k]
        results[qid] = {doc_ids[idx]: float(scores[i, idx]) for idx in top_indices}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation pipeline for ru-promptriever benchmarking."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_qwen3-4b.yaml",
        help="Path to evaluation config file.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=None,
        help="Filter: run only these models (by name). Default: all.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Filter: run only these datasets (by name). Default: all.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries for smoke testing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip model-dataset pairs that already have results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. cpu or cuda:0)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="Hugging Face Dataset repo ID for automatic intermediate uploads (e.g. 'Vladimirlv/my-results').",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not print intermediate or cumulative result tables.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    output_dir = config.get("output_dir", "./results")
    generic_instruction = config.get(
        "generic_instruction", "Найди релевантный документ."
    )
    retrieval_top_k = config.get("retrieval_top_k", 1000)

    requested_device = str(config.get("device", "cuda:0"))
    if requested_device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Config requests {requested_device}, but PyTorch cannot access CUDA. "
                "Fix the PyTorch/CUDA installation before starting evaluation."
            )

    # Filter models
    models_cfg = config.get("models", [])
    if args.models:
        available_models = {m["name"] for m in models_cfg}
        unknown_models = sorted(set(args.models) - available_models)
        if unknown_models:
            parser.error(
                f"Unknown model(s): {', '.join(unknown_models)}. "
                f"Available: {', '.join(sorted(available_models))}"
            )
        models_cfg = [m for m in models_cfg if m["name"] in args.models]

    # Filter datasets
    datasets_cfg = config.get("datasets", [])
    if args.datasets:
        available_datasets = {d["name"] for d in datasets_cfg}
        unknown_datasets = sorted(set(args.datasets) - available_datasets)
        if unknown_datasets:
            parser.error(
                f"Unknown dataset(s): {', '.join(unknown_datasets)}. "
                f"Available: {', '.join(sorted(available_datasets))}"
            )
        datasets_cfg = [d for d in datasets_cfg if d["name"] in args.datasets]

    repository_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code_revision = git_revision(repository_root)
    manifest_path = write_run_manifest(
        output_dir, args.config, config, code_revision
    )
    logger.info("Saved run manifest: %s", manifest_path)

    logger.info(f"Models to evaluate: {[m['name'] for m in models_cfg]}")
    logger.info(f"Datasets to evaluate: {[d['name'] for d in datasets_cfg]}")

    for model_cfg in models_cfg:
        model_name = model_cfg["name"]
        model_type = model_cfg["type"]
        batch_size = model_cfg.get("batch_size", 32)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Loading model: {model_name}")
        logger.info(f"{'=' * 60}")

        try:
            model = load_model(model_cfg, config)
        except Exception as e:
            logger.error(f"✗ Failed to load model {model_name}: {e}", exc_info=True)
            continue

        for dataset_cfg in datasets_cfg:
            dataset_name = dataset_cfg["name"]
            dataset_type = dataset_cfg["type"]
            pair_protocol = protocol_payload(
                config, model_cfg, dataset_cfg, code_revision
            )
            pair_fingerprint = protocol_fingerprint(pair_protocol)

            # --- Skip existing ---
            if args.skip_existing:
                existing = find_matching_result(output_dir, pair_fingerprint)
                if existing is not None:
                    logger.info(
                        f"Skipping {model_name} on {dataset_name} "
                        f"(matching protocol result: {existing})"
                    )
                    continue

            logger.info(f"\n--- Evaluating {model_name} on {dataset_name} ---")

            try:
                tasks = load_tasks(dataset_cfg)
                all_metrics = {}

                if dataset_type in ("rumteb", "en_mteb", "mfollowir"):
                    # Use MTEB for built-in retrieval and instruction-reranking
                    # tasks.  In particular, mFollowIR must retain its official
                    # per-query candidate lists.
                    if model_type == "bm25":
                        if dataset_type == "mfollowir":
                            raise ValueError(
                                "The local BM25 wrapper does not implement official "
                                "mFollowIR candidate reranking"
                            )
                        for task in tasks:
                            task_metrics = evaluate_bm25(
                                model, task, top_k=retrieval_top_k, max_queries=args.max_queries
                            )
                            t_name = (
                                task.metadata.name
                                if hasattr(task, "metadata")
                                and hasattr(task.metadata, "name")
                                else dataset_name
                            )
                            all_metrics[t_name] = task_metrics
                    else:
                        results = evaluate_with_mteb(
                            model,
                            tasks,
                            model_name,
                            model_type,
                            dataset_type,
                            generic_instruction,
                            output_dir,
                            batch_size,
                            save_predictions=dataset_cfg.get("save_predictions", False),
                        )
                        if results:
                            all_metrics["mteb"] = results

                else:
                    # Versioned synthetic-test custom path.
                    for task in tasks:
                        if model_type == "bm25":
                            bm25_metrics = evaluate_bm25(
                                model, task, top_k=retrieval_top_k, max_queries=args.max_queries
                            )
                            all_metrics["retrieval"] = bm25_metrics

                            # p-MRR for BM25: retrieve all queries using already-indexed corpus
                            if isinstance(task, RuPrompTrieverTestRetrieval):
                                task.load_data()
                                queries = task.queries["test"]
                                bm25_results = model.retrieve(queries, top_k=retrieval_top_k)
                                pmrr = evaluate_pmrr_synthetic(task, bm25_results)
                                all_metrics["p_mrr"] = pmrr
                                logger.info(f"p-MRR: {pmrr * 100:.2f} (raw: {pmrr:.4f})")
                        else:
                            retrieval_results, metrics = evaluate_dense_custom(
                                model,
                                task,
                                model_type,
                                dataset_type,
                                generic_instruction,
                                batch_size,
                                top_k=retrieval_top_k,
                                max_queries=args.max_queries,
                            )
                            all_metrics["retrieval"] = metrics

                            # p-MRR for dense models
                            if isinstance(task, RuPrompTrieverTestRetrieval):
                                pmrr = evaluate_pmrr_synthetic(task, retrieval_results)
                                all_metrics["p_mrr"] = pmrr
                                logger.info(f"p-MRR: {pmrr * 100:.2f} (raw: {pmrr:.4f})")

                save_results(
                    all_metrics,
                    model_name,
                    dataset_name,
                    output_dir,
                    protocol_fingerprint=pair_fingerprint,
                    protocol=pair_protocol,
                )
                if not args.no_summary:
                    print_intermediate_result(model_name, dataset_name, all_metrics)
                logger.info(f"✓ {model_name} on {dataset_name} completed.")

                # Upload intermediate results
                if args.hf_repo:
                    upload_to_huggingface(output_dir, args.hf_repo)

            except Exception as e:
                logger.error(
                    f"✗ Error evaluating {model_name} on {dataset_name}: {e}",
                    exc_info=True,
                )

        # Free GPU memory between models
        if model_type != "bm25":
            import torch

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not args.no_summary:
        print_summary_table(output_dir)


if __name__ == "__main__":
    main()
