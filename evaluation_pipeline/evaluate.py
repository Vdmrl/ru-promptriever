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
from typing import Dict, List

import mteb
import numpy as np

from models.bm25_retriever import BM25Retriever
from models.causal_retriever import CausalLMRetriever
from models.encoder_retriever import EncoderRetriever
from models.giga_embedding_retriever import GigaEmbeddingRetriever
from models.qwen3_embedding_retriever import Qwen3EmbeddingRetriever
from tasks.mfollowir_ru_task import MFollowIRRuRetrieval
from tasks.pmrr import compute_pmrr
from tasks.synthetic_test_task import RuPrompTrieverTestRetrieval
from utils.data_utils import load_config, print_summary_table, save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_cfg: dict, global_cfg: dict):
    """Instantiate a model wrapper based on config type."""
    model_type = model_cfg["type"]
    device = global_cfg.get("device", "cuda:0")
    dtype = global_cfg.get("dtype", "bfloat16")
    generic_instruction = global_cfg.get(
        "generic_instruction", "Найди релевантный документ."
    )

    if model_type == "bm25":
        return BM25Retriever()

    elif model_type == "encoder":
        return EncoderRetriever(
            model_name_or_path=model_cfg["model_name_or_path"],
            device=device,
            query_prefix=model_cfg.get("query_prefix", ""),
            passage_prefix=model_cfg.get("passage_prefix", ""),
            max_length=model_cfg.get("max_length", 512),
        )

    elif model_type == "causal_lm":
        return CausalLMRetriever(
            model_name_or_path=model_cfg["model_name_or_path"],
            device=device,
            dtype=dtype,
            max_length=model_cfg.get("max_length", 512),
            generic_instruction=generic_instruction,
        )

    elif model_type == "qwen3_embedding":
        return Qwen3EmbeddingRetriever(
            model_name_or_path=model_cfg["model_name_or_path"],
            device=device,
            max_length=model_cfg.get("max_length", 8192),
        )

    elif model_type == "giga_embedding":
        return GigaEmbeddingRetriever(
            model_name_or_path=model_cfg["model_name_or_path"],
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
        return [RuPrompTrieverTestRetrieval()]

    elif ds_type == "mfollowir":
        return [MFollowIRRuRetrieval()]

    elif ds_type == "rumteb":
        task_names = dataset_cfg.get("task_names", [])
        tasks = mteb.get_tasks(tasks=task_names)
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
      - synthetic_test / mfollowir: queries already contain instructions
        (baked into the query text). ALL models receive the same queries
        (this is the standard FollowIR protocol — p-MRR measures sensitivity).

      - rumteb: queries have NO instructions. For CausalLM (Promptriever)
        models, append the generic instruction to avoid OOD. For encoders
        and BM25, leave queries as-is.
    """
    if dataset_type == "rumteb" and model_type == "causal_lm":
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


def evaluate_bm25(
    model: BM25Retriever,
    task,
    top_k: int = 100,
    max_queries: int = None,
) -> Dict:
    """Run BM25 retrieval and compute metrics."""
    task.load_data()

    split = "test"
    corpus = task.corpus[split]
    queries = dict(task.queries[split])
    relevant_docs = task.relevant_docs[split]

    if max_queries:
        query_ids = list(queries.keys())[:max_queries]
        queries = {qid: queries[qid] for qid in query_ids}
        relevant_docs = {
            qid: relevant_docs[qid] for qid in query_ids if qid in relevant_docs
        }

    model.index_corpus(corpus)
    results = model.retrieve(queries, top_k=top_k)
    metrics = compute_retrieval_metrics(relevant_docs, results)
    return metrics


# ---------------------------------------------------------------------------
# Dense model evaluation (custom path for synthetic_test / mfollowir)
# ---------------------------------------------------------------------------


def evaluate_dense_custom(
    model,
    task,
    model_type: str,
    dataset_type: str,
    generic_instruction: str,
    batch_size: int = 32,
    top_k: int = 100,
    max_queries: int = None,
) -> Dict:
    """Evaluate a dense model on a custom task (synthetic_test, mfollowir).

    Handles query preparation and metric computation manually.
    """
    task.load_data()

    split = "test"
    corpus = task.corpus[split]
    queries = dict(task.queries[split])
    relevant_docs = task.relevant_docs[split]

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

    results = _dense_retrieve(model, queries, corpus, batch_size, top_k)
    metrics = compute_retrieval_metrics(relevant_docs, results)
    return results, metrics


# ---------------------------------------------------------------------------
# Dense model evaluation via MTEB (for ruMTEB built-in tasks)
# ---------------------------------------------------------------------------


class CausalLMRetrieverWithInstruction:
    """Wrapper that injects a generic instruction into queries for MTEB.

    MTEB calls model.encode(sentences, prompt_name="query") for queries.
    This wrapper intercepts "query" calls and appends the instruction.
    """

    def __init__(self, base_model: CausalLMRetriever):
        self.base_model = base_model
        self.generic_instruction = base_model.generic_instruction

    def encode(self, sentences, batch_size=32, prompt_name=None, **kwargs):
        if prompt_name == "query":
            sentences = [f"{s} {self.generic_instruction}" for s in sentences]
        return self.base_model.encode(
            sentences, batch_size=batch_size, prompt_name=prompt_name, **kwargs
        )


def evaluate_with_mteb(
    model,
    tasks: List,
    model_name: str,
    model_type: str,
    dataset_type: str,
    generic_instruction: str,
    output_dir: str,
    batch_size: int = 32,
) -> List[Dict]:
    """Run MTEB evaluation for ruMTEB built-in tasks.

    For CausalLM models on ruMTEB, wraps the model to inject the
    generic instruction into queries.
    """
    eval_model = model
    if model_type == "causal_lm" and dataset_type == "rumteb":
        logger.info(
            f'Wrapping CausalLM with generic instruction: "{generic_instruction}"'
        )
        eval_model = CausalLMRetrieverWithInstruction(model)

    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(
        eval_model,
        output_folder=os.path.join(output_dir, model_name),
        batch_size=batch_size,
    )
    return results


# ---------------------------------------------------------------------------
# p-MRR evaluation
# ---------------------------------------------------------------------------


def evaluate_pmrr_synthetic(
    task: RuPrompTrieverTestRetrieval,
    all_results: Dict[str, Dict[str, float]],
) -> float:
    """Compute p-MRR from pre-computed retrieval results.

    Uses query pairs from the synthetic test set.
    """
    pairs = task.get_query_pairs()
    if not pairs:
        logger.warning("No query pairs found for p-MRR computation.")
        return 0.0

    split = "test"
    relevant_docs = task.relevant_docs[split]

    std_results = {}
    inst_results = {}
    std_relevant = {}
    inst_relevant = {}

    for std_qid, inst_qid in pairs:
        if std_qid in all_results:
            std_results[std_qid] = all_results[std_qid]
        if inst_qid in all_results:
            inst_results[inst_qid] = all_results[inst_qid]
        if std_qid in relevant_docs:
            std_relevant[std_qid] = relevant_docs[std_qid]
        if inst_qid in relevant_docs:
            inst_relevant[inst_qid] = relevant_docs[inst_qid]

    return compute_pmrr(
        results_standard=std_results,
        results_instructed=inst_results,
        query_pairs=pairs,
        relevant_docs_standard=std_relevant,
        relevant_docs_instructed=inst_relevant,
    )


# ---------------------------------------------------------------------------
# Dense retrieval helper
# ---------------------------------------------------------------------------


def _dense_retrieve(
    model,
    queries: Dict[str, str],
    corpus: Dict[str, dict],
    batch_size: int = 32,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Retrieve using dense model: encode queries and corpus, compute cosine."""
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    doc_ids = list(corpus.keys())
    doc_texts = []
    for did in doc_ids:
        doc = corpus[did]
        title = doc.get("title", "")
        text = doc.get("text", "")
        doc_texts.append(f"{title}. {text}" if title else text)

    logger.info(f"Encoding {len(query_texts)} queries...")
    q_embs = model.encode(query_texts, batch_size=batch_size, prompt_name="query")

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

    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config.get("output_dir", "./results")
    generic_instruction = config.get(
        "generic_instruction", "Найди релевантный документ."
    )

    # Filter models
    models_cfg = config.get("models", [])
    if args.models:
        models_cfg = [m for m in models_cfg if m["name"] in args.models]

    # Filter datasets
    datasets_cfg = config.get("datasets", [])
    if args.datasets:
        datasets_cfg = [d for d in datasets_cfg if d["name"] in args.datasets]

    logger.info(f"Models to evaluate: {[m['name'] for m in models_cfg]}")
    logger.info(f"Datasets to evaluate: {[d['name'] for d in datasets_cfg]}")

    for model_cfg in models_cfg:
        model_name = model_cfg["name"]
        model_type = model_cfg["type"]
        batch_size = model_cfg.get("batch_size", 32)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Loading model: {model_name}")
        logger.info(f"{'=' * 60}")

        model = load_model(model_cfg, config)

        for dataset_cfg in datasets_cfg:
            dataset_name = dataset_cfg["name"]
            dataset_type = dataset_cfg["type"]

            # --- Skip existing ---
            if args.skip_existing:
                existing = (
                    [
                        f
                        for f in os.listdir(output_dir)
                        if f.startswith(f"{model_name}__{dataset_name}__")
                        and f.endswith(".json")
                    ]
                    if os.path.exists(output_dir)
                    else []
                )
                if existing:
                    logger.info(
                        f"Skipping {model_name} on {dataset_name} "
                        f"(found {len(existing)} existing result(s))"
                    )
                    continue

            logger.info(f"\n--- Evaluating {model_name} on {dataset_name} ---")

            try:
                tasks = load_tasks(dataset_cfg)
                all_metrics = {}

                if dataset_type == "rumteb":
                    # --- ruMTEB: use MTEB built-in evaluation ---
                    if model_type == "bm25":
                        for task in tasks:
                            task_metrics = evaluate_bm25(
                                model, task, max_queries=args.max_queries
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
                        )
                        if results:
                            all_metrics["mteb"] = results

                else:
                    # --- synthetic_test / mfollowir: custom path ---
                    for task in tasks:
                        if model_type == "bm25":
                            bm25_metrics = evaluate_bm25(
                                model, task, max_queries=args.max_queries
                            )
                            all_metrics["retrieval"] = bm25_metrics

                            # p-MRR for BM25
                            if isinstance(task, RuPrompTrieverTestRetrieval):
                                task.load_data()
                                queries = task.queries["test"]
                                corpus = task.corpus["test"]
                                model.index_corpus(corpus)
                                bm25_results = model.retrieve(queries, top_k=100)
                                pmrr = evaluate_pmrr_synthetic(task, bm25_results)
                                all_metrics["p_mrr"] = pmrr
                                logger.info(f"p-MRR: {pmrr:.2f}")
                        else:
                            retrieval_results, metrics = evaluate_dense_custom(
                                model,
                                task,
                                model_type,
                                dataset_type,
                                generic_instruction,
                                batch_size,
                                max_queries=args.max_queries,
                            )
                            all_metrics["retrieval"] = metrics

                            # p-MRR for dense models
                            if isinstance(task, RuPrompTrieverTestRetrieval):
                                pmrr = evaluate_pmrr_synthetic(task, retrieval_results)
                                all_metrics["p_mrr"] = pmrr
                                logger.info(f"p-MRR: {pmrr:.2f}")

                save_results(all_metrics, model_name, dataset_name, output_dir)
                logger.info(f"✓ {model_name} on {dataset_name} completed.")

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

    print_summary_table(output_dir)


if __name__ == "__main__":
    main()
