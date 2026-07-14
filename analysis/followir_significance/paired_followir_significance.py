"""Paired topic-level statistics for FollowIR.

The evaluation pipeline stores MTEB prediction files when ``save_predictions``
is enabled.  This script reconstructs the table-aligned retrieval metric and
the official FollowIR p-MRR for every topic, then compares two models with
paired bootstrap confidence intervals and a paired sign-flip test.

Example:
    python analysis/followir_significance/paired_followir_significance.py \
      --predictions-a evaluation_pipeline/results_followir_significance/predictions/ru-only-paper \
      --predictions-b evaluation_pipeline/results_followir_significance/predictions/promptriever-8b-paper \
      --label-a ru-only-paper --label-b promptriever-8b-paper \
      --output evaluation_pipeline/results_followir_significance/followir_significance.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytrec_eval
from datasets import load_dataset
import mteb


TASKS = (
    "Robust04InstructionRetrieval",
    "Core17InstructionRetrieval",
    "News21InstructionRetrieval",
)


def _as_plain_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "items"):
        return dict(value.items())
    return dict(value)


def _task_split(task):
    task.load_data()
    subsets = list(task.dataset.keys())
    subset = "default" if "default" in subsets else subsets[0]
    split_data = task.dataset[subset]["test"]
    qrels = _as_plain_dict(split_data["relevant_docs"])
    qrels = {
        str(qid): {str(doc): int(score) for doc, score in _as_plain_dict(docs).items()}
        for qid, docs in qrels.items()
    }
    return subset, qrels


def _prediction_file(root: Path, task_name: str) -> Path:
    if root.is_file():
        return root
    candidates = sorted(root.rglob(f"{task_name}_predictions.json"))
    if not candidates:
        raise FileNotFoundError(f"No predictions for {task_name} under {root}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple prediction files for {task_name} under {root}: {candidates}"
        )
    return candidates[0]


def _load_predictions(root: Path, task_name: str, subset: str) -> dict[str, dict[str, float]]:
    path = _prediction_file(root, task_name)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    available = [key for key in payload if key != "mteb_model_meta"]
    selected = subset if subset in payload else ("default" if "default" in payload else available[0])
    split = payload[selected].get("test")
    if split is None:
        raise KeyError(f"Prediction file {path} has no test split")
    return {
        str(qid): {str(doc): float(score) for doc, score in docs.items()}
        for qid, docs in split.items()
    }


def _rank(run: dict[str, float], doc_id: str) -> int:
    ordered = sorted(run.items(), key=lambda item: item[1], reverse=True)
    for index, (candidate, _) in enumerate(ordered, start=1):
        if candidate == doc_id:
            return index
    return len(ordered) + 1


def _rank_score(old_rank: int, new_rank: int) -> float:
    if old_rank >= new_rank:
        return ((1.0 / old_rank) / (1.0 / new_rank)) - 1.0
    return 1.0 - ((1.0 / new_rank) / (1.0 / old_rank))


def _qrel_diff(task) -> dict[str, list[str]]:
    ds = load_dataset(
        task.metadata.dataset["path"],
        "qrel_diff",
        split="qrel_diff",
        revision=task.metadata.dataset["revision"],
    )
    return {str(row["query-id"]): [str(x) for x in row["corpus-ids"]] for row in ds}


def _per_topic(task_name: str, task, run: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    _, qrels = _task_split(task)
    qdiff = _qrel_diff(task)
    metric_name = "map_cut_1000" if task_name != "News21InstructionRetrieval" else "ndcg_cut_5"
    changed_qrels = {qid: docs for qid, docs in qrels.items() if qid.endswith("-changed") and qid in run}
    changed_run = {qid: run[qid] for qid in changed_qrels}
    scores = pytrec_eval.RelevanceEvaluator(changed_qrels, {metric_name}).evaluate(changed_run)

    rows: dict[str, dict[str, float]] = {}
    for changed_qid, values in scores.items():
        base = changed_qid.removesuffix("-changed")
        rows.setdefault(base, {})["retrieval"] = float(values[metric_name])

    for base, changed_docs in qdiff.items():
        old_id, new_id = f"{base}-og", f"{base}-changed"
        if old_id not in run or new_id not in run or base not in rows:
            continue
        values = [_rank_score(_rank(run[old_id], doc), _rank(run[new_id], doc)) for doc in changed_docs]
        if values:
            rows[base]["pmrr"] = float(np.mean(values))

    return {qid: row for qid, row in rows.items() if "retrieval" in row and "pmrr" in row}


def _paired_stats(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, float | int | list[float]]:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = float(np.mean(diff))
    n = len(diff)
    boot = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, 5000):
        stop = min(start + 5000, n_boot)
        indices = rng.integers(0, n, size=(stop - start, n))
        boot[start:stop] = diff[indices].mean(axis=1)
    null = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, 5000):
        stop = min(start + 5000, n_boot)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(stop - start, n))
        null[start:stop] = (diff[None, :] * signs).mean(axis=1)
    p = (1.0 + float(np.sum(np.abs(null) >= abs(observed)))) / (n_boot + 1.0)
    return {
        "n_topics": int(n),
        "a_mean": float(np.mean(a)),
        "b_mean": float(np.mean(b)),
        "difference_a_minus_b": observed,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_value_sign_flip": p,
    }


def _macro_pmrr(task_pairs: dict[str, tuple[np.ndarray, np.ndarray]], rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    names = list(task_pairs)
    observed = float(np.mean([np.mean(a - b) for a, b in task_pairs.values()]))
    boot = np.empty(n_boot, dtype=float)
    null = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        task_means = []
        task_null = []
        for name in names:
            a, b = task_pairs[name]
            idx = rng.integers(0, len(a), size=len(a))
            d = a[idx] - b[idx]
            task_means.append(float(np.mean(d)))
            task_null.append(float(np.mean(d * rng.choice([-1.0, 1.0], size=len(d)))))
        boot[i] = np.mean(task_means)
        null[i] = np.mean(task_null)
    p = (1.0 + float(np.sum(np.abs(null) >= abs(observed)))) / (n_boot + 1.0)
    return {
        "n_tasks": len(names),
        "difference_a_minus_b": observed,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_value_sign_flip": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--label-a", default="model-a")
    parser.add_argument("--label-b", default="model-b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--n-bootstrap", type=int, default=100000)
    args = parser.parse_args()

    tasks = mteb.get_tasks(tasks=list(TASKS), languages=["eng"])
    rng = np.random.default_rng(args.seed)
    output: dict[str, Any] = {
        "model_a": args.label_a,
        "model_b": args.label_b,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "tasks": {},
    }
    macro_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for task in tasks:
        name = task.metadata.name
        subset, _ = _task_split(task)
        run_a = _load_predictions(args.predictions_a, name, subset)
        run_b = _load_predictions(args.predictions_b, name, subset)
        per_a = _per_topic(name, task, run_a)
        per_b = _per_topic(name, task, run_b)
        common = sorted(set(per_a) & set(per_b))
        retrieval_a = np.array([per_a[q]["retrieval"] for q in common])
        retrieval_b = np.array([per_b[q]["retrieval"] for q in common])
        pmrr_a = np.array([per_a[q]["pmrr"] for q in common])
        pmrr_b = np.array([per_b[q]["pmrr"] for q in common])
        output["tasks"][name] = {
            "retrieval_metric": "MAP@1000" if name != "News21InstructionRetrieval" else "nDCG@5",
            "n_common_topics": len(common),
            "retrieval": _paired_stats(retrieval_a, retrieval_b, rng, args.n_bootstrap),
            "pmrr": _paired_stats(pmrr_a, pmrr_b, rng, args.n_bootstrap),
        }
        macro_pairs[name] = (pmrr_a, pmrr_b)

    output["macro_pmrr"] = _macro_pmrr(macro_pairs, rng, args.n_bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
