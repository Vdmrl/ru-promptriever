"""Paired statistics for the official Russian mFollowIR reranking protocol.

The input is the raw prediction directory produced by MTEB 2.10.5 with
``save_predictions: true``.  Before computing anything, the script verifies
that each run contains exactly the official per-query candidate set.  This
prevents accidental analysis of the legacy full-corpus retrieval runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mteb
import numpy as np
import pytrec_eval


TASK_NAME = "mFollowIR"
SUBSET = "rus"
SPLIT = "test"
PROTOCOL = "mteb-2.10.5-mfollowir-rus-official-reranking"


def _plain(value: Any) -> dict:
    return dict(value.items()) if hasattr(value, "items") else dict(value)


def _prediction_file(root: Path) -> Path:
    if root.is_file():
        return root
    candidates = sorted(root.rglob(f"{TASK_NAME}_predictions.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {TASK_NAME}_predictions.json under {root}; "
            f"found {candidates}"
        )
    return candidates[0]


def _load_predictions(root: Path) -> dict[str, dict[str, float]]:
    path = _prediction_file(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    available = [key for key in payload if key != "mteb_model_meta"]
    selected = SUBSET if SUBSET in payload else available[0]
    if selected != SUBSET:
        raise ValueError(f"Expected Russian subset {SUBSET!r}, found {available}")
    split = payload[selected].get(SPLIT)
    if split is None:
        raise KeyError(f"Prediction file {path} has no {SPLIT!r} split")
    return {
        str(qid): {str(doc): float(score) for doc, score in _plain(docs).items()}
        for qid, docs in _plain(split).items()
    }


def _official_data():
    tasks = list(mteb.get_tasks(tasks=[TASK_NAME], languages=[SUBSET]))
    if len(tasks) != 1:
        raise RuntimeError(f"Expected one mFollowIR task, got {len(tasks)}")
    task = tasks[0]
    expected_revision = "09eecbe45c54b4a6dfb8e68e345cae77337768e2"
    actual_revision = task.metadata.dataset["revision"]
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"Unexpected mFollowIR revision {actual_revision}; expected {expected_revision}"
        )
    task.load_data()
    qrels = {
        str(qid): {str(doc): int(score) for doc, score in _plain(docs).items()}
        for qid, docs in _plain(task.relevant_docs[SUBSET][SPLIT]).items()
    }
    top_ranked = {
        str(qid): [str(doc) for doc in docs]
        for qid, docs in _plain(task.top_ranked[SUBSET][SPLIT]).items()
    }
    qrel_diff = {
        str(qid): [str(doc) for doc in docs]
        for qid, docs in _plain(task.qrels_diff[SUBSET][SPLIT]).items()
    }
    return qrels, top_ranked, qrel_diff, actual_revision


def _validate_candidate_sets(
    run: dict[str, dict[str, float]], top_ranked: dict[str, list[str]], label: str
) -> None:
    missing_queries = sorted(set(top_ranked) - set(run))
    extra_queries = sorted(set(run) - set(top_ranked))
    mismatched = []
    for qid in sorted(set(run) & set(top_ranked)):
        expected = set(top_ranked[qid])
        observed = set(run[qid])
        if observed != expected:
            mismatched.append(
                (qid, len(observed), len(expected), len(expected - observed))
            )
    if missing_queries or extra_queries or mismatched:
        raise ValueError(
            f"{label} is not an official mFollowIR candidate reranking run: "
            f"missing_queries={missing_queries[:3]}, extra_queries={extra_queries[:3]}, "
            f"candidate_mismatches={mismatched[:3]}"
        )


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


def _per_topic(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    qrel_diff: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    original_qrels = {
        qid: docs for qid, docs in qrels.items() if qid.endswith("-og") and qid in run
    }
    original_run = {qid: run[qid] for qid in original_qrels}
    ndcg = pytrec_eval.RelevanceEvaluator(
        original_qrels, {"ndcg_cut_20"}
    ).evaluate(original_run)

    rows: dict[str, dict[str, float]] = {}
    for original_qid, values in ndcg.items():
        rows.setdefault(original_qid.removesuffix("-og"), {})["ndcg_cut_20"] = float(
            values["ndcg_cut_20"]
        )

    for base, changed_docs in qrel_diff.items():
        old_id, new_id = f"{base}-og", f"{base}-changed"
        if old_id not in run or new_id not in run or base not in rows:
            continue
        values = [
            _rank_score(_rank(run[old_id], doc), _rank(run[new_id], doc))
            for doc in changed_docs
        ]
        if values:
            rows[base]["p_mrr"] = float(np.mean(values))
    # Keep all original-condition nDCG topics.  qrel_diff is available for 39
    # of the 40 Russian topics, so p-MRR legitimately has one fewer topic;
    # metric-specific pairing below selects the appropriate set independently.
    return rows


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n: int) -> list[float]:
    means = np.empty(n, dtype=float)
    for start in range(0, n, 10_000):
        stop = min(start + 10_000, n)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]


def _sign_flip(values: np.ndarray, rng: np.random.Generator, n: int) -> float:
    observed = abs(float(values.mean()))
    extreme = 0
    for start in range(0, n, 10_000):
        size = min(10_000, n - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(size, len(values)))
        randomized = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(randomized >= observed - 1e-15))
    return float((extreme + 1) / (n + 1))


def _metric_stats(
    per_a: dict,
    per_b: dict,
    metric: str,
    rng: np.random.Generator,
    repetitions: int,
) -> dict:
    ids = sorted(
        qid
        for qid in set(per_a) & set(per_b)
        if metric in per_a[qid] and metric in per_b[qid]
    )
    if not ids:
        raise ValueError(f"No paired topics for {metric}")
    a = np.asarray([per_a[qid][metric] for qid in ids], dtype=float)
    b = np.asarray([per_b[qid][metric] for qid in ids], dtype=float)
    diff = a - b
    scale = 100.0 if metric == "p_mrr" else 1.0
    return {
        "n_topics": len(ids),
        "model_a_mean": float(a.mean() * scale),
        "model_b_mean": float(b.mean() * scale),
        "difference_a_minus_b": float(diff.mean() * scale),
        "bootstrap_95_ci": [x * scale for x in _bootstrap_ci(diff, rng, repetitions)],
        "paired_sign_flip_p_two_sided": _sign_flip(diff, rng, repetitions),
        "topic_ids": ids,
    }


def _single_model_stats(
    per_topic: dict,
    metric: str,
    rng: np.random.Generator,
    repetitions: int,
) -> dict:
    ids = sorted(qid for qid, row in per_topic.items() if metric in row)
    values = np.asarray([per_topic[qid][metric] for qid in ids], dtype=float)
    scale = 100.0 if metric == "p_mrr" else 1.0
    return {
        "n_topics": len(ids),
        "mean": float(values.mean() * scale),
        "bootstrap_95_ci": [
            x * scale for x in _bootstrap_ci(values, rng, repetitions)
        ],
        "sign_flip_p_two_sided_vs_zero": _sign_flip(values, rng, repetitions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--label-a", default="model-a")
    parser.add_argument("--label-b", default="model-b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    qrels, candidates, qrel_diff, revision = _official_data()
    run_a = _load_predictions(args.predictions_a)
    run_b = _load_predictions(args.predictions_b)
    _validate_candidate_sets(run_a, candidates, args.label_a)
    _validate_candidate_sets(run_b, candidates, args.label_b)
    per_a = _per_topic(run_a, qrels, qrel_diff)
    per_b = _per_topic(run_b, qrels, qrel_diff)

    rng = np.random.default_rng(args.seed)
    output = {
        "protocol": PROTOCOL,
        "dataset_revision": revision,
        "model_a": args.label_a,
        "model_b": args.label_b,
        "difference_direction": "model_a - model_b",
        "seed": args.seed,
        "repetitions": args.repetitions,
        "metrics": {
            metric: _metric_stats(per_a, per_b, metric, rng, args.repetitions)
            for metric in ("ndcg_cut_20", "p_mrr")
        },
        "individual_intervals": {
            "model_a": {
                metric: _single_model_stats(
                    per_a, metric, rng, args.repetitions
                )
                for metric in ("ndcg_cut_20", "p_mrr")
            },
            "model_b": {
                metric: _single_model_stats(
                    per_b, metric, rng, args.repetitions
                )
                for metric in ("ndcg_cut_20", "p_mrr")
            },
        },
        "per_topic": {"model_a": per_a, "model_b": per_b},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in output.items() if k != "per_topic"}, indent=2))


if __name__ == "__main__":
    main()
