"""Paired uncertainty tests for per-topic mFollowIR metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = {
    "ndcg_cut_20": {"label": "nDCG@20", "scale": 1.0},
    "p_mrr": {"label": "p-MRR", "scale": 100.0},
}


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "per_query" not in payload:
        raise ValueError(f"{path} has no per_query object")
    return payload


def paired_values(a: dict, b: dict, metric: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(
        qid
        for qid in set(a["per_query"]) & set(b["per_query"])
        if metric in a["per_query"][qid] and metric in b["per_query"][qid]
    )
    if not ids:
        raise ValueError(f"No paired topics for {metric}")
    av = np.asarray([a["per_query"][qid][metric] for qid in ids], dtype=np.float64)
    bv = np.asarray([b["per_query"][qid][metric] for qid in ids], dtype=np.float64)
    return ids, av, bv


def paired_bootstrap(
    differences: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> tuple[float, float]:
    n = differences.size
    means = np.empty(repetitions, dtype=np.float64)
    chunk = 10_000
    for start in range(0, repetitions, chunk):
        stop = min(start + chunk, repetitions)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_randomization(
    differences: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> float:
    observed = abs(float(differences.mean()))
    extreme = 0
    chunk = 10_000
    for start in range(0, repetitions, chunk):
        size = min(chunk, repetitions - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, differences.size))
        randomized = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(randomized >= observed - 1e-15))
    return float((extreme + 1) / (repetitions + 1))


def randomization_greater_than_zero(
    values: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
) -> float:
    """One-sided sign-flip test for a positive mean."""
    observed = float(values.mean())
    extreme = 0
    chunk = 10_000
    for start in range(0, repetitions, chunk):
        size = min(chunk, repetitions - start)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(size, values.size))
        randomized = (signs * values).mean(axis=1)
        extreme += int(np.count_nonzero(randomized >= observed - 1e-15))
    return float((extreme + 1) / (repetitions + 1))


def analyze_against_zero(
    payload: dict,
    metric: str,
    rng: np.random.Generator,
    bootstrap_repetitions: int,
    randomization_repetitions: int,
) -> dict:
    ids = sorted(
        qid for qid, row in payload["per_query"].items() if metric in row
    )
    values = np.asarray(
        [payload["per_query"][qid][metric] for qid in ids], dtype=np.float64
    )
    ci_low, ci_high = paired_bootstrap(values, rng, bootstrap_repetitions)
    p_two_sided = paired_randomization(values, rng, randomization_repetitions)
    p_greater = randomization_greater_than_zero(
        values, rng, randomization_repetitions
    )
    scale = METRICS[metric]["scale"]
    return {
        "metric": metric,
        "label": METRICS[metric]["label"],
        "n_topics": len(ids),
        "mean": float(values.mean() * scale),
        "bootstrap_95_ci": [float(ci_low * scale), float(ci_high * scale)],
        "randomization_p_two_sided": p_two_sided,
        "randomization_p_greater_than_zero": p_greater,
        "topic_ids": ids,
    }


def analyze_metric(
    a: dict,
    b: dict,
    metric: str,
    rng: np.random.Generator,
    bootstrap_repetitions: int,
    randomization_repetitions: int,
) -> dict:
    ids, av, bv = paired_values(a, b, metric)
    differences = av - bv
    ci_low, ci_high = paired_bootstrap(differences, rng, bootstrap_repetitions)
    p_value = paired_randomization(differences, rng, randomization_repetitions)
    scale = METRICS[metric]["scale"]
    return {
        "metric": metric,
        "label": METRICS[metric]["label"],
        "n_topics": len(ids),
        "model_a_mean": float(av.mean() * scale),
        "model_b_mean": float(bv.mean() * scale),
        "difference_a_minus_b": float(differences.mean() * scale),
        "bootstrap_95_ci": [float(ci_low * scale), float(ci_high * scale)],
        "paired_randomization_p_two_sided": p_value,
        "topic_ids": ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=100_000)
    parser.add_argument("--randomization", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()

    a = load_payload(args.model_a)
    b = load_payload(args.model_b)
    rng = np.random.default_rng(args.seed)
    analyses = [
        analyze_metric(a, b, metric, rng, args.bootstrap, args.randomization)
        for metric in METRICS
    ]
    pmrr_against_zero = {
        "model_a": analyze_against_zero(
            a, "p_mrr", rng, args.bootstrap, args.randomization
        ),
        "model_b": analyze_against_zero(
            b, "p_mrr", rng, args.bootstrap, args.randomization
        ),
    }
    output = {
        "model_a": a.get("model", str(args.model_a)),
        "model_b": b.get("model", str(args.model_b)),
        "difference_direction": "model_a - model_b",
        "seed": args.seed,
        "bootstrap_repetitions": args.bootstrap,
        "randomization_repetitions": args.randomization,
        "metrics": analyses,
        "p_mrr_against_zero": pmrr_against_zero,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)

    print(f"A: {output['model_a']}")
    print(f"B: {output['model_b']}")
    for row in analyses:
        low, high = row["bootstrap_95_ci"]
        print(
            f"{row['label']}: n={row['n_topics']}, "
            f"A={row['model_a_mean']:.4f}, B={row['model_b_mean']:.4f}, "
            f"diff={row['difference_a_minus_b']:+.4f}, "
            f"95% CI [{low:+.4f}, {high:+.4f}], "
            f"p={row['paired_randomization_p_two_sided']:.6f}"
        )
    for model_key, row in pmrr_against_zero.items():
        low, high = row["bootstrap_95_ci"]
        print(
            f"{row['label']} {model_key} vs 0: n={row['n_topics']}, "
            f"mean={row['mean']:.4f}, 95% CI [{low:+.4f}, {high:+.4f}], "
            f"p(two-sided)={row['randomization_p_two_sided']:.6f}, "
            f"p(greater)={row['randomization_p_greater_than_zero']:.6f}"
        )


if __name__ == "__main__":
    main()
