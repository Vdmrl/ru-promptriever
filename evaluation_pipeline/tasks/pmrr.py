"""
p-MRR (Pairwise Mean Reciprocal Rank) metric implementation.

p-MRR measures how well a retrieval model follows instructions by comparing
the ranking of "changed" documents between original-instruction and
changed-instruction queries.

A positive p-MRR means the model correctly adjusts rankings based on
instructions. Range: [-100, +100] (when multiplied by 100).

This implementation mirrors the official MTEB `calculate_pmrr`:
  mteb._evaluators.retrieval_metrics.calculate_pmrr

Reference: Weller et al., 2024 — FollowIR / mFollowIR papers.
"""

from typing import Dict, List, Tuple

import pandas as pd


def compute_pmrr(
    results_original: Dict[str, Dict[str, float]],
    results_changed: Dict[str, Dict[str, float]],
    qrel_diff: Dict[str, List[str]],
) -> float:
    """Compute p-MRR between original-instruction and changed-instruction runs.

    Mirrors the official MTEB ``calculate_pmrr`` exactly.

    For each query in *qrel_diff*, we look at documents whose relevance
    **changed** between the original and changed instructions.  p-MRR
    measures whether these documents moved in rank as expected.

    Args:
        results_original: ``{qid-og: {doc_id: score}}`` — retrieval scores
            for queries with the **original** instruction.
        results_changed: ``{qid-changed: {doc_id: score}}`` — retrieval
            scores for queries with the **changed** instruction.
        qrel_diff: ``{qid: [doc_id, ...]}`` — mapping from bare query ID
            to the list of documents whose relevance changed.

    Returns:
        p-MRR score (raw, in [-1, +1]).  Multiply by 100 for display.
    """
    changes = []

    for qid, changed_docs in qrel_diff.items():
        og_key = f"{qid}-og"
        changed_key = f"{qid}-changed"

        if og_key not in results_original or changed_key not in results_changed:
            continue

        og_run = results_original[og_key]
        new_run = results_changed[changed_key]

        for doc_id in changed_docs:
            og_rank, og_score = _get_rank_from_dict(og_run, doc_id)
            new_rank, new_score = _get_rank_from_dict(new_run, doc_id)

            changes.append(
                {
                    "qid": qid,
                    "doc_id": doc_id,
                    "og_rank": og_rank,
                    "new_rank": new_rank,
                    "og_score": og_score,
                    "new_score": new_score,
                }
            )

    if not changes:
        return 0.0

    # Compute rank_score for each changed document
    df = pd.DataFrame(changes)
    df["p-MRR"] = df.apply(_rank_score, axis=1)

    # Average per query, then macro-average across queries
    qid_wise = df.groupby("qid").agg({"p-MRR": "mean"})
    return float(qid_wise["p-MRR"].mean())


def _rank_score(x) -> float:
    """Pairwise rank score — identical to MTEB ``rank_score``.

    If og_rank >= new_rank (doc moved up or stayed — bad for should-decrease docs):
        score = (1/og_rank) / (1/new_rank) - 1    → negative
    Else (doc moved down — good):
        score = 1 - (1/new_rank) / (1/og_rank)    → positive
    """
    if x["og_rank"] >= x["new_rank"]:
        return ((1 / x["og_rank"]) / (1 / x["new_rank"])) - 1
    else:
        return 1 - ((1 / x["new_rank"]) / (1 / x["og_rank"]))


def _get_rank_from_dict(
    doc_scores: Dict[str, float], doc_id: str
) -> Tuple[int, float]:
    """Find the 1-indexed rank and score of *doc_id* in a results dict.

    Identical to MTEB ``get_rank_from_dict``.  If the document is not
    found, returns ``(len(results) + 1, 0)``.
    """
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    for i, (did, score) in enumerate(sorted_docs):
        if did == doc_id:
            return i + 1, score
    return len(sorted_docs) + 1, 0.0
