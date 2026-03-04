"""
p-MRR (Pairwise Mean Reciprocal Rank) metric implementation.

p-MRR measures how well a retrieval model follows instructions by comparing
the ranking of documents between standard queries and instructed queries.

A positive p-MRR means the model correctly adjusts rankings based on
instructions. Range: [-100, +100].

Reference: Weller et al., 2024 — FollowIR / Promptriever papers.
"""

from typing import Dict, List, Tuple

import numpy as np


def compute_pmrr(
    results_standard: Dict[str, Dict[str, float]],
    results_instructed: Dict[str, Dict[str, float]],
    query_pairs: List[Tuple[str, str]],
    relevant_docs_standard: Dict[str, Dict[str, int]],
    relevant_docs_instructed: Dict[str, Dict[str, int]],
) -> float:
    """Compute p-MRR between standard and instructed query pairs.

    For each (standard_query, instructed_query) pair, we look at documents
    that are relevant to the standard query but NOT relevant to the
    instructed query (instruction negatives). p-MRR measures whether these
    documents decrease in rank when the instruction is added.

    Args:
        results_standard: {std_query_id: {doc_id: score}} — retrieval scores
            for standard queries (without instructions).
        results_instructed: {inst_query_id: {doc_id: score}} — retrieval
            scores for instructed queries.
        query_pairs: List of (std_query_id, inst_query_id) pairs.
        relevant_docs_standard: {std_query_id: {doc_id: relevance_score}}
        relevant_docs_instructed: {inst_query_id: {doc_id: relevance_score}}

    Returns:
        p-MRR score in [-100, +100].
    """
    query_pmrr_scores = []

    for std_qid, inst_qid in query_pairs:
        std_results = results_standard.get(std_qid, {})
        inst_results = results_instructed.get(inst_qid, {})

        std_relevant = {
            k for k, v in relevant_docs_standard.get(std_qid, {}).items() if v > 0
        }
        inst_relevant = {
            k for k, v in relevant_docs_instructed.get(inst_qid, {}).items() if v > 0
        }

        # Documents that should decrease in rank after adding instruction:
        # relevant to standard query, but NOT relevant to instructed query
        should_decrease = std_relevant - inst_relevant

        if not should_decrease:
            continue

        # Rank documents by score (descending)
        std_ranking = _rank_documents(std_results)
        inst_ranking = _rank_documents(inst_results)

        query_scores = []

        for doc_id in should_decrease:
            std_rank = std_ranking.get(doc_id)
            inst_rank = inst_ranking.get(doc_id)

            if std_rank is None and inst_rank is None:
                continue

            # If document is not in results, treat as max rank
            max_rank = max(len(std_ranking), len(inst_ranking)) + 1
            std_rank = std_rank if std_rank is not None else max_rank
            inst_rank = inst_rank if inst_rank is not None else max_rank

            std_rr = 1.0 / std_rank
            inst_rr = 1.0 / inst_rank

            # FollowIR Paper Equation 1:
            if std_rank > inst_rank:
                # Rank worsened (moved up to top, e.g. 10 -> 1) -> Negative reward
                # Ratio: (MRRog / MRRnew) - 1 => (std_rr / inst_rr) - 1.0
                score = (std_rr / inst_rr) - 1.0
            else:
                # Rank improved (moved down, e.g. 1 -> 10) -> Positive reward
                # Ratio: 1 - (MRRnew / MRRog) => 1.0 - (inst_rr / std_rr)
                score = 1.0 - (inst_rr / std_rr)

            query_scores.append(score)

        if query_scores:
            query_pmrr_scores.append(float(np.mean(query_scores)))

    if not query_pmrr_scores:
        return 0.0

    # Macro-average across all queries, scale to [-100, +100]
    return float(np.mean(query_pmrr_scores) * 100)


def _rank_documents(doc_scores: Dict[str, float]) -> Dict[str, int]:
    """Convert {doc_id: score} to {doc_id: rank} (1-indexed, descending)."""
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_docs)}
