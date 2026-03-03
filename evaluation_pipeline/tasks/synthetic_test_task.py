"""
Custom MTEB Task for the test split of the ru-promptriever-dataset.

Dataset: Vladimirlv/ru-promptriever-dataset-v0.1 (test.parquet)

The test split contains query pairs:
  - Standard queries (query_id → only_query)
  - Instructed queries (query_id-instruct → "{only_query} {only_instruction}")

The corpus is a global pool of all documents from the test split:
  positive_passages, negative_passages (including instruction negatives).

Metrics: nDCG@20, p-MRR.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict

import datasets as hf_datasets
from mteb import TaskMetadata
from mteb.abstasks.retrieval import AbsTaskRetrieval

logger = logging.getLogger(__name__)


class RuPrompTrieverTestRetrieval(AbsTaskRetrieval):
    """Custom MTEB task for the ru-promptriever synthetic test set."""

    metadata = TaskMetadata(
        name="RuPrompTrieverTestRetrieval",
        description=(
            "Retrieval evaluation on the test split of the Russian "
            "Promptriever synthetic dataset. Contains standard and "
            "instruction-augmented query pairs."
        ),
        dataset={
            "path": "Vladimirlv/ru-promptriever-dataset-v0.1",
            "revision": "main",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_20",
        reference=None,
        date=("2024-01-01", "2025-12-31"),
        domains=["Encyclopaedic"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="LM-generated",
        dialect=[],
        sample_creation="LM-generated and verified",
        bibtex_citation="",
        descriptive_stats={},
    )

    def load_data(self, **kwargs):
        """Load test.parquet and build corpus/queries/relevant_docs."""
        if self.data_loaded:
            return

        logger.info("Loading ru-promptriever test split...")
        ds = hf_datasets.load_dataset(
            "Vladimirlv/ru-promptriever-dataset-v0.1",
            split="test",
        )

        corpus = {}  # {doc_id: {"text": ..., "title": ...}}
        queries = {}  # {query_id: query_text}
        relevant_docs = defaultdict(dict)  # {query_id: {doc_id: score}}

        for row in ds:
            q_id = str(row["query_id"])
            only_query = row["only_query"]
            only_instruction = row.get("only_instruction", "")
            has_instruction = row.get("has_instruction", False)

            # --- Build corpus from ALL passages ---
            for passage in row.get("positive_passages", []) or []:
                doc_id = str(passage["docid"])
                if doc_id not in corpus:
                    title = passage.get("title", "")
                    text = passage.get("text", "")
                    corpus[doc_id] = {"text": text, "title": title}

            for passage in row.get("negative_passages", []) or []:
                doc_id = str(passage["docid"])
                if doc_id not in corpus:
                    title = passage.get("title", "")
                    text = passage.get("text", "")
                    corpus[doc_id] = {"text": text, "title": title}

            # Also check new_negatives if present
            for passage in row.get("new_negatives", []) or []:
                doc_id = str(passage["docid"])
                if doc_id not in corpus:
                    title = passage.get("title", "")
                    text = passage.get("text", "")
                    corpus[doc_id] = {"text": text, "title": title}

            # --- Build queries ---
            if has_instruction and only_instruction:
                # Instructed query: "{query} {instruction}"
                inst_q_id = f"{q_id}-instruct"
                queries[inst_q_id] = f"{only_query} {only_instruction}"

                # Relevant docs for instructed query
                for passage in row.get("positive_passages", []) or []:
                    relevant_docs[inst_q_id][str(passage["docid"])] = 1
            else:
                # Standard query (no instruction)
                queries[q_id] = only_query

                # Relevant docs for standard query
                for passage in row.get("positive_passages", []) or []:
                    relevant_docs[q_id][str(passage["docid"])] = 1

        logger.info(
            f"Loaded test split: {len(corpus)} docs, "
            f"{len(queries)} queries, "
            f"{sum(len(v) for v in relevant_docs.values())} relevance judgments"
        )

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": dict(relevant_docs)}
        self.data_loaded = True

    def get_query_pairs(self) -> list:
        """Return (standard_query_id, instructed_query_id) pairs for p-MRR.

        In the test split, pairs are linked by the base query_id:
        - Standard: query_id (without "-instruct" suffix, has_instruction=False)
        - Instructed: query_id (with has_instruction=True)
        """
        queries = self.queries.get("test", {})
        standard = set()
        instructed = set()

        for qid in queries:
            if qid.endswith("-instruct"):
                instructed.add(qid)
                standard.add(qid.replace("-instruct", ""))
            else:
                standard.add(qid)

        pairs = []
        for std_qid in standard:
            inst_qid = f"{std_qid}-instruct"
            if inst_qid in instructed:
                pairs.append((std_qid, inst_qid))

        return pairs
