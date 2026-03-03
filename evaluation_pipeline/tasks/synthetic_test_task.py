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
        category="t2t",
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
    )

    def load_data(self, **kwargs):
        """Load corpus from test_corpus.parquet and queries/relevance from test.parquet."""
        if self.data_loaded:
            return

        dataset_path = self.metadata.dataset["path"]

        # --- Load full retrieval corpus from test_corpus.parquet ---
        logger.info("Loading ru-promptriever corpus (test_corpus.parquet)...")
        corpus_ds = hf_datasets.load_dataset(
            dataset_path,
            split="test",
            data_files={"test": "test_corpus.parquet"},
        )
        corpus = {}
        for row in corpus_ds:
            doc_id = str(row["docid"])
            corpus[doc_id] = {
                "text": row.get("text", ""),
                "title": row.get("title", ""),
            }
        logger.info(f"Corpus loaded: {len(corpus)} documents")

        # --- Load queries and relevance judgments from test.parquet ---
        logger.info("Loading ru-promptriever test queries (test.parquet)...")
        query_ds = hf_datasets.load_dataset(
            dataset_path,
            split="test",
            data_files={"test": "test.parquet"},
        )
        queries = {}
        relevant_docs = defaultdict(dict)

        for row in query_ds:
            q_id = str(row["query_id"])
            only_query = row["only_query"]
            only_instruction = row.get("only_instruction", "")
            has_instruction = row.get("has_instruction", False)

            if has_instruction and only_instruction:
                # Instructed query: "{query} {instruction}"
                inst_q_id = f"{q_id}-instruct"
                queries[inst_q_id] = f"{only_query} {only_instruction}"
                for passage in row.get("positive_passages", []) or []:
                    relevant_docs[inst_q_id][str(passage["docid"])] = 1
            else:
                # Standard query (no instruction)
                queries[q_id] = only_query
                for passage in row.get("positive_passages", []) or []:
                    relevant_docs[q_id][str(passage["docid"])] = 1

        logger.info(
            f"Loaded test queries: {len(queries)} queries, "
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
