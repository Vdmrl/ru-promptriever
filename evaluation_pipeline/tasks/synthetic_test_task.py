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
            "path": "Vladimirlv/ru-promptriever-dataset",
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
        """Load corpus from test_corpus.parquet and queries/relevance from test.parquet.

        Uses hf_hub_download to fetch only the specific files needed,
        avoiding downloading the entire 11GB+ train split.

        Data structure from build_dataset.py:
          - Standard rows: query_id="12345", has_instruction=False,
            positive_passages=[rewritten_original_positive]
          - Instructed rows: query_id="12345-instruct", has_instruction=True,
            positive_passages=[final_positive],
            negative_passages=[instruction_negatives with explanation="instruction_negative"]

        We load them as-is (no extra suffix manipulation).
        """
        if self.data_loaded:
            return

        from huggingface_hub import hf_hub_download

        dataset_path = self.metadata.dataset["path"]

        # --- Download only the two files we need ---
        logger.info("Downloading test_corpus.parquet from HuggingFace...")
        corpus_path = hf_hub_download(
            repo_id=dataset_path,
            filename="test_corpus.parquet",
            repo_type="dataset",
        )

        logger.info("Downloading test.parquet from HuggingFace...")
        test_path = hf_hub_download(
            repo_id=dataset_path,
            filename="test.parquet",
            repo_type="dataset",
        )

        # --- Load corpus from test_corpus.parquet ---
        corpus_ds = hf_datasets.load_dataset(
            "parquet",
            data_files={"test": corpus_path},
            split="test",
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
        query_ds = hf_datasets.load_dataset(
            "parquet",
            data_files={"test": test_path},
            split="test",
        )
        queries = {}
        relevant_docs = defaultdict(dict)

        # Track instruction negatives per pair for p-MRR:
        # instruction_negatives[base_qid] = set of doc_ids that are relevant
        # to the standard query but NOT to the instructed query
        instruction_negatives_by_pair = defaultdict(set)

        for row in query_ds:
            q_id = str(row["query_id"])
            only_query = row["only_query"]
            only_instruction = row.get("only_instruction", "")
            has_instruction = row.get("has_instruction", False)

            if has_instruction and only_instruction:
                # Instructed row: query_id already ends with "-instruct"
                # Query text = "{query} {instruction}" (same as training format)
                queries[q_id] = f"{only_query} {only_instruction}"

                # Positive passages → relevance 1
                for passage in row.get("positive_passages", []) or []:
                    relevant_docs[q_id][str(passage["docid"])] = 1

                # Instruction negatives → relevance 0 for instructed query
                # These are docs relevant to the standard query but NOT to
                # the instructed query — they should decrease in rank.
                base_qid = q_id[: -len("-instruct")]
                for passage in row.get("negative_passages", []) or []:
                    doc_id = str(passage["docid"])
                    if doc_id not in relevant_docs[q_id]:
                        relevant_docs[q_id][doc_id] = 0
                    # Remember these for adding to standard query qrels
                    instruction_negatives_by_pair[base_qid].add(doc_id)
            else:
                # Standard row: plain query without instruction
                queries[q_id] = only_query
                for passage in row.get("positive_passages", []) or []:
                    relevant_docs[q_id][str(passage["docid"])] = 1

        # Second pass: add instruction negatives as RELEVANT (=1) for standard
        # queries. These docs ARE relevant to the standard query (they matched
        # the original topic) but are NOT relevant to the instructed query.
        # p-MRR needs: should_decrease = std_relevant - inst_relevant
        for base_qid, neg_doc_ids in instruction_negatives_by_pair.items():
            if base_qid in relevant_docs:
                for doc_id in neg_doc_ids:
                    if doc_id not in relevant_docs[base_qid]:
                        relevant_docs[base_qid][doc_id] = 1

        logger.info(
            f"Loaded test queries: {len(queries)} queries, "
            f"{sum(1 for v in relevant_docs.values() for s in v.values() if s > 0)} "
            f"positive judgments, "
            f"{sum(1 for v in relevant_docs.values() for s in v.values() if s == 0)} "
            f"instruction-negative judgments"
        )

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": dict(relevant_docs)}
        self.data_loaded = True

    def get_query_pairs(self) -> list:
        """Return (standard_query_id, instructed_query_id) pairs for p-MRR.

        In the test split, pairs are linked by the base query_id:
        - Standard: "12345" (has_instruction=False)
        - Instructed: "12345-instruct" (has_instruction=True, already in data)
        """
        queries = self.queries.get("test", {})
        pairs = []

        for qid in queries:
            if qid.endswith("-instruct"):
                std_qid = qid[: -len("-instruct")]
                if std_qid in queries:
                    pairs.append((std_qid, qid))

        return pairs
