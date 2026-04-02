"""
Custom MTEB Task for mFollowIR Russian split.

Dataset: jhu-clsp/mFollowIR (rus_map_final.jsonl)
Corpus: Official pooled mFollowIR corpus (hard negatives)

mFollowIR evaluates instruction-following in multilingual retrieval
using TREC NeuCLIR narratives as instructions.

Query construction (matching official MTEB methodology):
  - Original query  (id: "<qid>-og")      — topic + instruction_og
  - Changed query   (id: "<qid>-changed")  — topic + instruction_changed

p-MRR is computed on documents listed in qrel_diff (documents whose
relevance changed between instruction_og and instruction_changed).

Metrics: nDCG@20, p-MRR.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List

from mteb import TaskMetadata
from mteb.abstasks.retrieval import AbsTaskRetrieval

logger = logging.getLogger(__name__)

# mFollowIR Russian data path (downloaded to evaluation_pipeline/data/mfollowir/)
_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "mfollowir"
)


class MFollowIRRuRetrieval(AbsTaskRetrieval):
    """Custom MTEB task for mFollowIR Russian retrieval."""

    metadata = TaskMetadata(
        name="MFollowIRRuRetrieval",
        description=(
            "Russian split of mFollowIR — multilingual benchmark for "
            "instruction-following in retrieval. Based on TREC NeuCLIR "
            "2022 narratives."
        ),
        dataset={
            "path": "jhu-clsp/mFollowIR",
            "revision": "main",
        },
        type="Retrieval",
        category="t2t",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["rus-Cyrl"],
        main_score="ndcg_at_20",
        reference="https://arxiv.org/abs/2501.03516",
        date=("2022-01-01", "2025-12-31"),
        domains=["News"],
        task_subtypes=["Article retrieval"],
        license="cc-by-4.0",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def __init__(self, data_dir: str = _DEFAULT_DATA_DIR, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir

    def load_data(self, **kwargs):
        """Load mFollowIR Russian data and NeuCLIR corpus.

        Creates two sets of queries for p-MRR computation:
          - Original queries  (id: "<qid>-og")      — topic + instruction_og
          - Changed queries   (id: "<qid>-changed")  — topic + instruction_changed
        Both are stored in self.queries["test"] together.

        Qrels: Uses qrels_og (original relevance), assigned to BOTH
        query variants (since p-MRR measures rank change, not relevance).

        qrel_diff: Pre-computed list of documents whose relevance changed.
        """
        if self.data_loaded:
            return

        # --- Load queries from rus_map_final.jsonl ---
        jsonl_path = os.path.join(self.data_dir, "rus_map_final.jsonl")
        if not os.path.exists(jsonl_path):
            self._download_mfollowir()

        logger.info(f"Loading mFollowIR-RU queries from {jsonl_path}")
        queries = {}
        self._query_pairs = []  # (og_qid, changed_qid) for p-MRR

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                query_id = str(record["query_id"])
                topic = record.get("ht_text", "")
                instruction_og = record.get("instruction_og", "")
                instruction_changed = record.get("instruction_changed", "")

                # Original query (topic + original instruction)
                og_qid = f"{query_id}-og"
                queries[og_qid] = f"{topic} {instruction_og}" if instruction_og else topic

                # Changed query (topic + changed instruction)
                changed_qid = f"{query_id}-changed"
                queries[changed_qid] = (
                    f"{topic} {instruction_changed}" if instruction_changed else topic
                )

                self._query_pairs.append((og_qid, changed_qid))

        # --- Load qrels_og (original relevance judgments) ---
        # Same qrels are used for BOTH -og and -changed queries.
        # The p-MRR metric compares RANK changes, not relevance labels.
        qrels_original_path = os.path.join(self.data_dir, "qrels_original_test.jsonl")
        if not os.path.exists(qrels_original_path):
            self._download_qrels("qrels_og/test.jsonl", qrels_original_path)

        relevant_docs = defaultdict(dict)
        required_doc_ids = set()
        with open(qrels_original_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                q_id = str(record["query-id"])
                c_id = str(record["corpus-id"])
                score = int(record["score"])

                # Assign same qrels to both -og and -changed query IDs
                og_qid = f"{q_id}-og"
                changed_qid = f"{q_id}-changed"
                relevant_docs[og_qid][c_id] = score
                relevant_docs[changed_qid][c_id] = score
                required_doc_ids.add(c_id)

        # --- Load qrel_diff (documents whose relevance changed) ---
        self._qrel_diff = self._load_qrel_diff()

        # --- Load official pooled mFollowIR corpus ---
        corpus = self._load_pooled_corpus()

        logger.info(
            f"Loaded mFollowIR-RU: {len(corpus)} pooled docs, "
            f"{len(queries)} queries ({len(self._query_pairs)} pairs for p-MRR), "
            f"{sum(len(v) for v in relevant_docs.values())} judgments, "
            f"{len(self._qrel_diff)} qrel_diff entries"
        )

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": dict(relevant_docs)}
        self.data_loaded = True

    def get_query_pairs(self):
        """Return (og_query_id, changed_query_id) pairs for p-MRR computation."""
        return self._query_pairs

    def get_qrel_diff(self) -> Dict[str, List[str]]:
        """Return {bare_qid: [doc_id, ...]} mapping of changed documents."""
        return self._qrel_diff

    def _load_qrel_diff(self) -> Dict[str, List[str]]:
        """Load qrel_diff from the monolingual mFollowIR-parquet-mteb dataset.

        qrel_diff lists documents whose relevance changed between
        instruction_og and instruction_changed for each query.
        """
        import datasets

        logger.info("Loading qrel_diff-rus from jhu-clsp/mFollowIR-parquet-mteb...")
        try:
            ds = datasets.load_dataset(
                "jhu-clsp/mFollowIR-parquet-mteb",
                "qrel_diff-rus",
                split="qrel_diff",
            )
            qrel_diff = {
                str(item["query-id"]): item["corpus-ids"] for item in ds
            }
            logger.info(f"Loaded qrel_diff: {len(qrel_diff)} queries")
            return qrel_diff
        except Exception as e:
            logger.error(f"Failed to load qrel_diff: {e}")
            raise

    def _download_qrels(self, hf_filename, local_path):
        """Download a qrels file from mFollowIR-rus-cl on HuggingFace."""
        from huggingface_hub import hf_hub_download

        logger.info(f"Downloading {hf_filename} for mFollowIR...")
        try:
            hf_hub_download(
                repo_id="jhu-clsp/mFollowIR-rus-cl",
                filename=hf_filename,
                repo_type="dataset",
                local_dir=self.data_dir,
            )
            import shutil

            # Move from nested directory to flat path
            downloaded = os.path.join(self.data_dir, hf_filename)
            if os.path.exists(downloaded):
                shutil.move(downloaded, local_path)
        except Exception as e:
            logger.error(f"Failed to download {hf_filename}: {e}")
            raise

    def _download_mfollowir(self):
        """Download mFollowIR Russian data from HuggingFace."""
        from huggingface_hub import hf_hub_download

        os.makedirs(self.data_dir, exist_ok=True)
        logger.info("Downloading mFollowIR Russian data...")

        hf_hub_download(
            repo_id="jhu-clsp/mFollowIR",
            filename="rus_map_final.jsonl",
            repo_type="dataset",
            local_dir=self.data_dir,
        )
        logger.info(f"Downloaded to {self.data_dir}")

    def _load_pooled_corpus(self) -> Dict[str, dict]:
        """Load official pooled corpus directly via HuggingFace Hub.

        This avoids the need for massive truncation and correctly uses hard negatives.
        """
        from huggingface_hub import hf_hub_download

        corpus_path = os.path.join(self.data_dir, "corpus.jsonl")

        if not os.path.exists(corpus_path):
            logger.info("Downloading official pooled mFollowIR corpus (hard negatives)...")
            hf_hub_download(
                repo_id="jhu-clsp/mFollowIR-rus-cl",
                filename="corpus.jsonl",
                repo_type="dataset",
                local_dir=self.data_dir,
            )

        logger.info(f"Loading cached pooled corpus from {corpus_path}")
        corpus = {}
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = str(doc.get("_id", doc.get("doc_id", "")))
                corpus[doc_id] = {
                    "text": doc.get("text", ""),
                    "title": doc.get("title", ""),
                }
        return corpus
