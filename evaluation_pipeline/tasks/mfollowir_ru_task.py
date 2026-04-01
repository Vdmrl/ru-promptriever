"""
Custom MTEB Task for mFollowIR Russian split.

Dataset: jhu-clsp/mFollowIR (rus_map_final.jsonl)
Corpus: NeuCLIR-2022 Russian documents (loaded via ir_datasets)

mFollowIR evaluates instruction-following in multilingual retrieval
using TREC NeuCLIR narratives as instructions.

Metrics: nDCG@20, p-MRR.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Dict

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
          - Standard queries (id: "<qid>") — topic text only
          - Instructed queries (id: "<qid>_inst") — topic + narrative instruction
        Both are stored in self.queries["test"] together.
        """
        if self.data_loaded:
            return

        # --- Load queries from rus_map_final.jsonl ---
        jsonl_path = os.path.join(self.data_dir, "rus_map_final.jsonl")
        if not os.path.exists(jsonl_path):
            self._download_mfollowir()

        logger.info(f"Loading mFollowIR-RU queries from {jsonl_path}")
        queries = {}
        self._query_pairs = []  # (std_qid, inst_qid) for p-MRR
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                query_id = str(record["query_id"])
                query_text = record.get("ht_text", "")
                instruction = record.get("instruction_changed", "")

                # Standard query (topic only)
                std_qid = query_id
                queries[std_qid] = query_text

                # Instructed query (topic + narrative)
                inst_qid = f"{query_id}_inst"
                if instruction:
                    queries[inst_qid] = f"{query_text} {instruction}"
                else:
                    queries[inst_qid] = query_text

                self._query_pairs.append((std_qid, inst_qid))

        # --- Load BOTH qrels: original (pre-instruction) and changed (post-instruction) ---
        # Changed qrels (post-instruction relevance)
        qrels_changed_path = os.path.join(self.data_dir, "qrels_changed_test.jsonl")
        if not os.path.exists(qrels_changed_path):
            self._download_qrels("qrels_changed/test.jsonl", qrels_changed_path)

        # Original qrels (pre-instruction relevance)
        qrels_original_path = os.path.join(self.data_dir, "qrels_original_test.jsonl")
        if not os.path.exists(qrels_original_path):
            self._download_qrels("qrels_og/test.jsonl", qrels_original_path)

        # Parse changed qrels → keyed by instructed query IDs
        relevant_docs = defaultdict(dict)
        required_doc_ids = set()
        with open(qrels_changed_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                q_id = str(record["query-id"])
                c_id = str(record["corpus-id"])
                score = int(record["score"])
                inst_qid = f"{q_id}_inst"
                relevant_docs[inst_qid][c_id] = score
                required_doc_ids.add(c_id)

        # Parse original qrels → keyed by standard query IDs
        with open(qrels_original_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                q_id = str(record["query-id"])
                c_id = str(record["corpus-id"])
                score = int(record["score"])
                relevant_docs[q_id][c_id] = score
                required_doc_ids.add(c_id)

        # --- Load official pooled mFollowIR corpus ---
        corpus = self._load_pooled_corpus()

        logger.info(
            f"Loaded mFollowIR-RU: {len(corpus)} pooled docs (hard negatives), "
            f"{len(queries)} queries ({len(self._query_pairs)} pairs for p-MRR), "
            f"{sum(len(v) for v in relevant_docs.values())} judgments"
        )

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": dict(relevant_docs)}
        self.data_loaded = True

    def get_query_pairs(self):
        """Return (std_query_id, inst_query_id) pairs for p-MRR computation."""
        return self._query_pairs

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
