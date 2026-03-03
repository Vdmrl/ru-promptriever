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
        """Load mFollowIR Russian data and NeuCLIR corpus."""
        if self.data_loaded:
            return

        # --- Load queries & relevance judgments from rus_map_final.jsonl ---
        jsonl_path = os.path.join(self.data_dir, "rus_map_final.jsonl")
        if not os.path.exists(jsonl_path):
            self._download_mfollowir()

        logger.info(f"Loading mFollowIR-RU from {jsonl_path}")
        queries = {}
        relevant_docs = defaultdict(dict)

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line.strip())
                query_id = str(record["query_id"])
                query_text = record.get("query", "")
                instruction = record.get("instruction", "")

                # Format: "{query} {instruction}" — same as Promptriever
                if instruction:
                    full_query = f"{query_text} {instruction}"
                else:
                    full_query = query_text

                queries[query_id] = full_query

                # Relevance judgments
                for doc_id, rel_score in record.get("relevance", {}).items():
                    relevant_docs[query_id][str(doc_id)] = int(rel_score)

        # --- Load NeuCLIR corpus ---
        corpus = self._load_neuclir_corpus()

        logger.info(
            f"Loaded mFollowIR-RU: {len(corpus)} docs, "
            f"{len(queries)} queries, "
            f"{sum(len(v) for v in relevant_docs.values())} judgments"
        )

        self.corpus = {"test": corpus}
        self.queries = {"test": queries}
        self.relevant_docs = {"test": dict(relevant_docs)}
        self.data_loaded = True

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

    def _load_neuclir_corpus(self) -> Dict[str, dict]:
        """Load NeuCLIR-2022 Russian corpus.

        Tries ir_datasets first, falls back to a local JSONL file.
        """
        corpus_path = os.path.join(self.data_dir, "neuclir_ru_corpus.jsonl")

        if os.path.exists(corpus_path):
            return self._load_corpus_from_jsonl(corpus_path)

        # Try HuggingFace datasets first — bypasses ir_datasets MD5 hash issues
        try:
            import datasets as hf_datasets

            logger.info(
                "Loading NeuCLIR-2022 Russian corpus via HuggingFace datasets..."
            )
            # neuclir/neuclir1 has a 'default' config, and 'rus' is likely a split
            # Let's use the explicit data files if the builder config is weird.
            corpus_ds = hf_datasets.load_dataset(
                "neuclir/neuclir1",
                data_files={"rus": "data/rus-*.jsonl.gz"},
                split="rus",
            )

            corpus = {}
            for row in corpus_ds:
                doc_id = str(row.get("doc_id", row.get("id", "")))
                text = row.get("text", row.get("segment", ""))
                title = row.get("title", "")
                corpus[doc_id] = {"text": text, "title": title}

            logger.info(f"NeuCLIR corpus loaded: {len(corpus)} documents")

            # Cache to disk
            logger.info(f"Caching corpus to {corpus_path}")
            os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
            with open(corpus_path, "w", encoding="utf-8") as f:
                for doc_id, doc_data in corpus.items():
                    f.write(
                        json.dumps({"doc_id": doc_id, **doc_data}, ensure_ascii=False)
                        + "\n"
                    )

            return corpus

        except Exception as e:
            logger.warning(f"HF datasets loading failed ({e}), trying ir_datasets...")

        # Fallback: ir_datasets (may have MD5 issues with outdated version)
        try:
            import ir_datasets

            logger.info("Loading NeuCLIR-2022 Russian corpus via ir_datasets...")
            ds = ir_datasets.load("neuclir/1/ru")
            corpus = {}
            for doc in ds.docs_iter():
                corpus[doc.doc_id] = {
                    "text": doc.text,
                    "title": doc.title if hasattr(doc, "title") else "",
                }

            os.makedirs(os.path.dirname(corpus_path), exist_ok=True)
            with open(corpus_path, "w", encoding="utf-8") as f:
                for doc_id, doc_data in corpus.items():
                    f.write(
                        json.dumps({"doc_id": doc_id, **doc_data}, ensure_ascii=False)
                        + "\n"
                    )
            return corpus

        except Exception as e:
            logger.error(
                f"Failed to load NeuCLIR corpus: {e}. "
                f"Place neuclir_ru_corpus.jsonl in {self.data_dir}"
            )
            raise

    def _load_corpus_from_jsonl(self, path: str) -> Dict[str, dict]:
        """Load corpus from a cached JSONL file."""
        logger.info(f"Loading cached NeuCLIR corpus from {path}")
        corpus = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = str(doc["doc_id"])
                corpus[doc_id] = {
                    "text": doc.get("text", ""),
                    "title": doc.get("title", ""),
                }
        return corpus
