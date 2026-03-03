"""
BM25 retriever using bm25s.

BM25 does not produce dense embeddings, so it cannot go through the standard
MTEB encode() → cosine-similarity path. Instead we implement a custom
retrieval method that is called directly from evaluate.py.
"""

import logging
from typing import Dict, List, Optional

import bm25s
import numpy as np
from Stemmer import Stemmer

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 retriever using bm25s with Snowball stemming for Russian."""

    def __init__(self):
        self.stemmer = Stemmer("russian")
        self.index: Optional[bm25s.BM25] = None
        self.doc_ids: Optional[List[str]] = None

    def index_corpus(self, corpus: Dict[str, dict]) -> None:
        """Build a BM25 index from a corpus dict.

        Args:
            corpus: {doc_id: {"text": str, "title": str, ...}}
        """
        self.doc_ids = list(corpus.keys())
        texts = []
        for doc_id in self.doc_ids:
            doc = corpus[doc_id]
            title = doc.get("title", "")
            text = doc.get("text", "")
            texts.append(f"{title} {text}" if title else text)

        logger.info(f"Indexing {len(texts)} documents with BM25...")
        corpus_tokens = bm25s.tokenize(texts, stemmer=self.stemmer)
        self.index = bm25s.BM25()
        self.index.index(corpus_tokens)
        logger.info("BM25 index built.")

    def retrieve(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """Retrieve top-k documents for each query.

        Args:
            queries: {query_id: query_text}
            top_k: Number of results per query.

        Returns:
            {query_id: {doc_id: score, ...}}
        """
        if self.index is None or self.doc_ids is None:
            raise RuntimeError("Index not built. Call index_corpus() first.")

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]

        query_tokens = bm25s.tokenize(query_texts, stemmer=self.stemmer)
        results, scores = self.index.retrieve(query_tokens, k=top_k)

        output = {}
        for i, qid in enumerate(query_ids):
            doc_scores = {}
            for j in range(results.shape[1]):
                doc_idx = results[i, j]
                if doc_idx < len(self.doc_ids):
                    doc_scores[self.doc_ids[doc_idx]] = float(scores[i, j])
            output[qid] = doc_scores

        return output
