"""
Qwen3-Embedding retriever wrapper for MTEB.

Uses sentence-transformers with native prompt support.
Qwen3-Embedding-4B is an instruction-aware embedding model that uses
its own internal prompt format (Instruct: ... / Query: ...).
"""

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever

logger = logging.getLogger(__name__)


class Qwen3EmbeddingRetriever(BaseRetriever):
    """Wrapper for Qwen3-Embedding using sentence-transformers.

    Qwen3-Embedding natively handles instruction prompts through the
    sentence-transformers prompt mechanism. No manual prefix manipulation
    is needed — just pass prompts={"query": "Instruct: ..."} when encoding.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Embedding-4B",
        device: str = "cuda:0",
        max_length: int = 8192,
        **kwargs,
    ):
        logger.info(f"Loading Qwen3-Embedding: {model_name_or_path}")
        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            trust_remote_code=True,
        )
        self.model.max_seq_length = max_length
        logger.info(
            f"Loaded {model_name_or_path}, "
            f"dim={self.model.get_sentence_embedding_dimension()}, "
            f"max_seq_length={self.model.max_seq_length}"
        )

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 16,
        prompt_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences using Qwen3-Embedding.

        For queries, MTEB will pass prompt_name="query" and the model
        will use its internal prompt template. For passages, no prompt.

        Args:
            sentences: List of texts to encode.
            batch_size: Batch size for encoding.
            prompt_name: MTEB prompt name ("query" or "passage").

        Returns:
            np.ndarray of shape [len(sentences), hidden_dim], L2-normalized.
        """
        # MTEB evaluate_dense_custom passes "passage" for documents
        # but Qwen models use "document" as the prompt name
        if prompt_name == "passage":
            prompts = getattr(self.model, "prompts", {})
            if "document" in prompts and "passage" not in prompts:
                prompt_name = "document"
            elif "passage" not in prompts:
                prompt_name = None

        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            prompt_name=prompt_name,
        )
        return embeddings
