"""
Encoder-Only retriever wrapper for MTEB.

Supports models that work natively with sentence-transformers:
  - intfloat/multilingual-e5-large (requires "query: " / "passage: " prefixes)
  - BAAI/bge-m3 (no prefix needed)
"""

import logging
from typing import List, Optional

import numpy as np
from mteb import EncoderProtocol
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever

logger = logging.getLogger(__name__)


class EncoderRetriever(EncoderProtocol, BaseRetriever):
    """Wrapper for Encoder-Only models via sentence-transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        query_prefix: str = "",
        passage_prefix: str = "",
        max_length: int = 512,
        **kwargs,
    ):
        """
        Args:
            model_name_or_path: HF model ID or local path.
            device: Torch device string.
            query_prefix: Prefix to prepend to queries (e.g. "query: " for E5).
            passage_prefix: Prefix to prepend to passages (e.g. "passage: " for E5).
            max_length: Maximum token length.
        """
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        logger.info(f"Loading encoder model: {model_name_or_path}")
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
        batch_size: int = 64,
        prompt_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences with optional query/passage prefix.

        Args:
            sentences: List of texts to encode.
            batch_size: Batch size for encoding.
            prompt_name: If "query", prepend query_prefix; if "passage",
                         prepend passage_prefix. MTEB passes this automatically.

        Returns:
            np.ndarray of shape [len(sentences), hidden_dim], L2-normalized.
        """
        if prompt_name == "query" and self.query_prefix:
            sentences = [f"{self.query_prefix}{s}" for s in sentences]
        elif prompt_name == "passage" and self.passage_prefix:
            sentences = [f"{self.passage_prefix}{s}" for s in sentences]

        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return embeddings
