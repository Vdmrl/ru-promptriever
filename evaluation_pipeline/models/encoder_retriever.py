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
        sentences,
        batch_size: int = 64,
        prompt_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences with optional query/passage prefix.

        Accepts both list[str] (our custom eval) and DataLoader (MTEB 2.10+).
        """
        # MTEB 2.10+ passes sentences as a DataLoader for efficiency.
        # Detect and extract raw strings from it.
        from torch.utils.data import DataLoader as TorchDataLoader

        if isinstance(sentences, TorchDataLoader):
            texts = []
            for batch in sentences:
                if isinstance(batch, dict):
                    batch_texts = batch.get("text", batch.get("sentence", []))
                    texts.extend(
                        batch_texts
                        if isinstance(batch_texts, list)
                        else list(batch_texts)
                    )
                elif isinstance(batch, (list, tuple)):
                    texts.extend(batch)
                else:
                    texts.append(str(batch))
            sentences = texts

        if prompt_name == "query" and self.query_prefix:
            sentences = [f"{self.query_prefix}{s}" for s in sentences]
        elif prompt_name == "passage" and self.passage_prefix:
            sentences = [f"{self.passage_prefix}{s}" for s in sentences]

        # Encode in explicit batches to avoid SentenceTransformers 5.x sort bug
        all_embeddings = []
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            embs = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(embs)

        return (
            np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])
        )

    def similarity(self, e1, e2):
        """Cosine similarity via dot product (embeddings are L2-normalized)."""
        import torch

        if not isinstance(e1, torch.Tensor):
            e1 = torch.as_tensor(e1)
        if not isinstance(e2, torch.Tensor):
            e2 = torch.as_tensor(e2)
        return e1 @ e2.T

    def similarity_pairwise(self, e1, e2):
        import torch

        if not isinstance(e1, torch.Tensor):
            e1 = torch.as_tensor(e1)
        if not isinstance(e2, torch.Tensor):
            e2 = torch.as_tensor(e2)
        return (e1 * e2).sum(dim=1)
