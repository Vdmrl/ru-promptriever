"""
Giga-Embeddings-instruct retriever wrapper for MTEB.

Supports ai-sage/Giga-Embeddings-instruct which requires flash_attention_2
and specific query prompt formatting:
"Instruct: {task_description}\nQuery: {query}"
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever

logger = logging.getLogger(__name__)


class GigaEmbeddingRetriever(BaseRetriever):
    """Wrapper for Giga-Embeddings-instruct using sentence-transformers."""

    def __init__(
        self,
        model_name_or_path: str = "ai-sage/Giga-Embeddings-instruct",
        device: str = "cuda:0",
        query_prompt: str = "Instruct: Дан вопрос, необходимо найти абзац текста с ответом\nQuery: ",
        max_length: int = 4096,
        **kwargs,
    ):
        # Monkeypatch for transformers >= 4.45 where ROPE_INIT_FUNCTIONS["default"] was removed.
        # Giga-Embeddings' custom modeling_gigarembed.py hardcodes this key and crashes without it.
        try:
            import transformers.modeling_rope_utils as rope_utils

            if (
                hasattr(rope_utils, "ROPE_INIT_FUNCTIONS")
                and "default" not in rope_utils.ROPE_INIT_FUNCTIONS
            ):
                # "default" was the old name for the standard linear/default RoPE
                # fallback to "linear" if it exists, else we just grab the first available function.
                if "linear" in rope_utils.ROPE_INIT_FUNCTIONS:
                    rope_utils.ROPE_INIT_FUNCTIONS["default"] = (
                        rope_utils.ROPE_INIT_FUNCTIONS["linear"]
                    )
                elif rope_utils.ROPE_INIT_FUNCTIONS:
                    rope_utils.ROPE_INIT_FUNCTIONS["default"] = next(
                        iter(rope_utils.ROPE_INIT_FUNCTIONS.values())
                    )
        except Exception as e:
            logger.warning(f"Failed to monkeypatch ROPE_INIT_FUNCTIONS: {e}")

        """
        Args:
            model_name_or_path: HF model ID.
            device: Torch device string.
            query_prompt: Prompt to prepend to queries.
            max_length: Maximum token length.
        """
        self.query_prompt = query_prompt

        logger.info(f"Loading Giga-Embeddings: {model_name_or_path}")
        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
            },
            config_kwargs={
                "trust_remote_code": True,
                "rope_scaling": {"type": "linear", "factor": 1.0},
            },
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
        """Encode sentences using Giga-Embeddings.

        Args:
            sentences: List of texts to encode.
            batch_size: Batch size for encoding.
            prompt_name: MTEB prompt name ("query" or "passage").

        Returns:
            np.ndarray of shape [len(sentences), hidden_dim], L2-normalized.
        """
        prompt = self.query_prompt if prompt_name == "query" else ""

        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            prompt=prompt,
        )
        return embeddings
