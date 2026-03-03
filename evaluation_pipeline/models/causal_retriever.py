"""
CausalLM retriever wrapper for MTEB.

For Promptriever-style models (samaya-ai/promptriever-llama3.1-8b-v1,
Vladimirlv/ru-promptriever-qwen3-4b).

Pooling: last non-padding token (EOS pooling) — identical to the
_last_token_pool function used during training.

Query format: "{query} {instruction}" — plain concatenation, no prefixes.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseRetriever

logger = logging.getLogger(__name__)


def _last_token_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Extract the embedding of the last non-padding token for each sequence.

    This is the same pooling used during training (trainer.py).

    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len]

    Returns:
        Tensor of shape [batch_size, hidden_dim].
    """
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[
        torch.arange(batch_size, device=hidden_states.device), sequence_lengths
    ]


class CausalLMRetriever(BaseRetriever):
    """Wrapper for CausalLM bi-encoder models with EOS-token pooling."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_length: int = 512,
        generic_instruction: str = "Найди релевантный документ.",
        **kwargs,
    ):
        """
        Args:
            model_name_or_path: HF model ID or local path.
            device: Torch device string.
            dtype: Data type string ("bfloat16", "float16", "float32").
            max_length: Maximum token length for queries and passages.
            generic_instruction: Default instruction appended on OOD benchmarks.
        """
        self.max_length = max_length
        self.generic_instruction = generic_instruction
        self.device = device

        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        logger.info(f"Loading CausalLM: {model_name_or_path} ({dtype})")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # For left-padding in decoder models (so the last real token is EOS)
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        logger.info(
            f"Loaded {model_name_or_path}, "
            f"hidden_size={self.model.config.hidden_size}, "
            f"dtype={torch_dtype}"
        )

    @torch.no_grad()
    def encode(
        self,
        sentences: List[str],
        batch_size: int = 8,
        prompt_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences using last-token pooling + L2 normalization.

        Args:
            sentences: List of texts to encode.
            batch_size: Batch size for encoding.
            prompt_name: "query" or "passage" (not used to add prefixes —
                         instructions are expected to be already in the text).

        Returns:
            np.ndarray of shape [len(sentences), hidden_dim], L2-normalized.
        """
        all_embeddings = []

        for start in trange(0, len(sentences), batch_size, desc="Encoding"):
            batch_texts = sentences[start : start + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            # Use the last hidden state layer
            last_hidden = outputs.hidden_states[-1]
            embeddings = _last_token_pool(last_hidden, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)
