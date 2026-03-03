"""
CausalLM retriever wrapper for MTEB.

For Promptriever-style models (samaya-ai/promptriever-llama3.1-8b-v1,
Vladimirlv/ru-promptriever-qwen3-4b).

Supports both:
- PEFT/LoRA models: loaded via PeftModel + merged (Promptriever)
- Plain CausalLM models: loaded via AutoModelForCausalLM

Pooling: last non-padding token (EOS pooling) — identical to the
_last_token_pool function used during training.

Query format: "{prefix}{query}" — prefix from config (e.g. "query:  " for Promptriever).
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mteb import EncoderProtocol
from tqdm import trange
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from .base import BaseRetriever

logger = logging.getLogger(__name__)


def _last_token_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Extract the embedding of the last non-padding token for each sequence."""
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden_states.shape[0]
    return hidden_states[
        torch.arange(batch_size, device=hidden_states.device), sequence_lengths
    ]


def _is_peft_model(model_name_or_path: str) -> bool:
    """Check if the model is a PEFT/LoRA model by looking for adapter_config.json."""
    try:
        from peft import PeftConfig

        PeftConfig.from_pretrained(model_name_or_path)
        return True
    except Exception:
        return False


class CausalLMRetriever(EncoderProtocol, BaseRetriever):
    """Wrapper for CausalLM bi-encoder models with EOS-token pooling.

    Supports both plain CausalLM and PEFT/LoRA models (e.g. Promptriever).
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_length: int = 512,
        generic_instruction: str = "Найди релевантный документ.",
        query_prefix: str = "",
        passage_prefix: str = "",
        **kwargs,
    ):
        self.max_length = max_length
        self.generic_instruction = generic_instruction
        self.device = device
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        logger.info(f"Loading CausalLM: {model_name_or_path} ({dtype})")

        if _is_peft_model(model_name_or_path):
            logger.info(
                "Detected PEFT/LoRA model. Loading base model + merging LoRA..."
            )
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            base_model_name = peft_config.base_model_name_or_path
            logger.info(f"Base model: {base_model_name}")

            base_model = AutoModel.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name, trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base_model, model_name_or_path)
            self.model = self.model.merge_and_unload()
            self.model.eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

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
        """Encode sentences using last-token pooling + L2 normalization."""
        # Apply query/passage prefix if configured (e.g. Promptriever uses "query:  ")
        if prompt_name == "query" and self.query_prefix:
            sentences = [f"{self.query_prefix}{s}" for s in sentences]
        elif prompt_name == "passage" and self.passage_prefix:
            sentences = [f"{self.passage_prefix}{s}" for s in sentences]

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

            last_hidden = outputs.hidden_states[-1]
            embeddings = _last_token_pool(last_hidden, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)
