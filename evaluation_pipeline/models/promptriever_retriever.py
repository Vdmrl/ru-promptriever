"""
PrompTriever-style retriever (Retrieval via Language Models).

Unified wrapper for instruction-following retrieval models:
  - samaya-ai/promptriever-llama3.1-8b-v1  (PEFT/LoRA over LLaMA 3.1)
  - Vladimirlv/ru-promptriever-qwen3-4b     (our trained model, merged weights)

Supports both PEFT/LoRA and plain CausalLM checkpoints (auto-detected).
Pooling: last non-padding token (EOS pooling) — same as training.
Query format: "{prefix}{query}" where prefix comes from config.
"""

import logging
from typing import Optional

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
    """Check if the model is a PEFT/LoRA model with actual adapter weights.

    Merged models may retain a leftover adapter_config.json without adapter
    weights. We verify that adapter_model.safetensors or adapter_model.bin
    actually exists before treating the model as PEFT.
    """
    try:
        from peft import PeftConfig

        PeftConfig.from_pretrained(model_name_or_path)
    except Exception:
        return False

    import os

    if os.path.isdir(model_name_or_path):
        return os.path.exists(
            os.path.join(model_name_or_path, "adapter_model.safetensors")
        ) or os.path.exists(
            os.path.join(model_name_or_path, "adapter_model.bin")
        )

    # HuggingFace Hub: check if adapter weight files exist in the repo
    try:
        from huggingface_hub import repo_info

        info = repo_info(model_name_or_path)
        filenames = {f.rfilename for f in info.siblings}
        has_adapter_weights = (
            "adapter_model.safetensors" in filenames
            or "adapter_model.bin" in filenames
        )
        if not has_adapter_weights:
            logger.info(
                f"Found adapter_config.json but no adapter weights in "
                f"{model_name_or_path} — treating as merged model"
            )
        return has_adapter_weights
    except Exception:
        return True


def _detect_peft_base_class(model_name_or_path: str):
    """Detect whether PEFT adapter was trained on AutoModel or AutoModelForCausalLM.

    Inspects adapter weight key prefixes:
      - 'base_model.model.model.' → trained on AutoModelForCausalLM
      - 'base_model.model.' (no extra .model.) → trained on AutoModel
    """
    import os

    from safetensors import safe_open

    # Locate adapter weights
    if os.path.isdir(model_name_or_path):
        adapter_path = os.path.join(model_name_or_path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            adapter_path = os.path.join(model_name_or_path, "adapter_model.bin")
    else:
        from huggingface_hub import hf_hub_download

        try:
            adapter_path = hf_hub_download(
                model_name_or_path, "adapter_model.safetensors"
            )
        except Exception:
            adapter_path = hf_hub_download(model_name_or_path, "adapter_model.bin")

    # Read keys without loading tensors
    if adapter_path.endswith(".safetensors"):
        with safe_open(adapter_path, framework="pt") as f:
            keys = list(f.keys())
    else:
        state_dict = torch.load(adapter_path, map_location="cpu", weights_only=True)
        keys = list(state_dict.keys())

    uses_causal_lm = any("base_model.model.model." in k for k in keys)
    base_cls = AutoModelForCausalLM if uses_causal_lm else AutoModel
    logger.info(f"Detected PEFT base class: {base_cls.__name__}")
    return base_cls


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
        self._is_peft = False  # Will be set to True if PEFT model detected
        logger.info(f"Loading CausalLM: {model_name_or_path} ({dtype})")

        if _is_peft_model(model_name_or_path):
            self._is_peft = True
            logger.info(
                "Detected PEFT/LoRA model. Loading base model + merging LoRA..."
            )
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(model_name_or_path)
            base_model_name = peft_config.base_model_name_or_path
            logger.info(f"Base model: {base_model_name}")

            base_cls = _detect_peft_base_class(model_name_or_path)
            base_model = base_cls.from_pretrained(
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
        self.tokenizer.padding_side = "right"

        logger.info(
            f"Loaded {model_name_or_path} (peft={self._is_peft}), "
            f"hidden_size={self.model.config.hidden_size}, "
            f"dtype={torch_dtype}"
        )

    def _tokenize_with_eos(self, texts, max_length):
        """Official Promptriever tokenization: truncate, append EOS, then pad.

        The original samaya-ai Promptriever models were trained with explicit
        EOS appending after truncation. This ensures the last-token-pool
        always reads from an EOS position, as the model was trained to produce
        sentence embeddings at that position.
        """
        batch_dict = self.tokenizer(
            texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            ids + [self.tokenizer.eos_token_id] for ids in batch_dict["input_ids"]
        ]
        return self.tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

    @torch.no_grad()
    def encode(
        self,
        sentences,
        batch_size: int = 8,
        prompt_name: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """Encode sentences using last-token pooling + L2 normalization."""
        # MTEB 2.10+ passes sentences as a DataLoader — extract raw strings
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

        # Apply query/passage prefix if configured (e.g. Promptriever uses "query:  ")
        if prompt_name == "query" and self.query_prefix:
            sentences = [f"{self.query_prefix}{s}" for s in sentences]
        elif prompt_name == "passage" and self.passage_prefix:
            sentences = [f"{self.passage_prefix}{s}" for s in sentences]

        all_embeddings = []

        for start in trange(0, len(sentences), batch_size, desc="Encoding"):
            batch_texts = sentences[start : start + batch_size]

            if self._is_peft:
                # Official Promptriever tokenization protocol:
                # 1. Truncate to max_length - 1 (leave room for EOS)
                # 2. Manually append EOS token
                # 3. Pad to uniform length
                inputs = self._tokenize_with_eos(batch_texts, self.max_length).to(
                    self.model.device
                )
            else:
                # Merged models (e.g. ru-promptriever): standard tokenization
                # matches how the model was trained (RetrieverCollator)
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.model.device)

            if hasattr(self.model, "lm_head"):
                # AutoModelForCausalLM path: bypass lm_head with Identity
                # to get post-norm hidden states (matching training).
                original_lm_head = self.model.lm_head
                self.model.lm_head = torch.nn.Identity()
                try:
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                    )
                    last_hidden = outputs.logits
                finally:
                    self.model.lm_head = original_lm_head
            else:
                # AutoModel (PEFT) path: model already returns post-norm
                # hidden states as last_hidden_state.
                outputs = self.model(
                    **inputs,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
                last_hidden = outputs.last_hidden_state

            embeddings = _last_token_pool(last_hidden, inputs["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0)

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
