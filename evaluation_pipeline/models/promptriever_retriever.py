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
from .prompt_utils import apply_role_prefix, materialize_texts, resolve_prompt_name

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


def _is_peft_model(model_name_or_path: str, revision: Optional[str] = None) -> bool:
    """Check if the model is a PEFT/LoRA model with actual adapter weights.

    Merged models may retain a leftover adapter_config.json without adapter
    weights. We verify that adapter_model.safetensors or adapter_model.bin
    actually exists before treating the model as PEFT.
    """
    try:
        from peft import PeftConfig

        PeftConfig.from_pretrained(model_name_or_path, revision=revision)
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

        info = repo_info(model_name_or_path, revision=revision)
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


def _detect_peft_base_class(
    model_name_or_path: str, revision: Optional[str] = None
):
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
                model_name_or_path, "adapter_model.safetensors", revision=revision
            )
        except Exception:
            adapter_path = hf_hub_download(
                model_name_or_path, "adapter_model.bin", revision=revision
            )

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

    @property
    def mteb_model_meta(self):
        """MTEB metadata used when serializing raw task predictions."""
        return getattr(self, "_mteb_model_meta", None)

    @mteb_model_meta.setter
    def mteb_model_meta(self, value):
        self._mteb_model_meta = value

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_length: int = 512,
        generic_instruction: str = "Найди релевантный документ.",
        query_prefix: str = "",
        passage_prefix: str = "",
        append_eos: Optional[bool] = None,
        mteb_document_title_separator: Optional[str] = None,
        revision: Optional[str] = None,
        base_revision: Optional[str] = None,
        **kwargs,
    ):
        self._mteb_model_meta = None
        self._logged_prompt_roles = set()
        self.max_length = max_length
        self.generic_instruction = generic_instruction
        self.device = device
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        # Tokenization is an explicit evaluation-protocol choice. Historical
        # configs may omit it; those retain the previous PEFT-based behavior.
        self.append_eos = append_eos
        self.mteb_document_title_separator = mteb_document_title_separator

        torch_dtype = getattr(torch, dtype, torch.bfloat16)
        self._is_peft = False  # Will be set to True if PEFT model detected
        logger.info(f"Loading CausalLM: {model_name_or_path} ({dtype})")
        logger.info(
            "Pinned model protocol: adapter_revision=%r, base_revision=%r, "
            "query_prefix=%r, passage_prefix=%r, append_eos=%r, "
            "mteb_document_title_separator=%r",
            revision,
            base_revision,
            query_prefix,
            passage_prefix,
            append_eos,
            mteb_document_title_separator,
        )

        if _is_peft_model(model_name_or_path, revision=revision):
            self._is_peft = True
            logger.info(
                "Detected PEFT/LoRA model. Loading base model + merging LoRA..."
            )
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(
                model_name_or_path, revision=revision
            )

            base_model_name = peft_config.base_model_name_or_path
            logger.info(f"Base model: {base_model_name}")

            base_cls = _detect_peft_base_class(
                model_name_or_path, revision=revision
            )
            base_model = base_cls.from_pretrained(
                base_model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                revision=base_revision,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                revision=base_revision,
            )

            # Resolve accelerate/PEFT compatibility issue where no_split_module_classes
            # contains a set object, causing unhashable type exceptions.
            import accelerate.utils.modeling
            import peft.peft_model

            orig_accel_get_balanced_memory = accelerate.utils.modeling.get_balanced_memory
            orig_peft_get_balanced_memory = getattr(peft.peft_model, "get_balanced_memory", None)

            def patched_get_balanced_memory(*args, **kwargs):
                if "no_split_module_classes" in kwargs:
                    classes = kwargs["no_split_module_classes"]
                    if classes is not None:
                        clean_classes = []
                        for c in classes:
                            if isinstance(c, (set, list, tuple)):
                                clean_classes.extend(c)
                            elif isinstance(c, str):
                                clean_classes.append(c)
                        kwargs["no_split_module_classes"] = clean_classes
                return orig_accel_get_balanced_memory(*args, **kwargs)

            accelerate.utils.modeling.get_balanced_memory = patched_get_balanced_memory
            if orig_peft_get_balanced_memory:
                peft.peft_model.get_balanced_memory = patched_get_balanced_memory

            try:
                self.model = PeftModel.from_pretrained(
                    base_model, model_name_or_path, config=peft_config, revision=revision
                )
            finally:
                accelerate.utils.modeling.get_balanced_memory = orig_accel_get_balanced_memory
                if orig_peft_get_balanced_memory:
                    peft.peft_model.get_balanced_memory = orig_peft_get_balanced_memory

            self.model = self.model.merge_and_unload()
            self.model.eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=True, revision=revision
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
                revision=revision
            )
            self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        # Preserve the historical behavior for old configs while allowing
        # adapters trained with standard tokenization to opt out explicitly.
        if self.append_eos is None:
            self.append_eos = self._is_peft

        logger.info(
            f"Loaded {model_name_or_path} (peft={self._is_peft}), "
            f"hidden_size={self.model.config.hidden_size}, "
            f"dtype={torch_dtype}, append_eos={self.append_eos}"
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
        # MTEB 2.10+ uses ``prompt_type`` (PromptType.query/document), while
        # the custom evaluation path historically uses ``prompt_name``
        # ("query"/"passage").  Normalize both APIs before applying prefixes.
        prompt_name = resolve_prompt_name(prompt_name, kwargs.get("prompt_type"))

        # The default None preserves MTEB's official ``title + ' ' + text``.
        # A non-None separator is an explicit diagnostic override and only
        # applies to document DataLoaders, never to queries or plain lists.
        diagnostic_separator = (
            self.mteb_document_title_separator
            if prompt_name == "passage"
            else None
        )
        sentences = materialize_texts(
            sentences, document_title_separator=diagnostic_separator
        )

        if prompt_name not in self._logged_prompt_roles:
            prefix = (
                self.query_prefix
                if prompt_name == "query"
                else self.passage_prefix if prompt_name == "passage" else ""
            )
            logger.info(
                "Encoding prompt role=%r (MTEB prompt_type=%r), prefix=%r",
                prompt_name,
                kwargs.get("prompt_type"),
                prefix,
            )
            self._logged_prompt_roles.add(prompt_name)

        # Apply query/passage prefix if configured (e.g. Promptriever uses "query:  ")
        sentences = apply_role_prefix(
            sentences,
            prompt_name,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
        )

        all_embeddings = []

        for start in trange(0, len(sentences), batch_size, desc="Encoding"):
            batch_texts = sentences[start : start + batch_size]

            if self.append_eos:
                # Official Promptriever tokenization protocol:
                # 1. Truncate to max_length - 1 (leave room for EOS)
                # 2. Manually append EOS token
                # 3. Pad to uniform length
                inputs = self._tokenize_with_eos(batch_texts, self.max_length).to(
                    self.model.device
                )
            else:
                # Standard tokenizer output without manual EOS insertion.
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
