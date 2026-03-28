"""
Custom Trainer for contrastive retriever training with GradCache.

Architecture overview:
  - EncoderWrapper:     wraps a CausalLM, runs forward + last-token pooling + L2 norm.
                        GradCache operates on this wrapper directly.
  - ContrastiveLoss:    InfoNCE loss with correct positive indexing for grouped passages.
  - RetrieverTrainer:   subclass of HuggingFace Trainer; overrides compute_loss to
                        delegate gradient computation to GradCache.

GradCache splits each batch into sub-batches (chunks), runs forward/backward per
chunk, and accumulates gradients — effectively enabling large virtual batch sizes
on limited GPU memory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from grad_cache import GradCache
from grad_cache.context_managers import RandContext
from contextlib import nullcontext


def _last_token_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Extract the embedding of the last non-padding token for each sequence.

    Args:
        hidden_states: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len]

    Returns:
        Tensor of shape [batch_size, hidden_dim].
    """
    sequence_lengths = attention_mask.sum(dim=1) - 1  # [batch_size]
    batch_size = hidden_states.shape[0]
    reps = hidden_states[
        torch.arange(batch_size, device=hidden_states.device), sequence_lengths
    ]
    return reps


class EncoderWrapper(nn.Module):
    """
    Thin wrapper around a CausalLM that produces normalized embeddings.

    GradCache calls this module's forward() with tokenized inputs.
    The output is a single L2-normalized vector per input sequence,
    obtained via last-token pooling (same strategy as RepLLaMA / Promptriever).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # PRE-CACHE LM HEAD OWNER:
        # When using PEFT/LoRA, the layer structure gets aggressively rewritten.
        # Instead of traversing the tree blindly every single forward pass,
        # we find the CausalLM base exactly ONE time and cache it.
        self._owner_of_lm_head = None
        for name, module in self.model.named_modules():
            if "lm_head" in module._modules:
                self._owner_of_lm_head = module
                break

        # Fallback for strange Peft configurations where lm_head is a direct attribute
        if self._owner_of_lm_head is None:
            curr = self.model
            while curr is not None:
                if "lm_head" in getattr(curr, "_modules", {}):
                    self._owner_of_lm_head = curr
                    break
                if hasattr(curr, "base_model") and curr.base_model is not curr:
                    curr = curr.base_model
                elif hasattr(curr, "model") and curr.model is not curr:
                    curr = curr.model
                elif hasattr(curr, "module"):
                    curr = curr.module
                else:
                    break

    @property
    def no_sync(self):
        """Delegate no_sync to the underlying DDP model if it exists."""
        if hasattr(self.model, "no_sync"):
            return self.model.no_sync
        # If no_sync is not available, return a dummy context manager
        return nullcontext

    def forward(self, **kwargs):
        """CausalLM forward -> last-token pool -> L2 normalize."""
        attention_mask = kwargs.get("attention_mask")

        # Memory Optimization:
        # 1. We don't need all 33 hidden states for pooling (saves ~3.6GB).
        # 2. We don't need KV caches for past generated tokens (saves huge VRAM per layer).
        # 3. We absolutely must return dicts to safely extract logits later.
        kwargs["output_hidden_states"] = False
        kwargs["use_cache"] = False
        kwargs["return_dict"] = True

        # We use the pre-cached owner of lm_head to bypass graph traversal cost.
        owner_of_lm_head = self._owner_of_lm_head

        # Temporarily replace lm_head with Identity to bypass the massive linear projection
        # without breaking DDP autograd hooks.
        original_lm_head = None
        if owner_of_lm_head is not None:
            original_lm_head = getattr(owner_of_lm_head, "lm_head", None)
            owner_of_lm_head.lm_head = nn.Identity()
        elif owner_of_lm_head is None:
            # We never expect this since named_modules is exhaustive, but just in case
            pass

        try:
            # For CausalLM without output_hidden_states, outputs.logits IS the final hidden state
            # because we replaced the lm_head projection with an Identity pass-through!
            outputs = self.model(**kwargs)
            last_hidden = outputs.logits
        finally:
            # Restore the real head
            if owner_of_lm_head is not None and original_lm_head is not None:
                owner_of_lm_head.lm_head = original_lm_head

        if attention_mask is not None:
            reps = _last_token_pool(last_hidden, attention_mask)
        else:
            # Fallback: use the very last token of each sequence
            reps = last_hidden[:, -1, :]

        return F.normalize(reps, p=2, dim=1)


class ContrastiveLoss(nn.Module):
    """
    InfoNCE (cross-entropy) loss for bi-encoder retrieval.

    Passage layout per batch:
        [Pos_1, Neg_1_1, ..., Neg_1_N,  Pos_2, Neg_2_1, ..., Neg_2_N, ...]

    For query_i the positive passage sits at index  i * (1 + N).

    Args:
        temperature: scaling factor for the similarity scores.
    """

    def __init__(self, temperature: float = 0.01):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, query_reps: torch.Tensor, passage_reps: torch.Tensor
    ) -> torch.Tensor:
        B = query_reps.shape[0]
        total_passages = passage_reps.shape[0]
        group_size = total_passages // B  # 1 + N

        # Similarity matrix: [B, B * (1+N)]
        scores = torch.matmul(query_reps, passage_reps.t())
        scores = scores / self.temperature

        # Positive target for query_i is passage at index i * group_size
        targets = torch.arange(
            0,
            B * group_size,
            group_size,
            device=scores.device,
            dtype=torch.long,
        )

        return F.cross_entropy(scores, targets)


class RetrieverGradCache(GradCache):
    """
    Subclass of GradCache compatible with HuggingFace Trainer and DDP.
    Implements no_sync gradient accumulation and avoids checkpointing CUDA errors.
    """

    def forward_no_grad(self, model: nn.Module, model_inputs):
        """
        The first forward pass without gradient computation.
        In modern transformers, running gradient_checkpointing inside torch.no_grad()
        with bitsandbytes 4-bit quantization causes CUBLAS_STATUS_INVALID_VALUE errors.
        We temporarily disable gradient checkpointing for this pass if it's active.
        """
        rnd_states = []
        model_reps = []

        # We MUST completely bypass DDP during the no-grad pass to prevent
        # early buffer broadcasts or DDP hang deadlocks.
        original_model = model.model
        from torch.nn.parallel import DistributedDataParallel as DDP

        has_ddp = isinstance(original_model, DDP)
        if has_ddp:
            model.model = original_model.module

        # Recursively find any module with gradient_checkpointing and disable it
        # Cache this lookup to avoid O(N) traversal on every step
        if not hasattr(self, "_gc_modules"):
            self._gc_modules = []
            for m in model.modules():
                if getattr(m, "gradient_checkpointing", False) is True:
                    self._gc_modules.append(m)

        gc_modules = self._gc_modules
        for m in gc_modules:
            try:
                m.gradient_checkpointing = False
            except Exception:
                pass

        with torch.no_grad():
            for x in model_inputs:
                rnd_states.append(RandContext(*self.get_input_tensors(x)))
                y = self.model_call(model, x)
                model_reps.append(self.get_reps(y))

        # Re-enable
        for m in gc_modules:
            try:
                m.gradient_checkpointing = True
            except Exception:
                pass

        if has_ddp:
            model.model = original_model

        model_reps = torch.cat(model_reps, dim=0)
        return model_reps, rnd_states

    def cache_step(self, *model_inputs, no_sync_except_last=False, **loss_kwargs):
        all_reps = []
        all_rnd_states = []

        model_inputs = [
            self.split_inputs(x, chunk_size)
            for x, chunk_size in zip(model_inputs, self.chunk_sizes)
        ]

        for model, x in zip(self.models, model_inputs):
            model_reps, rnd_states = self.forward_no_grad(model, x)
            all_reps.append(model_reps)
            all_rnd_states.append(rnd_states)

        cache, loss = self.build_cache(*all_reps, **loss_kwargs)
        cache = [c.split(chunk_size) for c, chunk_size in zip(cache, self.chunk_sizes)]

        for i, (model, x, model_cache, rnd_states) in enumerate(
            zip(self.models, model_inputs, cache, all_rnd_states)
        ):
            # Only allow sync on the very last chunk of the very last input (passages)
            sync_last_chunk = (
                True if (no_sync_except_last and i == len(self.models) - 1) else False
            )

            self.forward_backward(
                model,
                x,
                model_cache,
                rnd_states,
                sync_last_chunk=sync_last_chunk,
            )

        return loss

    def build_cache(self, *reps: torch.Tensor, **loss_kwargs):
        reps = [r.detach().requires_grad_() for r in reps]
        with torch.cuda.amp.autocast() if self.fp16 else nullcontext():
            loss = self.compute_loss(*reps, **loss_kwargs)

        # Compute gradient of loss w.r.t representations.
        # This graph is detached from the model, so local backward() is perfectly safe.
        if self.fp16 and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        cache = [r.grad for r in reps]
        return cache, loss.detach()

    def forward_backward(
        self,
        model: nn.Module,
        model_inputs,
        cached_gradients,
        random_states,
        sync_last_chunk=False,
    ):
        # Cache the DDP engine lookup to avoid O(N) traversal on every chunk
        if not hasattr(self, "_ds_engine"):
            self._ds_engine = None
            if hasattr(model, "no_sync"):
                self._ds_engine = model
            else:
                for name, m in model.named_modules():
                    if hasattr(m, "no_sync"):
                        self._ds_engine = m
                        break
            if self._ds_engine is None:
                self._ds_engine = model  # fallback

        ds_engine = self._ds_engine

        if hasattr(ds_engine, "no_sync"):
            if sync_last_chunk:
                sync_contexts = [
                    ds_engine.no_sync for _ in range(len(model_inputs) - 1)
                ] + [nullcontext]
            else:
                sync_contexts = [ds_engine.no_sync for _ in range(len(model_inputs))]
        else:
            sync_contexts = [nullcontext for _ in range(len(model_inputs))]

        for x, state, gradient, sync_context in zip(
            model_inputs, random_states, cached_gradients, sync_contexts
        ):
            with sync_context():
                with state:
                    y = self.model_call(model, x)
                reps = self.get_reps(y)

                surrogate = torch.dot(reps.flatten(), gradient.flatten())
                # DeepSpeed requires we use engine.backward(loss) if available
                if hasattr(ds_engine, "backward"):
                    ds_engine.backward(surrogate)
                else:
                    surrogate.backward()


class RetrieverTrainer(Trainer):
    """
    HuggingFace Trainer with GradCache integration for contrastive learning.

    GradCache enables training with large virtual batch sizes by splitting
    the actual batch into memory-friendly sub-batches (controlled by gc_chunk_size).
    """

    def __init__(
        self,
        *args,
        gc_chunk_size: int = 2,
        temperature: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.gc_chunk_size = gc_chunk_size
        self._gc = None  # Lazy initialization
        self._encoder_wrapper = None

    def _get_gc(self, queries, passages) -> GradCache:
        """Lazily build GradCache once the model is on GPU."""
        if self._gc is None:
            self._encoder_wrapper = EncoderWrapper(self.model)

            loss_fn = ContrastiveLoss(temperature=self.temperature)

            # Extract GradScaler safely: in newer transformers it lives
            # inside Accelerator, not directly on the Trainer.
            scaler = getattr(self, "scaler", None)
            if scaler is None:
                scaler = getattr(getattr(self, "accelerator", None), "scaler", None)

            # Dynamically calc physical batch sizes
            q_bs = queries["input_ids"].shape[0]
            p_bs = passages["input_ids"].shape[0]
            group_size = p_bs // q_bs

            import os
            import sys

            rank = os.environ.get("LOCAL_RANK", "0")
            print(
                f"[Rank {rank}] Initializing GradCache with q_chunks={self.gc_chunk_size}, p_chunks={self.gc_chunk_size * group_size}"
            )
            sys.stdout.flush()

            self._gc = RetrieverGradCache(
                models=[self._encoder_wrapper, self._encoder_wrapper],
                chunk_sizes=[self.gc_chunk_size, self.gc_chunk_size * group_size],
                loss_fn=loss_fn,
                fp16=self.args.fp16,
                scaler=scaler,
            )
        return self._gc

    def _get_encoder_wrapper(self, model) -> EncoderWrapper:
        """Get or create an EncoderWrapper for the given model."""
        if getattr(self, "_current_wrapped_model", None) is not model:
            self._current_wrapped_model = model
            self._encoder_wrapper = EncoderWrapper(model)
        return self._encoder_wrapper

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute contrastive loss.

        Uses GradCache during training and a direct forward pass during evaluation.
        """
        queries = inputs["queries"]
        passages = inputs["passages"]

        if not model.training:
            # Bypass GradCache for evaluation runs
            wrapper = self._get_encoder_wrapper(model)
            q_reps = wrapper(**queries)
            p_reps = wrapper(**passages)
            loss_fn = ContrastiveLoss(temperature=self.temperature)
            loss = loss_fn(q_reps, p_reps)
            return (loss, None) if return_outputs else loss

        # --- Training: use GradCache ---
        gc = self._get_gc(queries, passages)

        # Ensure GradCache uses the final, wrapped model instance (e.g. DDP)
        wrapper = self._get_encoder_wrapper(model)
        gc.models = [wrapper, wrapper]

        # Pass no_sync_except_last=True to optimize DDP accumulation
        loss = gc(queries, passages, no_sync_except_last=True)

        return (loss, None) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to bypass HF Trainer's native backward() call,
        because DeepSpeedGradCache computes and accumulates gradients internally.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Map queries to input_ids to satisfy HF Trainer parameter checks
        if "input_ids" not in inputs and "queries" in inputs:
            inputs["input_ids"] = inputs["queries"]["input_ids"]

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        # Note: self.accelerator.backward(loss) is bypassed.
        # RetrieverGradCache._forward_backward accumulates gradients internally.

        # Explicitly step DeepSpeedEngine when backward pass is handled externally
        if self.deepspeed:
            self.deepspeed.step()

        return loss.detach() / self.args.gradient_accumulation_steps
