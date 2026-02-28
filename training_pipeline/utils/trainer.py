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

    def forward(self, **kwargs):
        """CausalLM forward -> last-token pool -> L2 normalize."""
        attention_mask = kwargs.get("attention_mask")

        outputs = self.model(**kwargs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]

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

    def _get_gc(self) -> GradCache:
        """Lazily build GradCache once the model is on GPU."""
        if self._gc is None:
            self._encoder_wrapper = EncoderWrapper(self.model)

            loss_fn = ContrastiveLoss(temperature=self.temperature)

            # Extract GradScaler safely: in newer transformers it lives
            # inside Accelerator, not directly on the Trainer.
            scaler = getattr(self, "scaler", None)
            if scaler is None:
                scaler = getattr(getattr(self, "accelerator", None), "scaler", None)

            self._gc = GradCache(
                models=[self._encoder_wrapper, self._encoder_wrapper],
                chunk_sizes=self.gc_chunk_size,
                loss_fn=loss_fn,
                fp16=self.args.fp16,
                scaler=scaler,
            )
        return self._gc

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute contrastive loss via GradCache.

        Expected ``inputs`` keys (produced by RetrieverCollator):
          - queries:       {input_ids, attention_mask}
          - passages:      {input_ids, attention_mask}
          - num_negatives: int
        """
        queries = inputs["queries"]
        passages = inputs["passages"]

        gc = self._get_gc()

        loss = gc(queries, passages, no_sync_except_last=True)

        return (loss, None) if return_outputs else loss
