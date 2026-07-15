"""Compatibility helpers for prompt roles across evaluation APIs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional


def _batch_texts(batch: Any) -> list[str]:
    """Extract strings from one MTEB/DataLoader text batch."""
    if isinstance(batch, str):
        return [batch]
    if isinstance(batch, dict):
        for key in ("text", "sentence"):
            if key in batch:
                value = batch[key]
                if isinstance(value, str):
                    return [value]
                return [str(item) for item in value]
        raise ValueError(
            "Text batch dictionary has neither a 'text' nor a 'sentence' field"
        )
    if isinstance(batch, (list, tuple)):
        return [str(item) for item in batch]
    return [str(batch)]


def materialize_texts(sentences: Any) -> list[str]:
    """Normalize ordinary iterables and MTEB DataLoaders to a string list."""
    if isinstance(sentences, str):
        return [sentences]

    # Keep this helper importable in lightweight CPU-only checks.
    try:
        from torch.utils.data import DataLoader as TorchDataLoader
    except (ImportError, OSError):  # pragma: no cover - environment dependent
        TorchDataLoader = ()

    if TorchDataLoader and isinstance(sentences, TorchDataLoader):
        texts: list[str] = []
        for batch in sentences:
            texts.extend(_batch_texts(batch))
        return texts

    return [str(item) for item in sentences]


def resolve_prompt_name(
    prompt_name: Optional[str] = None,
    prompt_type: Any = None,
) -> Optional[str]:
    """Return the legacy query/passage role used by local model wrappers.

    Custom evaluation code passes ``prompt_name`` (``query``/``passage``),
    whereas MTEB 2.10 passes ``prompt_type``
    (``PromptType.query``/``PromptType.document``).  Supporting both avoids
    silently dropping model-specific query and document prefixes.
    """
    if prompt_name is not None:
        return prompt_name

    value = getattr(prompt_type, "value", prompt_type)
    if value == "query":
        return "query"
    if value in ("document", "passage"):
        return "passage"
    return None


def apply_role_prefix(
    texts: Iterable[str],
    prompt_name: Optional[str],
    query_prefix: str = "",
    passage_prefix: str = "",
) -> list[str]:
    """Apply an explicit role prefix while preserving its internal spacing."""
    prefix = (
        query_prefix
        if prompt_name == "query"
        else passage_prefix if prompt_name == "passage" else ""
    )
    materialized = materialize_texts(texts)
    if not prefix:
        return materialized
    return [f"{prefix}{text.strip()}".strip() for text in materialized]
