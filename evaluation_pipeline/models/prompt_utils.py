"""Compatibility helpers for prompt roles across evaluation APIs."""

from __future__ import annotations

from typing import Any, Iterable, Optional


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
    if not prefix:
        return list(texts)
    return [f"{prefix}{text.strip()}".strip() for text in texts]
