"""Embedding input-role formatting shared by production and benchmark paths."""

from __future__ import annotations

from typing import Iterable, Literal

EmbeddingInputType = Literal["passage", "query"]
EmbeddingInputFormat = Literal["e5", "raw"]


def prepare_embedding_texts(
    texts: Iterable[str],
    *,
    input_type: EmbeddingInputType,
    input_format: EmbeddingInputFormat | str,
) -> list[str]:
    """Apply the model-family input contract without changing stored source text."""

    if input_type not in {"passage", "query"}:
        raise ValueError(f"unsupported embedding input_type: {input_type}")
    normalized_format = str(input_format).strip().lower()
    values = [str(text) for text in texts]
    if normalized_format == "raw":
        return values
    if normalized_format != "e5":
        raise ValueError(f"unsupported embedding input format: {input_format}")

    prefix = f"{input_type}: "
    return [text if text.startswith(prefix) else f"{prefix}{text}" for text in values]
