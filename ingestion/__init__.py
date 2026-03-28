from __future__ import annotations

from typing import Any


def ingest_one(*args: Any, **kwargs: Any):
    from ingestion.orchestrator import ingest_one as _ingest_one

    return _ingest_one(*args, **kwargs)


__all__ = ["ingest_one"]
