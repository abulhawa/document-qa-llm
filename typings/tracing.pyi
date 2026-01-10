from __future__ import annotations

from contextlib import AbstractContextManager
from enum import Enum
from typing import Any, Iterator


class StatusCode(Enum):
    OK: int
    ERROR: int


class Status:
    status_code: StatusCode
    description: str | None
    def __init__(self, status_code: StatusCode, description: str | None = ...) -> None: ...


def get_current_span() -> Any: ...

def start_span(name: str, kind: str) -> AbstractContextManager[Any]: ...

def record_span_error(span: Any, error: Exception) -> None: ...

CHAIN: str
LLM: str
RETRIEVER: str
EMBEDDING: str
TOOL: str
STATUS_OK: StatusCode
INPUT_VALUE: str
OUTPUT_VALUE: str
