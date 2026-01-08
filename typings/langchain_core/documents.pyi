from typing import Any, Mapping

class Document:
    page_content: str
    metadata: dict[str, Any]
    def __init__(self, page_content: str = "", metadata: Mapping[str, Any] | None = None) -> None: ...
