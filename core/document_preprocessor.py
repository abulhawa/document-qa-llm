from __future__ import annotations
from typing import Any, Iterable, List, Dict
from core.preprocessing import preprocess_document, PreprocessConfig
from langchain_core.documents import Document


def _to_documents(obj: Any, source_path: str) -> List[Document]:
    """Coerce common loader outputs into List[Document] (preserving metadata when present)."""
    seq: Iterable[Any] = obj if isinstance(obj, list) else [obj]
    out: List[Document] = []
    for o in seq:
        if isinstance(o, Document):
            out.append(o)
            continue
        if hasattr(o, "page_content") and hasattr(o, "metadata"):
            out.append(
                Document(
                    page_content=getattr(o, "page_content", "") or "",
                    metadata=getattr(o, "metadata", {}) or {},
                )
            )
            continue
        if isinstance(o, dict):
            text = str(o.get("text") or o.get("content") or "")
            base_meta: Dict[str, Any] = dict(o.get("metadata", {}))
            # fold any extra keys into metadata to avoid data loss
            extras = {
                k: v for k, v in o.items() if k not in ("text", "content", "metadata")
            }
            meta = {**extras, **base_meta}
            meta.setdefault("source", source_path)
            out.append(Document(page_content=text, metadata=meta))
            continue
        # unknown type → stringify, keep at least the source
        out.append(
            Document(
                page_content=str(o),
                metadata={"source": source_path, "_orig_type": type(o).__name__},
            )
        )
    return out


def preprocess_to_documents(
    docs_like: Any,
    *,
    source_path: str,
    cfg: PreprocessConfig | None = None,
    doc_type: str,
) -> List[Document]:
    """
    Convert input to List[Document], run text preprocessing across pages,
    and return the same list with cleaned page_content. Metadata is preserved.
    """
    docs: List[Document] = _to_documents(docs_like, source_path)
    if not docs:
        return docs

    cfg = cfg or PreprocessConfig()  # use your defaults
    dt = doc_type

    pages = [d.page_content or "" for d in docs]
    _, cleaned = preprocess_document(pages, cfg, doc_type=dt)

    for d, c in zip(docs, cleaned):
        d.page_content = c

    return docs
