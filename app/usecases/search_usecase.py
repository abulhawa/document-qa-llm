"""Use case for document search."""

from __future__ import annotations

from app.schemas import SearchHit, SearchRequest, SearchResponse
from config import CHUNKS_INDEX, FULLTEXT_INDEX
from core.opensearch_client import get_client
from utils.fulltext_search import search_documents
from utils.opensearch_utils import list_files_missing_fulltext


def search(req: SearchRequest) -> SearchResponse:
    """Run full-text search and normalize output for the UI."""
    if not req.query.strip():
        return SearchResponse()

    result = search_documents(
        req.query,
        from_=req.page * req.page_size,
        size=req.page_size,
        sort=req.sort,
        filetypes=req.filetypes or None,
        modified_from=req.modified_from,
        modified_to=req.modified_to,
        created_from=req.created_from,
        created_to=req.created_to,
        path_contains=req.path_contains,
    )

    hits = []
    for hit in result.get("hits", []):
        bytes_value = hit.get("size_bytes")
        if bytes_value is None:
            bytes_value = hit.get("bytes")
        hits.append(
            SearchHit(
                path=hit.get("path", "") or "",
                filename=hit.get("filename"),
                filetype=hit.get("filetype"),
                score=hit.get("score"),
                bytes=bytes_value,
                modified_at=hit.get("modified_at"),
                created_at=hit.get("created_at"),
                highlights=hit.get("highlights") or [],
                metadata={
                    "id": hit.get("_id"),
                    "checksum": hit.get("checksum"),
                },
            )
        )

    return SearchResponse(
        total=result.get("total", 0),
        took_ms=result.get("took", 0),
        hits=hits,
        aggregations=result.get("aggs", {}),
    )


def find_missing_files(limit: int) -> list[str]:
    """Return file paths missing from the full-text index."""
    missing_files = list_files_missing_fulltext(size=limit)
    return [
        file_meta.get("path", "")
        for file_meta in missing_files
        if file_meta.get("path")
    ]


def refresh_search_index() -> bool:
    """Refresh the search indices so queries see the latest data."""
    try:
        get_client().indices.refresh(index=",".join([CHUNKS_INDEX, FULLTEXT_INDEX]))
        return True
    except Exception:
        return False
