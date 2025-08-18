from __future__ import annotations
from typing import Any, Dict, List, Optional

from core.opensearch_client import get_client
from config import OPENSEARCH_FULLTEXT_INDEX, logger


DEFAULT_FRAGMENT_SIZE = 200
DEFAULT_NUM_FRAGMENTS = 3


def build_query(
    q: str,
    *,
    from_: int = 0,
    size: int = 10,
    sort: str = "relevance",
    filetypes: Optional[List[str]] = None,
    modified_from: Optional[str] = None,
    modified_to: Optional[str] = None,
    created_from: Optional[str] = None,
    created_to: Optional[str] = None,
    path_prefix: Optional[str] = None,
    size_gte: Optional[int] = None,
    size_lte: Optional[int] = None,
    fragment_size: int = DEFAULT_FRAGMENT_SIZE,
    num_fragments: int = DEFAULT_NUM_FRAGMENTS,
) -> Dict[str, Any]:
    """Build an OpenSearch query for full-document search."""

    query: Dict[str, Any] = {
        "from": from_,
        "size": size,
        "query": {
            "bool": {
                "must": [
                    {
                        "simple_query_string": {
                            "query": q,
                            "fields": ["text_full^3", "filename"],
                            "default_operator": "or",
                        }
                    }
                ],
                "filter": [],
            }
        },
        "highlight": {
            "fields": {
                "text_full": {
                    "fragment_size": fragment_size,
                    "number_of_fragments": num_fragments,
                }
            }
        },
    }

    filters = query["query"]["bool"]["filter"]

    if filetypes:
        filters.append({"terms": {"filetype": filetypes}})

    if modified_from or modified_to:
        range_body: Dict[str, Any] = {}
        if modified_from:
            range_body["gte"] = modified_from
        if modified_to:
            range_body["lte"] = modified_to
        filters.append({"range": {"modified_at": range_body}})

    if created_from or created_to:
        range_body = {}
        if created_from:
            range_body["gte"] = created_from
        if created_to:
            range_body["lte"] = created_to
        filters.append({"range": {"created_at": range_body}})

    if path_prefix:
        filters.append({"prefix": {"path": path_prefix}})

    if size_gte is not None or size_lte is not None:
        range_body = {}
        if size_gte is not None:
            range_body["gte"] = size_gte
        if size_lte is not None:
            range_body["lte"] = size_lte
        filters.append({"range": {"size_bytes": range_body}})

    if sort == "modified":
        query["sort"] = [{"modified_at": {"order": "desc"}}]
    else:
        query["sort"] = [
            {"_score": {"order": "desc"}},
            {"modified_at": {"order": "desc"}},
        ]

    return query


def search_documents(
    q: str,
    *,
    from_: int = 0,
    size: int = 10,
    sort: str = "relevance",
    filetypes: Optional[List[str]] = None,
    modified_from: Optional[str] = None,
    modified_to: Optional[str] = None,
    created_from: Optional[str] = None,
    created_to: Optional[str] = None,
    path_prefix: Optional[str] = None,
    size_gte: Optional[int] = None,
    size_lte: Optional[int] = None,
    fragment_size: int = DEFAULT_FRAGMENT_SIZE,
    num_fragments: int = DEFAULT_NUM_FRAGMENTS,
) -> Dict[str, Any]:
    """Execute a full-document search and format hits."""

    client = get_client()
    body = build_query(
        q,
        from_=from_,
        size=size,
        sort=sort,
        filetypes=filetypes,
        modified_from=modified_from,
        modified_to=modified_to,
        created_from=created_from,
        created_to=created_to,
        path_prefix=path_prefix,
        size_gte=size_gte,
        size_lte=size_lte,
        fragment_size=fragment_size,
        num_fragments=num_fragments,
    )

    logger.info("Searching full-text index with body: %s", body)
    resp = client.search(index=OPENSEARCH_FULLTEXT_INDEX, body=body)
    hits = resp.get("hits", {})
    results: List[Dict[str, Any]] = []
    for hit in hits.get("hits", []):
        src = hit.get("_source", {})
        results.append(
            {
                **src,
                "score": hit.get("_score"),
                "highlights": hit.get("highlight", {}).get("text_full", []),
                "_id": hit.get("_id"),
            }
        )

    return {
        "hits": results,
        "total": hits.get("total", {}).get("value", 0),
        "took": resp.get("took", 0),
    }
