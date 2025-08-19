from __future__ import annotations
from typing import Any, Dict, List, Optional

from core.opensearch_client import get_client
from config import OPENSEARCH_FULLTEXT_INDEX, logger
import html


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
    path_contains: Optional[str] = None,
    size_gte: Optional[int] = None,
    size_lte: Optional[int] = None,
    fragment_size: int = DEFAULT_FRAGMENT_SIZE,
    num_fragments: int = DEFAULT_NUM_FRAGMENTS,
) -> Dict[str, Any]:
    """Build an OpenSearch query for full-document search."""

    base_bool = {
        "must": [
            {
                "simple_query_string": {
                    "query": q,
                    "fields": [
                        "text_full^3",
                        "filename^5",  # text field
                        "filename.keyword^8",  # exact filename boost
                    ],
                    "default_operator": "or",
                }
            }
        ],
        "filter": [],
    }

    # Add path substring
    add_path_contains(base_bool, path_contains)

    # Keep a handle to the bool filter list so downstream code can append ranges, etc.
    filters = base_bool["filter"]

    query: Dict[str, Any] = {
        "from": from_,
        "size": size,
        "track_total_hits": True,
        # soft recency boost without breaking relevance
        "query": {
            "function_score": {
                "query": {"bool": base_bool},
                "boost_mode": "sum",
                "score_mode": "sum",
                "functions": [
                    {
                        "gauss": {
                            "modified_at": {
                                "origin": "now",
                                "scale": "90d",
                                "decay": 0.5,
                            }
                        },
                        "weight": 0.5,
                    }
                ],
            }
        },
        "highlight": {
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
            "fields": {
                "text_full": {
                    "fragment_size": fragment_size,
                    "number_of_fragments": num_fragments,
                    "no_match_size": fragment_size,
                }
            },
        },
        "aggs": {
            "filetypes": {"terms": {"field": "filetype", "size": 20}},
            "top_paths": {"terms": {"field": "path", "size": 10}},
        },
    }

    # IMPORTANT: put file_types in post_filter so aggs remain stable (unaffected by file_type selection)
    selected_types = filetypes if filetypes is not None else filetypes
    post_filters: list[dict[str, Any]] = []
    if selected_types:
        post_filters.append({"terms": {"filetype": selected_types}})
    if post_filters:
        query["post_filter"] = {"bool": {"filter": post_filters}}

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
    path_contains: Optional[str] = None,
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
        path_contains=path_contains,
        size_gte=size_gte,
        size_lte=size_lte,
        fragment_size=fragment_size,
        num_fragments=num_fragments,
    )

    logger.info("Searching full-text index with query: %s", q)
    resp = client.search(index=OPENSEARCH_FULLTEXT_INDEX, body=body)
    hits = resp.get("hits", {})
    results: List[Dict[str, Any]] = []
    for hit in hits.get("hits", []):
        src = hit.get("_source", {})
        # Escape everything, then re-enable our <mark> tags only
        raw_frags = hit.get("highlight", {}).get("text_full", [])
        safe_frags = []
        for f in raw_frags:
            e = html.escape(f)
            e = e.replace("&lt;mark&gt;", "<mark>").replace("&lt;/mark&gt;", "</mark>")
            safe_frags.append(e)
        results.append(
            {
                **src,
                "score": hit.get("_score"),
                "highlights": safe_frags,
                "_id": hit.get("_id"),
            }
        )

    return {
        "hits": results,
        "total": hits.get("total", {}).get("value", 0),
        "took": resp.get("took", 0),
        "aggs": resp.get("aggregations", {}),
    }


def add_path_contains(base_bool: dict, path_contains: str | None) -> None:
    if not path_contains:
        return
    s = path_contains.strip()
    if len(s) < 3:
        # guard very short substrings (ngram min_gram=3)
        return
    base_bool["filter"].append(
        {"match": {"path.ngram": {"query": s, "operator": "and"}}}
    )
