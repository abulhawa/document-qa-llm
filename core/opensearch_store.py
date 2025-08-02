from typing import List, Dict, Any
from opensearchpy import OpenSearch, helpers
from config import OPENSEARCH_HOST, OPENSEARCH_PORT, logger
from tracing import start_span, INPUT_VALUE, OUTPUT_VALUE, TOOL

# Define your index name
INDEX_NAME = "documents"

# Set up the OpenSearch client
client = OpenSearch(hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}])

# Analyzer/mapping config (optional: can also be created manually in advance)
INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "custom_text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "asciifolding"],
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "custom_text_analyzer"},
            "path": {"type": "text"},
            "chunk_index": {"type": "integer"},
            "checksum": {"type": "keyword"},
            "filetype": {"type": "keyword"},
            "indexed_at": {"type": "date"},
            "created_at": {"type": "date"},
            "modified_at": {"type": "date"},
            "page": {"type": "integer"},
            "location_percent": {"type": "float"},
        }
    },
}


def ensure_index_exists():
    if not client.indices.exists(index=INDEX_NAME):
        logger.info(f"Creating OpenSearch index: {INDEX_NAME}")
        client.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)


def index_documents(chunks: List[Dict[str, Any]]) -> None:
    """Index a list of chunks into OpenSearch."""

    ensure_index_exists()
    actions = [
        {
            "_index": INDEX_NAME,
            "_id": chunk["id"],
            "_source": {k: v for k, v in chunk.items() if k != "id"},
        }
        for chunk in chunks
    ]
    success_count, errors = helpers.bulk(client, actions)
    if errors:
        logger.error(f"❌ OpenSearch indexing failed for {len(errors)} chunks")
    else:
        logger.info(f"✅ OpenSearch successfully indexed {success_count} chunks")


def search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    with start_span("opensearch_search", kind=TOOL) as span:
        logger.info(f"Searching OpenSearch for query: '{query}' with top_k={top_k}")
        span.set_attribute(INPUT_VALUE, query)
        span.set_attribute("top_k", top_k)

        response = client.search(
            index=INDEX_NAME,
            body={
                "size": top_k * 3,  # fetch extra for dedup
                "query": {"match": {"text": {"query": query, "operator": "and"}}},
                "sort": [{"modified_at": {"order": "desc"}}],
            },
        )
        hits = response.get("hits", {}).get("hits", [])
        logger.info(f"OpenSearch returned {len(hits)} hits before deduplication.")
        span.set_attribute("raw_hits", len(hits))

        seen_checksums = set()
        results = []
        for hit in hits:
            src = hit["_source"]
            if src["checksum"] in seen_checksums:
                logger.info(f"Duplicate found for file: {src['path']}, skipping.")
                continue
            seen_checksums.add(src["checksum"])
            results.append({**src, "score": hit["_score"]})
            if len(results) >= top_k:
                break

        logger.info(f"Returning {len(results)} results.")
        span.set_attribute(OUTPUT_VALUE, f"{len(results)} results")
        return results
