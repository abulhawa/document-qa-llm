from core.opensearch_client import get_client
from config import (
    CHUNKS_INDEX,
    FULLTEXT_INDEX,
    INGEST_LOG_INDEX,
    WATCH_INVENTORY_INDEX,
    WATCHLIST_INDEX,
    logger,
)
from utils.opensearch_utils import (
    CHUNKS_INDEX_SETTINGS,
    FULLTEXT_INDEX_SETTINGS,
    INGEST_LOGS_INDEX_SETTINGS,
)


def ensure_index_exists(index: str) -> None:
    """Ensure an index exists using the appropriate settings for known indices."""
    if index == CHUNKS_INDEX:
        body = CHUNKS_INDEX_SETTINGS
    elif index == FULLTEXT_INDEX:
        body = FULLTEXT_INDEX_SETTINGS
    elif index == INGEST_LOG_INDEX:
        body = INGEST_LOGS_INDEX_SETTINGS
    elif index == WATCH_INVENTORY_INDEX:
        from utils.inventory import INVENTORY_INDEX_SETTINGS  # lazy import
        body = INVENTORY_INDEX_SETTINGS
    elif index == WATCHLIST_INDEX:
        from utils.watchlist import WATCHLIST_INDEX_SETTINGS  # lazy import
        body = WATCHLIST_INDEX_SETTINGS
    else:
        body = None
    if body is None:
        raise ValueError(f"Unknown index '{index}' for ensure_index_exists")
    client = get_client()
    if not client.indices.exists(index=index):
        logger.info(f"Creating OpenSearch index: {index}")
        client.indices.create(
            index=index,
            body=body,
            params={"wait_for_active_shards": "1"},
        )

