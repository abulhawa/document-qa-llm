import streamlit as st


@st.cache_resource(show_spinner=False)
def warmup_infra_once() -> bool:
    # Import inside to avoid import-time side effects
    from config import (
        CHUNKS_INDEX,
        FULLTEXT_INDEX,
        INGEST_LOG_INDEX,
        WATCHLIST_INDEX,
    )
    from utils.opensearch.indexes import ensure_index_exists
    from utils.qdrant_utils import ensure_collection_exists

    ensure_index_exists(CHUNKS_INDEX)
    ensure_index_exists(FULLTEXT_INDEX)
    ensure_index_exists(INGEST_LOG_INDEX)
    ensure_index_exists(WATCHLIST_INDEX)
    ensure_collection_exists()
    return True


def force_warmup():
    """Manual reset if the user deleted indices while the app is open."""
    warmup_infra_once.clear()
    warmup_infra_once()
