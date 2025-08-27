import streamlit as st

@st.cache_resource(show_spinner=False)
def warmup_infra_once() -> bool:
    # Import inside to avoid import-time side effects
    from utils.opensearch_utils import (
        ensure_index_exists,
        ensure_fulltext_index_exists,
        ensure_ingest_log_index_exists,
    )
    from utils.qdrant_utils import ensure_collection_exists

    ensure_index_exists()
    ensure_fulltext_index_exists()
    ensure_ingest_log_index_exists()
    ensure_collection_exists()
    return True

def force_warmup():
    """Manual reset if the user deleted indices while the app is open."""
    warmup_infra_once.clear()
    warmup_infra_once()
