import contextlib
import json
import logging
import time
from contextvars import ContextVar
from typing import Any, Dict, Iterator


_RUN_ID: ContextVar[str] = ContextVar("run_id", default="unknown")


def set_run_id(run_id: str | None) -> None:
    _RUN_ID.set(run_id or "unknown")


def _sanitize_extra(extra: Dict[str, Any] | None) -> Dict[str, Any]:
    payload: Dict[str, Any] = dict(extra or {})
    payload.setdefault("run_id", _RUN_ID.get() or "unknown")
    for key, value in list(payload.items()):
        try:
            json.dumps(value)
        except TypeError:
            payload[key] = str(value)
    return payload


@contextlib.contextmanager
def timed_block(
    label: str,
    extra: Dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> Iterator[None]:
    logger = logger or logging.getLogger(__name__)
    safe_extra = _sanitize_extra(extra)
    start = time.perf_counter()
    logger.info("START %s | %s", label, json.dumps(safe_extra, default=str))
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(
            "END   %s | %.3fs | %s",
            label,
            elapsed,
            json.dumps(safe_extra, default=str),
        )


# Instrumentation map (labels -> emit locations):
# - action.chat.get_answer: pages/0_chat.py
# - action.chat.chat_input: pages/0_chat.py
# - action.tools_file_sorter.preview_classification_dry_run: pages/tools_file_sorter.py
# - action.tools_file_sorter.smart_sort_apply: pages/tools_file_sorter.py
# - action.file_resync.scan_plan: pages/7_file_resync.py
# - action.file_resync.apply_safe_actions: pages/7_file_resync.py
# - action.file_resync.apply_destructive_actions: pages/7_file_resync.py
# - action.topic_discovery.run_clustering: pages/11_topic_discovery.py
# - action.topic_discovery.generate_names_llm: pages/11_topic_discovery.py
# - action.topic_discovery.apply_names: pages/11_topic_discovery.py
# - step.files.enumerate: core/sync/file_sorter.py, core/sync/file_resync.py
# - step.content.parse: ingestion/orchestrator.py, core/sync/file_sorter.py
# - step.opensearch.query: utils/opensearch_utils.py, core/vector_store.py, services/topic_naming.py
# - step.opensearch.significant_terms: services/topic_naming.py
# - step.qdrant.call: utils/qdrant_utils.py, core/vector_store.py, services/qdrant_file_vectors.py,
#   services/topic_discovery_clusters.py, core/sync/file_resync.py
# - step.clustering.run: services/topic_discovery_clusters.py
# - step.clustering.k_sweep: services/topic_discovery_clusters.py
# - step.topic_naming.run: pages/11_topic_discovery.py
# - step.topic_naming.cluster: services/topic_naming.py (TIMING_VERBOSE=1)
# - step.topic_naming.parent: services/topic_naming.py (TIMING_VERBOSE=1)
# - action.apply_moves: core/sync/file_sorter.py, core/sync/file_resync.py
