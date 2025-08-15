import time
import uuid
from datetime import datetime, timezone
from typing import Any

from config import logger, INGEST_LOG_INDEX
from core.opensearch_client import get_client
from opensearchpy import exceptions
from utils.opensearch_utils import ensure_ingest_log_index_exists


class IngestLogEmitter:
    """Context manager to emit ingest attempt logs to OpenSearch."""

    def __init__(
        self,
        *,
        path: str,
        index: str = INGEST_LOG_INDEX,
        op: str = "ingest",
        source: str = "ingest_page",
        run_id: str | None = None,
    ) -> None:
        self._client = None
        self.index = index
        self.log_id = str(uuid.uuid4())
        self.doc: dict[str, Any] = {
            "log_id": self.log_id,
            "path": path,
            "op": op,
            "source": source,
        }
        if run_id:
            self.doc["run_id"] = run_id
        self._start = 0.0
        self._finished = False

    def __enter__(self) -> "IngestLogEmitter":
        self._start = time.time()
        self.doc["attempt_at"] = datetime.now(timezone.utc).isoformat()
        try:
            ensure_ingest_log_index_exists()
        except Exception as e:  # best effort
            logger.warning(f"ensure_ingest_log_index_exists failed: {e}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # If an exception bubbled up without calling fail/done
        if exc and not self._finished:
            self.fail(stage="unknown", error_type=exc.__class__.__name__, reason=str(exc))

    def set(self, **fields: Any) -> None:
        self.doc.update(fields)

    def done(self, *, status: str) -> None:
        if self._finished:
            return
        self._finished = True
        self.doc["status"] = status
        self.doc["duration_ms"] = int((time.time() - self._start) * 1000)
        self._write()

    def fail(self, *, stage: str, error_type: str, reason: str) -> None:
        if self._finished:
            return
        self._finished = True
        self.doc["status"] = "Failed"
        self.doc["stage"] = stage
        self.doc["error_type"] = error_type
        self.doc["reason"] = reason
        self.doc["duration_ms"] = int((time.time() - self._start) * 1000)
        self._write()

    def _write(self) -> None:
        if self._client is None:
            try:
                self._client = get_client()
            except Exception as e:
                logger.warning(f"Failed to init ingest log client: {e}")
                return
        try:
            self._client.index(index=self.index, id=self.log_id, body=self.doc)
        except exceptions.OpenSearchException as e:
            logger.warning(f"Failed to write ingest log: {e}")
        except Exception as e:
            logger.warning(f"Unexpected ingest log failure: {e}")
