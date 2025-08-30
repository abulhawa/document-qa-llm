import time
from datetime import datetime
from typing import Any

from config import logger, INGEST_LOG_INDEX
from core.opensearch_client import get_client
from opensearchpy import exceptions


class IngestLogEmitter:
    """Context manager to emit ingest attempt logs to OpenSearch."""

    def __init__(
        self,
        *,
        path: str,
        index: str = INGEST_LOG_INDEX,
        op: str = "ingest",
        source: str = "ingest_page",
    ) -> None:
        self._client = None
        self.index = index
        self.doc: dict[str, Any] = {
            "path": path,
            "op": op,
            "source": source,
        }
        self._start = 0.0
        self._finished = False

    def __enter__(self) -> "IngestLogEmitter":
        self._start = time.time()
        # Record attempt time in the local timezone
        self.doc["attempt_at"] = datetime.now().astimezone().isoformat()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # If an exception bubbled up without calling fail/done
        if exc and not self._finished:
            self.fail(
                stage="unknown", error_type=exc.__class__.__name__, reason=str(exc)
            )

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
            self._client.index(index=self.index, body=self.doc, op_type="create")
        except exceptions.OpenSearchException as e:
            logger.warning(f"Failed to write ingest log: {e}")
        except Exception as e:
            logger.warning(f"Unexpected ingest log failure: {e}")
