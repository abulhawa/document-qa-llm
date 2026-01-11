"""Use case for querying ingestion logs."""

from __future__ import annotations

from app.schemas import IngestLogEntry, IngestLogRequest, IngestLogResponse
from utils.opensearch_utils import search_ingest_logs


def fetch_ingest_logs(req: IngestLogRequest) -> IngestLogResponse:
    """Query ingest logs and normalize results for the UI."""
    try:
        logs = search_ingest_logs(
            status=req.status,
            path_query=req.path_query,
            start=req.start_date,
            end=req.end_date,
            size=req.size,
        )
    except Exception:  # noqa: BLE001
        return IngestLogResponse(logs=[])

    entries = [
        IngestLogEntry(
            path=log.get("path") or "",
            bytes=log.get("bytes"),
            status=log.get("status"),
            error_type=log.get("error_type"),
            reason=log.get("reason"),
            stage=log.get("stage"),
            attempt_at=log.get("attempt_at"),
        )
        for log in logs
    ]

    return IngestLogResponse(logs=entries)
