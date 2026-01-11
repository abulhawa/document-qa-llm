"""Use case for document ingestion."""

from __future__ import annotations

from app.schemas import IngestRequest, IngestResponse
from ui.ingest_client import enqueue_delete_by_path, enqueue_paths
from utils.inventory import upsert_watch_inventory_for_paths


def ingest(req: IngestRequest) -> IngestResponse:
    """Enqueue ingestion-related actions and normalize output for the UI."""
    errors: list[str] = []
    task_ids: list[str] = []

    if not req.paths:
        return IngestResponse(task_ids=[], queued_count=0, errors=[])

    if req.mode in {"ingest", "reingest"}:
        try:
            upsert_watch_inventory_for_paths(req.paths)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        try:
            task_ids = enqueue_paths(req.paths, mode=req.mode)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    elif req.mode == "delete":
        try:
            task_ids = enqueue_delete_by_path(req.paths)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    else:
        errors.append(f"Unsupported ingest mode: {req.mode}")

    return IngestResponse(
        task_ids=task_ids,
        queued_count=len(task_ids),
        errors=errors,
    )
