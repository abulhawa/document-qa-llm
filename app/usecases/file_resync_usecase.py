"""Use case for file path resync operations."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple

from app.schemas import (
    FileResyncApplyRequest,
    FileResyncApplyResponse,
    FileResyncPlanItem,
    FileResyncPlanResponse,
    FileResyncScanRequest,
    ResyncAction,
)
from core.sync.file_resync import (
    ApplyOptions,
    DEFAULT_ALLOWED_EXTENSIONS,
    Action,
    ApplyResult,
    PlanItem,
    ReconciliationPlan,
    build_reconciliation_plan,
    apply_plan as apply_resync_plan,
    scan_files,
)


def _normalize_extensions(allowed_extensions: Iterable[str]) -> list[str]:
    cleaned = [ext.strip().lower() for ext in allowed_extensions if ext and ext.strip()]
    return cleaned or sorted(DEFAULT_ALLOWED_EXTENSIONS)


def _to_schema_plan_item(item: PlanItem) -> FileResyncPlanItem:
    return FileResyncPlanItem(
        bucket=item.bucket,
        reason=item.reason,
        checksum=item.checksum,
        content_id=item.content_id,
        disk_paths=item.disk_paths,
        indexed_paths=item.indexed_paths,
        actions=[ResyncAction(type=action.type, payload=action.payload) for action in item.actions],
        explanation=item.explanation,
        new_checksum=item.new_checksum,
    )


def _to_core_plan_item(item: FileResyncPlanItem) -> PlanItem:
    return PlanItem(
        bucket=item.bucket,
        reason=item.reason,
        checksum=item.checksum,
        content_id=item.content_id,
        disk_paths=item.disk_paths,
        indexed_paths=item.indexed_paths,
        actions=[Action(type=action.type, payload=dict(action.payload)) for action in item.actions],
        explanation=item.explanation,
        new_checksum=item.new_checksum,
    )


def _count_buckets(items: Iterable[FileResyncPlanItem]) -> dict[str, int]:
    counts = Counter(item.bucket for item in items)
    return {bucket: count for bucket, count in counts.items()}


def _to_schema_apply_response(result: ApplyResult) -> FileResyncApplyResponse:
    return FileResyncApplyResponse(
        ingested=result.ingested,
        updated_fulltext=result.updated_fulltext,
        updated_chunks=result.updated_chunks,
        updated_qdrant=result.updated_qdrant,
        deleted_checksums=result.deleted_checksums,
        errors=result.errors,
        counts_by_bucket=result.counts_by_bucket,
    )


def scan_and_plan(
    req: FileResyncScanRequest,
    *,
    retire_replaced_content: bool = False,
) -> Tuple[FileResyncPlanResponse, dict[str, Any]]:
    """Scan filesystem roots and build a reconciliation plan for the UI."""
    allowed_extensions = _normalize_extensions(req.allowed_extensions)
    scan_result = scan_files(req.roots, allowed_extensions)
    plan = build_reconciliation_plan(
        scan_result,
        req.roots,
        retire_replaced_content=retire_replaced_content,
    )

    response = FileResyncPlanResponse(
        items=[_to_schema_plan_item(item) for item in plan.items],
        counts=plan.counts,
        scanned_roots=scan_result.scanned_roots_successful,
        scanned_roots_failed=scan_result.scanned_roots_failed,
        generated_at=plan.generated_at,
    )
    scan_meta = {
        "ignored": scan_result.ignored_files,
        "scanned_roots": scan_result.scanned_roots_successful,
        "failed_roots": scan_result.scanned_roots_failed,
    }
    return response, scan_meta


def apply_plan(req: FileResyncApplyRequest) -> FileResyncApplyResponse:
    """Apply a file resync plan and normalize the response for the UI."""
    core_items = [_to_core_plan_item(item) for item in req.items]
    plan = ReconciliationPlan(
        items=core_items,
        counts=_count_buckets(req.items),
        scanned_roots=[],
        scanned_roots_failed=[],
        generated_at=datetime.now(timezone.utc),
    )
    result = apply_resync_plan(
        plan,
        ApplyOptions(
            ingest_missing=req.ingest_missing,
            apply_safe_only=req.apply_safe_only,
            delete_orphaned=req.delete_orphaned,
            retire_replaced_content=req.retire_replaced_content,
        ),
    )
    return _to_schema_apply_response(result)
