"""Use case helpers for smart file sorter."""

from __future__ import annotations

from typing import Callable, Dict, List

from config import logger
from core.sync.file_sorter import SortOptions, SortPlanItem, apply_sort_plan, build_sort_plan
from utils.timing import set_run_id, timed_block


def preview_sort_plan(
    options: SortOptions,
    run_id: str,
    progress_callback: Callable[[str, Dict[str, int]], None] | None = None,
) -> List[SortPlanItem]:
    """Build a dry-run sorting plan with timing instrumentation."""
    set_run_id(run_id)
    with timed_block(
        "action.tools_file_sorter.preview_classification_dry_run",
        extra={
            "run_id": run_id,
            "include_content": options.include_content,
            "max_files": options.max_files,
        },
        logger=logger,
    ):
        return build_sort_plan(options, progress_callback=progress_callback)


def apply_sort_plan_action(
    plan: List[SortPlanItem],
    min_confidence: float,
    dry_run: bool,
    run_id: str,
) -> Dict[str, object]:
    """Apply a sorting plan with timing instrumentation."""
    set_run_id(run_id)
    with timed_block(
        "action.tools_file_sorter.smart_sort_apply",
        extra={
            "run_id": run_id,
            "dry_run": dry_run,
            "min_confidence": float(min_confidence),
        },
        logger=logger,
    ):
        return apply_sort_plan(plan, min_confidence=min_confidence, dry_run=dry_run)
