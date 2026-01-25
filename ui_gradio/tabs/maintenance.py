"""Knowledge base maintenance tab."""

from __future__ import annotations

import math

import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List

from celery import Celery

import gradio as gr
import pandas as pd
import redis

from app.schemas import FileResyncApplyRequest, FileResyncPlanItem, FileResyncScanRequest
from app.usecases import (
    admin_usecase,
    duplicates_usecase,
    file_resync_usecase,
    index_viewer_usecase,
    worker_emergency_usecase,
    watchlist_usecase,
)
from core.sync.file_resync import DEFAULT_ALLOWED_EXTENSIONS
from ui.task_status import add_records, clear_finished, fetch_states
from utils.file_utils import format_file_size, open_file_local, show_in_folder
from utils.time_utils import format_timestamp, format_timestamp_ampm

DEFAULT_ROOT = os.getenv("LOCAL_SYNC_ROOT", "")
INDEX_COLUMNS = [
    "Filename",
    "Path",
    "Filetype",
    "Modified",
    "Created",
    "Indexed",
    "Size",
    "OpenSearch Chunks",
    "Qdrant Chunks",
    "Checksum",
]
PAGE_SIZE_OPTIONS = [5, 25, 50, 100]
MAX_BULK_OPEN = 10
REASON_ORDER = [
    "DUPLICATE_INDEX_DOCS",
    "NOT_INDEXED",
    "ADD_ALIAS",
    "REMOVE_ALIAS",
    "SET_CANONICAL",
    "CANONICAL_AMBIGUOUS",
    "ORPHANED_INDEX_CONTENT",
    "PATH_REPLACED",
    "MIXED",
]
REASON_DETAIL_MAP = {
    "DUPLICATE_INDEX_DOCS": "Multiple full-text docs share the same checksum.",
    "NOT_INDEXED": "File exists on disk but is missing from the index.",
    "ADD_ALIAS": "Disk path exists that is not in aliases yet.",
    "REMOVE_ALIAS": "Alias path is missing on disk within scanned roots.",
    "SET_CANONICAL": (
        "Canonical path missing; auto-selected via shortest path, then newest mtime, then first."
    ),
    "CANONICAL_AMBIGUOUS": "Legacy state; canonical selection is now automatic.",
    "ORPHANED_INDEX_CONTENT": "Indexed content has no disk paths under scanned roots.",
    "PATH_REPLACED": "Canonical path now points to a different checksum.",
    "MIXED": "Multiple reasons apply; review actions and details.",
}
REASON_ACTION_MAP = {
    "DUPLICATE_INDEX_DOCS": "Blocked: dedupe full-text docs before applying.",
    "NOT_INDEXED": "Optional ingest (checkbox in Apply phase).",
    "ADD_ALIAS": "Apply: add alias path.",
    "REMOVE_ALIAS": "Apply: remove alias path.",
    "SET_CANONICAL": "Apply: set canonical using auto-selection rules.",
    "CANONICAL_AMBIGUOUS": "No longer used; auto-selection applies.",
    "ORPHANED_INDEX_CONTENT": "Optional delete (Destructive checkbox).",
    "PATH_REPLACED": "Manual review; optional retire replaced content.",
    "MIXED": "Review actions list for applyable steps.",
}


def _parse_exts(raw: str) -> set[str]:
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    exts = {part if part.startswith(".") else f".{part}" for part in parts}
    return exts or set(DEFAULT_ALLOWED_EXTENSIONS)


def _split_reasons(reason: str) -> list[str]:
    if not reason:
        return []
    parts = [part.strip() for part in str(reason).split(";") if part.strip()]
    return parts or []


def _map_reason_details(reason: str) -> str:
    parts = _split_reasons(reason)
    if not parts:
        return REASON_DETAIL_MAP["MIXED"]
    mapped = [REASON_DETAIL_MAP.get(part, f"Unknown reason: {part}") for part in parts]
    return "; ".join(dict.fromkeys(mapped))


def _map_reason_actions(reason: str) -> str:
    parts = _split_reasons(reason)
    if not parts:
        return REASON_ACTION_MAP["MIXED"]
    mapped = [REASON_ACTION_MAP.get(part, f"Manual review for {part}") for part in parts]
    return "; ".join(dict.fromkeys(mapped))


def _plan_items_to_rows(items: List[FileResyncPlanItem]) -> List[dict]:
    return [
        {
            "bucket": item.bucket,
            "reason": item.reason,
            "checksum": item.checksum,
            "content_id": item.content_id,
            "indexed_paths": "; ".join(item.indexed_paths),
            "disk_paths": "; ".join(item.disk_paths),
            "actions": ", ".join(sorted({action.type for action in item.actions})),
            "explanation": item.explanation,
            "new_checksum": item.new_checksum,
        }
        for item in items
    ]


def _build_filtered_df(rows: Iterable[dict], buckets: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if buckets:
        df = df.loc[df["bucket"].isin(buckets)]
    df = df.copy()
    df["reason_detail"] = df["reason"].apply(_map_reason_details)
    df["apply_action"] = df["reason"].apply(_map_reason_actions)
    preferred_cols = [
        "bucket",
        "reason",
        "reason_detail",
        "apply_action",
        "checksum",
        "content_id",
        "indexed_paths",
        "disk_paths",
        "actions",
        "new_checksum",
        "explanation",
    ]
    return df[[col for col in preferred_cols if col in df.columns]]


def _format_scan_summary(plan, scan_meta: dict) -> str:
    if plan is None:
        return "No plan available yet. Run a scan to populate results."
    scanned = ", ".join(scan_meta.get("scanned_roots", [])) or "None"
    failed = ", ".join(scan_meta.get("failed_roots", [])) or "None"
    ignored = scan_meta.get("ignored", 0)
    return (
        f"Plan generated with {len(plan.items)} item(s). Buckets: {plan.counts}. "
        f"Roots scanned: {scanned}; failed roots: {failed}; ignored files: {ignored}."
    )


def _write_csv(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(tmp.name, index=False)
    return tmp.name


def _roots_to_df(roots: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"Sync roots": roots}) if roots else pd.DataFrame({"Sync roots": []})


def build_index_viewer_section(*, use_accordion: bool = True) -> None:
    wrapper = gr.Accordion("Index Viewer", open=True) if use_accordion else nullcontext()
    with wrapper:
        files_state = gr.State([])
        page_state = gr.State(0)
        total_pages_state = gr.State(1)
        qdrant_memo_state = gr.State({})
        task_records_state = gr.State([])

        control_row = gr.Row()
        with control_row:
            index_button = gr.Button("Load indexed files", variant="primary")
            refresh_button = gr.Button("Refresh", variant="secondary")

        filter_row = gr.Row()
        with filter_row:
            path_filter = gr.Textbox(label="Filter by path substring", scale=4)
            clear_filter = gr.Button("Reset filter", scale=1)
            missing_only = gr.Checkbox(
                label="Only embedding discrepancies",
                info="Show rows where Qdrant count differs from OpenSearch",
            )

        sort_row = gr.Row()
        with sort_row:
            sort_col = gr.Dropdown(
                label="Sort by",
                choices=INDEX_COLUMNS,
                value="Modified",
                interactive=True,
            )
            sort_dir = gr.Radio(
                ["Ascending", "Descending"],
                label="Order",
                value="Descending",
            )
            show_qdrant = gr.Checkbox(
                label="Show Qdrant counts (slower)",
            )

        pager_row = gr.Row()
        with pager_row:
            page_size = gr.Dropdown(
                label="Results per page",
                choices=PAGE_SIZE_OPTIONS,
                value=PAGE_SIZE_OPTIONS[1],
                interactive=True,
            )
            prev_page = gr.Button("◀ Prev")
            page_info = gr.Markdown("Page 1 of 1")
            next_page = gr.Button("Next ▶")

        index_table = gr.Dataframe(
            headers=INDEX_COLUMNS,
            datatype=[
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "str",
                "number",
                "number",
                "str",
            ],
            row_count=0,
            column_count=(len(INDEX_COLUMNS), "fixed"),
            interactive=False,
        )

        selected_paths = gr.CheckboxGroup(
            label="Selected files (current page)",
            choices=[],
            value=[],
        )

        action_row = gr.Row()
        with action_row:
            open_selected = gr.Button("📂 Open selected")
            show_selected = gr.Button("📁 Show in folder")
            reembed_selected = gr.Button("🧠 Re-embed selected", variant="primary")
            reingest_selected = gr.Button("🔄 Reingest selected")
            delete_selected = gr.Button("🗑️ Delete selected", variant="stop")
            open_all = gr.Button("📂 Open all shown")

        confirm_row = gr.Row()
        with confirm_row:
            bulk_open_confirm = gr.Checkbox(
                label=f"Confirm bulk open/show when > {MAX_BULK_OPEN}",
                value=False,
            )
            delete_confirm = gr.Checkbox(label="Confirm delete selected", value=False)

        action_status = gr.Markdown()
        action_details = gr.JSON()

        gr.Markdown("**Task panel (queued actions)**")
        task_controls = gr.Row()
        with task_controls:
            refresh_tasks = gr.Button("🔄 Refresh task status")
            clear_tasks = gr.Button("🧹 Clear finished")
        task_table = gr.Dataframe(
            headers=["Path", "Task ID", "Action", "State", "Result"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=0,
            column_count=(5, "fixed"),
            interactive=False,
        )

    def _build_index_rows(files: list[dict]) -> pd.DataFrame:
        rows = []
        for file_meta in files:
            modified_at = file_meta.get("modified_at")
            created_at = file_meta.get("created_at")
            size_bytes = file_meta.get("bytes", 0)
            rows.append(
                {
                    "Filename": file_meta.get("filename", ""),
                    "Path": file_meta.get("path", ""),
                    "Filetype": file_meta.get("filetype", ""),
                    "Modified": format_timestamp_ampm(modified_at or ""),
                    "Created": format_timestamp_ampm(created_at or ""),
                    "Indexed": format_timestamp(file_meta.get("indexed_at") or ""),
                    "Size": format_file_size(size_bytes),
                    "OpenSearch Chunks": file_meta.get("num_chunks", 0),
                    "Qdrant Chunks": file_meta.get("qdrant_count", 0),
                    "Checksum": file_meta.get("checksum", ""),
                    "Modified Raw": modified_at,
                    "Created Raw": created_at,
                    "Size Bytes": size_bytes,
                }
            )
        return pd.DataFrame(rows)

    def _apply_index_filters(
        files: list[dict],
        path_value: str,
        only_missing: bool,
        show_qdrant_counts: bool,
        sort_by: str,
        sort_order: str,
        page_size_value: int,
        page_index: int,
        qdrant_memo: dict,
    ) -> tuple[pd.DataFrame, str, int, int, dict]:
        df = _build_index_rows(files)
        if df.empty:
            return df, "Page 1 of 1", 0, 1, qdrant_memo

        if path_value:
            df = df.loc[df["Path"].str.contains(path_value, case=False, na=False)]

        need_counts = only_missing or show_qdrant_counts
        if need_counts and not df.empty:
            checksum_series = df["Checksum"].dropna().astype(str)
            visible_checksums = checksum_series.unique().tolist()
            missing = [cs for cs in visible_checksums if cs and cs not in qdrant_memo]
            if missing:
                qdrant_memo.update(index_viewer_usecase.compute_qdrant_counts(missing))
            df["Qdrant Chunks"] = (
                df["Checksum"].astype(str).map(qdrant_memo).fillna(df["Qdrant Chunks"])
            )

        if only_missing:
            df = df.loc[
                df["OpenSearch Chunks"].fillna(0) != df["Qdrant Chunks"].fillna(0)
            ]

        if sort_by not in df.columns:
            sort_by = "Modified"
        sort_key = sort_by
        if sort_by == "Modified":
            df["_sort_key"] = pd.to_datetime(df["Modified Raw"], errors="coerce")
            sort_key = "_sort_key"
        elif sort_by == "Created":
            df["_sort_key"] = pd.to_datetime(df["Created Raw"], errors="coerce")
            sort_key = "_sort_key"
        elif sort_by == "Size":
            df["_sort_key"] = pd.to_numeric(df["Size Bytes"], errors="coerce")
            sort_key = "_sort_key"
        df = df.sort_values(
            sort_key,
            ascending=(sort_order == "Ascending"),
            na_position="last",
        )

        total_rows = len(df)
        page_size_value = page_size_value or PAGE_SIZE_OPTIONS[1]
        total_pages = max(1, math.ceil(total_rows / page_size_value))
        page_index = max(0, min(page_index, total_pages - 1))
        start = page_index * page_size_value
        end = start + page_size_value
        page_df = df.iloc[start:end].reset_index(drop=True)[INDEX_COLUMNS]
        info = f"Page {page_index + 1} of {total_pages} • {total_rows} file(s)"
        return page_df, info, page_index, total_pages, qdrant_memo

    def load_indexed_files():
        files = index_viewer_usecase.fetch_indexed_files()
        return files, 0, 1, {}

    def update_table(
        files: list[dict],
        path_value: str,
        only_missing: bool,
        show_qdrant_counts: bool,
        sort_by: str,
        sort_order: str,
        page_size_value: int,
        page_index: int,
        qdrant_memo: dict,
    ):
        table, info, page_index, total_pages, qdrant_memo = _apply_index_filters(
            files,
            path_value,
            only_missing,
            show_qdrant_counts,
            sort_by,
            sort_order,
            page_size_value,
            page_index,
            qdrant_memo,
        )
        return table, info, page_index, total_pages, qdrant_memo

    def _extract_paths(df: pd.DataFrame | None) -> list[str]:
        if df is None or getattr(df, "empty", True):
            return []
        if "Path" not in df.columns:
            return []
        return df["Path"].dropna().astype(str).unique().tolist()

    def _sync_selection(table_df: pd.DataFrame, current_selection: list[str]):
        choices = _extract_paths(table_df)
        selected = [path for path in current_selection if path in choices]
        return gr.update(choices=choices, value=selected)

    def _build_task_table(records: list[dict]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame(columns=["Path", "Task ID", "Action", "State", "Result"])
        task_ids = [record["task_id"] for record in records]
        states = fetch_states(task_ids)
        rows = []
        for record in records:
            task_id = record["task_id"]
            state = states.get(task_id, {}).get("state", "UNKNOWN")
            result = states.get(task_id, {}).get("result")
            rows.append(
                {
                    "Path": record.get("path", ""),
                    "Task ID": task_id,
                    "Action": record.get("action", ""),
                    "State": state,
                    "Result": result,
                }
            )
        return pd.DataFrame(rows)

    def _refresh_task_table(records: list[dict]):
        return records, _build_task_table(records)

    def _clear_finished_tasks(records: list[dict]):
        if not records:
            return records, _build_task_table(records), "No queued tasks to clear."
        task_ids = [record["task_id"] for record in records]
        states = fetch_states(task_ids)
        remaining = clear_finished(records, states)
        return remaining, _build_task_table(remaining), "Cleared finished tasks."

    def _open_selected(
        selected: list[str], confirm_bulk: bool
    ) -> tuple[str, dict | None]:
        if not selected:
            return "Select one or more rows first.", None
        if len(selected) > MAX_BULK_OPEN and not confirm_bulk:
            return (
                f"Confirm bulk open to proceed with {len(selected)} file(s).",
                {"blocked": True, "count": len(selected)},
            )
        for path in sorted(selected):
            open_file_local(path)
        return f"Opened {len(selected)} file(s).", {"opened": selected}

    def _show_selected(
        selected: list[str], confirm_bulk: bool
    ) -> tuple[str, dict | None]:
        if not selected:
            return "Select one or more rows first.", None
        if len(selected) > MAX_BULK_OPEN and not confirm_bulk:
            return (
                f"Confirm bulk open to proceed with {len(selected)} file(s).",
                {"blocked": True, "count": len(selected)},
            )
        for path in sorted(selected):
            show_in_folder(path)
        return f"Opened {len(selected)} folder(s).", {"shown": selected}

    def _open_all_shown(
        table_df: pd.DataFrame, confirm_bulk: bool
    ) -> tuple[str, dict | None]:
        paths = _extract_paths(table_df)
        if not paths:
            return "No rows shown.", None
        if len(paths) > MAX_BULK_OPEN and not confirm_bulk:
            return (
                f"Confirm bulk open to proceed with {len(paths)} file(s).",
                {"blocked": True, "count": len(paths)},
            )
        for path in sorted(paths):
            open_file_local(path)
        return f"Opened {len(paths)} file(s).", {"opened": paths}

    def _enqueue_reembed(selected: list[str], records: list[dict]):
        if not selected:
            return records, _build_task_table(records), "Select one or more rows first.", None
        try:
            task_ids = index_viewer_usecase.enqueue_reembed(selected)
        except Exception as exc:  # noqa: BLE001
            return (
                records,
                _build_task_table(records),
                f"Re-embed enqueue failed: {exc}",
                {"error": str(exc)},
            )
        updated = add_records(records, selected, task_ids, action="reembed")
        return (
            updated,
            _build_task_table(updated),
            f"Queued re-embed for {len(task_ids)} file(s).",
            {"action": "reembed", "task_ids": task_ids, "paths": selected},
        )

    def _enqueue_reingest(selected: list[str], records: list[dict]):
        if not selected:
            return records, _build_task_table(records), "Select one or more rows first.", None
        try:
            task_ids = index_viewer_usecase.enqueue_reingest(selected)
        except Exception as exc:  # noqa: BLE001
            return (
                records,
                _build_task_table(records),
                f"Reingest enqueue failed: {exc}",
                {"error": str(exc)},
            )
        updated = add_records(records, selected, task_ids, action="reingest")
        return (
            updated,
            _build_task_table(updated),
            f"Queued reingest for {len(task_ids)} file(s).",
            {"action": "reingest", "task_ids": task_ids, "paths": selected},
        )

    def _enqueue_delete(selected: list[str], confirm: bool, records: list[dict]):
        if not selected:
            return records, _build_task_table(records), "Select one or more rows first.", None
        if not confirm:
            return (
                records,
                _build_task_table(records),
                "Confirm delete to proceed.",
                {"blocked": True, "count": len(selected)},
            )
        try:
            task_ids = index_viewer_usecase.enqueue_delete(selected)
        except Exception as exc:  # noqa: BLE001
            return (
                records,
                _build_task_table(records),
                f"Delete enqueue failed: {exc}",
                {"error": str(exc)},
            )
        updated = add_records(records, selected, task_ids, action="delete")
        return (
            updated,
            _build_task_table(updated),
            f"Queued delete for {len(task_ids)} file(s).",
            {"action": "delete", "task_ids": task_ids, "paths": selected},
        )

    def reset_page():
        return 0

    def clear_path_filter():
        return "", 0

    def go_prev(page_index: int, total_pages: int) -> int:
        return max(0, page_index - 1)

    def go_next(page_index: int, total_pages: int) -> int:
        return min(max(total_pages - 1, 0), page_index + 1)

    index_button.click(
        load_indexed_files,
        outputs=[files_state, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    refresh_button.click(
        load_indexed_files,
        outputs=[files_state, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    path_filter.change(reset_page, outputs=[page_state]).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    clear_filter.click(clear_path_filter, outputs=[path_filter, page_state]).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    missing_only.change(reset_page, outputs=[page_state]).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    show_qdrant.change(reset_page, outputs=[page_state]).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    sort_col.change(reset_page, outputs=[page_state]).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    sort_dir.change(reset_page, outputs=[page_state]).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    page_size.change(reset_page, outputs=[page_state]).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    prev_page.click(go_prev, inputs=[page_state, total_pages_state], outputs=[page_state]).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    next_page.click(go_next, inputs=[page_state, total_pages_state], outputs=[page_state]).then(
        update_table,
        inputs=[
            files_state,
            path_filter,
            missing_only,
            show_qdrant,
            sort_col,
            sort_dir,
            page_size,
            page_state,
            qdrant_memo_state,
        ],
        outputs=[index_table, page_info, page_state, total_pages_state, qdrant_memo_state],
    ).then(
        _sync_selection,
        inputs=[index_table, selected_paths],
        outputs=[selected_paths],
    )

    open_selected.click(
        _open_selected,
        inputs=[selected_paths, bulk_open_confirm],
        outputs=[action_status, action_details],
    )
    show_selected.click(
        _show_selected,
        inputs=[selected_paths, bulk_open_confirm],
        outputs=[action_status, action_details],
    )
    open_all.click(
        _open_all_shown,
        inputs=[index_table, bulk_open_confirm],
        outputs=[action_status, action_details],
    )
    reembed_selected.click(
        _enqueue_reembed,
        inputs=[selected_paths, task_records_state],
        outputs=[task_records_state, task_table, action_status, action_details],
    )
    reingest_selected.click(
        _enqueue_reingest,
        inputs=[selected_paths, task_records_state],
        outputs=[task_records_state, task_table, action_status, action_details],
    )
    delete_selected.click(
        _enqueue_delete,
        inputs=[selected_paths, delete_confirm, task_records_state],
        outputs=[task_records_state, task_table, action_status, action_details],
    )

    refresh_tasks.click(
        _refresh_task_table,
        inputs=[task_records_state],
        outputs=[task_records_state, task_table],
    )
    clear_tasks.click(
        _clear_finished_tasks,
        inputs=[task_records_state],
        outputs=[task_records_state, task_table, action_status],
    )


def build_duplicates_section(*, use_accordion: bool = True) -> None:
    duplicate_selection = gr.State(None)
    wrapper = gr.Accordion("Duplicate Files", open=False) if use_accordion else nullcontext()
    with wrapper:
        dup_button = gr.Button("Load duplicates", variant="primary")
        dup_table = gr.Dataframe(
            headers=["Checksum", "Location", "Filetype", "Created", "Modified"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=0,
            column_count=(5, "fixed"),
            interactive=False,
        )
        delete_button = gr.Button("Delete Selected")
        dup_status = gr.Markdown()

    def load_duplicates():
        response = duplicates_usecase.lookup_duplicates()
        rows = duplicates_usecase.format_duplicate_rows(response)
        trimmed = [
            {
                "Checksum": row.get("Checksum"),
                "Location": row.get("Location"),
                "Filetype": row.get("Filetype"),
                "Created": row.get("Created"),
                "Modified": row.get("Modified"),
            }
            for row in rows
        ]
        return pd.DataFrame(trimmed)

    def select_duplicate(evt: gr.SelectData):
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return row_index

    def delete_selected(selection, df):
        if selection is None or df is None:
            return "Select a duplicate row first."
        try:
            row = df.iloc[int(selection)]
        except Exception:
            return "Unable to read selection."
        path = row.get("Location") if hasattr(row, "get") else row["Location"]
        if not path:
            return "Selected row has no path."
        task_ids = index_viewer_usecase.enqueue_delete([path])
        return f"Queued deletion for {path}. Task IDs: {', '.join(task_ids)}"

    dup_button.click(load_duplicates, outputs=[dup_table])
    dup_table.select(select_duplicate, outputs=[duplicate_selection])
    delete_button.click(delete_selected, inputs=[duplicate_selection, dup_table], outputs=[dup_status])


def build_watchlist_section(*, use_accordion: bool = True) -> None:
    wrapper = gr.Accordion("Watchlist", open=False) if use_accordion else nullcontext()
    with wrapper:
        watchlist_prefix = gr.Textbox(label="Watchlist prefix")
        add_watchlist = gr.Button("Add to Watchlist")
        watchlist_status = gr.Markdown()

    def add_prefix(prefix: str) -> str:
        if not prefix:
            return "Enter a prefix to add."
        added = watchlist_usecase.add_prefix(prefix)
        return "Prefix added." if added else "Prefix already exists or failed."

    add_watchlist.click(add_prefix, inputs=[watchlist_prefix], outputs=[watchlist_status])


def build_file_resync_section(*, use_accordion: bool = True) -> None:
    resync_roots = gr.State([DEFAULT_ROOT] if DEFAULT_ROOT else [])
    resync_plan = gr.State(None)
    resync_scan_meta = gr.State({})
    resync_rows = gr.State([])
    resync_filtered = gr.State(pd.DataFrame())
    wrapper = gr.Accordion("File Resync", open=False) if use_accordion else nullcontext()
    with wrapper:
        gr.Markdown(
            """
            **Phase A: Scan & Plan** (dry run only)
            - Builds a plan without changing data.

            **Phase B: Apply SAFE actions**
            - Canonical/alias updates that are unambiguous.
            - Optional ingestion of clearly missing files.
            - No deletions.

            **Phase C: Apply destructive actions**
            - REVIEW + SAFE buckets.
            - Optional orphan cleanup and retire-on-replace.
            - Use with care; vectors/chunks/full-text may be deleted.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                root_input = gr.Textbox(label="Add root", placeholder="/path/to/root")
            with gr.Column(scale=1):
                add_root_button = gr.Button("Add root", variant="secondary")
            with gr.Column(scale=1):
                clear_roots_button = gr.Button("Clear roots", variant="secondary")

        root_status = gr.Markdown()
        roots_table = gr.Dataframe(
            value=_roots_to_df([DEFAULT_ROOT] if DEFAULT_ROOT else []),
            headers=["Sync roots"],
            datatype=["str"],
            row_count=0,
            column_count=(1, "fixed"),
            interactive=False,
        )
        ext_input = gr.Textbox(
            label="Allowed extensions (comma-separated)",
            value=", ".join(sorted(DEFAULT_ALLOWED_EXTENSIONS)),
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Step 1: Scan & Plan**")
                scan_button = gr.Button("Scan & Plan", variant="primary")
            with gr.Column():
                gr.Markdown("**Step 2: Apply SAFE actions**")
                ingest_missing_safe = gr.Checkbox(
                    label="Ingest missing (SAFE phase)", value=False
                )
                apply_safe_button = gr.Button(
                    "Apply SAFE actions",
                    interactive=False,
                )
            with gr.Column():
                gr.Markdown("**Step 3: Apply destructive actions**")
                ingest_missing_destructive = gr.Checkbox(
                    label="Ingest missing (Destructive phase)", value=False
                )
                delete_orphaned = gr.Checkbox(
                    label="Delete orphaned content (Destructive)", value=False
                )
                retire_replaced = gr.Checkbox(
                    label="Retire replaced content (Destructive)", value=False
                )
                apply_destructive_button = gr.Button(
                    "Apply destructive actions",
                    interactive=False,
                )

        scan_summary = gr.Markdown()
        apply_summary = gr.Markdown()

        bucket_filters = gr.CheckboxGroup(
            label="Filter by bucket",
            choices=["SAFE", "REVIEW", "BLOCKED", "INFO"],
            value=["SAFE", "REVIEW", "BLOCKED", "INFO"],
        )
        plan_table = gr.Dataframe(
            headers=[
                "bucket",
                "reason",
                "reason_detail",
                "apply_action",
                "checksum",
                "content_id",
                "indexed_paths",
                "disk_paths",
                "actions",
                "new_checksum",
                "explanation",
            ],
            row_count=0,
            column_count=(11, "fixed"),
            interactive=False,
        )
        export_button = gr.DownloadButton("Export CSV", visible=False)
        apply_result = gr.JSON()

        reason_map_rows = [
            {
                "Reason": reason,
                "Meaning": REASON_DETAIL_MAP.get(reason, ""),
                "Apply action": REASON_ACTION_MAP.get(reason, ""),
            }
            for reason in REASON_ORDER
        ]
        gr.Dataframe(
            value=pd.DataFrame(reason_map_rows),
            headers=["Reason", "Meaning", "Apply action"],
            datatype=["str", "str", "str"],
            interactive=False,
            label="Reason Map (Mismatch -> Apply Action)",
        )

    def _add_root(roots: list[str], new_root: str):
        if not new_root:
            return roots, _roots_to_df(roots), "Enter a root path to add."
        entries = [part.strip() for part in new_root.split(",") if part.strip()]
        merged = list(dict.fromkeys(roots + entries))
        message = f"Added {len(merged) - len(roots)} root(s)."
        return merged, _roots_to_df(merged), message

    def _clear_roots():
        return [], _roots_to_df([]), "Cleared all roots."

    def _scan_plan(roots: list[str], raw_exts: str, retire_replaced_flag: bool, buckets: list[str]):
        if not roots:
            empty_df = pd.DataFrame()
            return (
                None,
                {},
                [],
                empty_df,
                "Please provide at least one root to scan.",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(visible=False),
            )
        try:
            exts = sorted(_parse_exts(raw_exts))
            plan, scan_meta = file_resync_usecase.scan_and_plan(
                FileResyncScanRequest(roots=roots, allowed_extensions=exts),
                retire_replaced_content=retire_replaced_flag,
            )
        except Exception as exc:  # noqa: BLE001
            empty_df = pd.DataFrame()
            return (
                None,
                {},
                [],
                empty_df,
                f"Scan failed: {exc}",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(visible=False),
            )

        rows = _plan_items_to_rows(plan.items)
        filtered_df = _build_filtered_df(rows, buckets)
        csv_path = _write_csv(filtered_df)
        return (
            plan,
            scan_meta,
            rows,
            filtered_df,
            _format_scan_summary(plan, scan_meta),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=csv_path, visible=bool(csv_path)),
        )

    def _filter_table(rows: list[dict], buckets: list[str]):
        filtered_df = _build_filtered_df(rows, buckets)
        csv_path = _write_csv(filtered_df)
        return filtered_df, filtered_df, gr.update(value=csv_path, visible=bool(csv_path))

    def _apply_actions(
        plan,
        ingest_missing: bool,
        apply_safe_only: bool,
        delete_orphaned_flag: bool,
        retire_replaced_flag: bool,
    ):
        if plan is None:
            return "No plan available. Run a scan first.", None
        try:
            result = file_resync_usecase.apply_plan(
                FileResyncApplyRequest(
                    items=plan.items,
                    ingest_missing=ingest_missing,
                    apply_safe_only=apply_safe_only,
                    delete_orphaned=delete_orphaned_flag,
                    retire_replaced_content=retire_replaced_flag,
                )
            )
        except Exception as exc:  # noqa: BLE001
            return f"Apply failed: {exc}", None
        summary = "Actions completed." if not result.errors else "Actions completed with warnings."
        return summary, {
            "ingested": result.ingested,
            "updated_fulltext": result.updated_fulltext,
            "updated_chunks": result.updated_chunks,
            "updated_qdrant": result.updated_qdrant,
            "deleted_checksums": result.deleted_checksums,
            "errors": result.errors,
        }

    add_root_button.click(
        _add_root,
        inputs=[resync_roots, root_input],
        outputs=[resync_roots, roots_table, root_status],
    )
    clear_roots_button.click(
        _clear_roots,
        outputs=[resync_roots, roots_table, root_status],
    )

    scan_button.click(
        _scan_plan,
        inputs=[resync_roots, ext_input, retire_replaced, bucket_filters],
        outputs=[
            resync_plan,
            resync_scan_meta,
            resync_rows,
            plan_table,
            scan_summary,
            apply_safe_button,
            apply_destructive_button,
            export_button,
        ],
    )

    bucket_filters.change(
        _filter_table,
        inputs=[resync_rows, bucket_filters],
        outputs=[plan_table, resync_filtered, export_button],
    )

    apply_safe_button.click(
        _apply_actions,
        inputs=[resync_plan, ingest_missing_safe, gr.State(True), gr.State(False), gr.State(False)],
        outputs=[apply_summary, apply_result],
    )
    apply_destructive_button.click(
        _apply_actions,
        inputs=[
            resync_plan,
            ingest_missing_destructive,
            gr.State(False),
            delete_orphaned,
            retire_replaced,
        ],
        outputs=[apply_summary, apply_result],
    )

# ----------------------------
# Worker emergency env config
# ----------------------------
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
QUEUE_NAMES = os.getenv("CELERY_QUEUES", "ingest,celery").split(",")
DEFAULT_TASK = os.getenv("CELERY_DEFAULT_TASK", "tasks.ingest_document")
COMPOSE_DIR = Path(os.getenv("COMPOSE_DIR", Path.cwd()))
COMPOSE_PROJECT = os.getenv("COMPOSE_PROJECT", "document_qa")
CELERY_SERVICE = os.getenv("CELERY_SERVICE", "celery")


def build_maintenance_tab() -> None:
    build_index_viewer_section()
    build_duplicates_section()
    build_watchlist_section()
    build_file_resync_section()

    celery_app = Celery(broker=BROKER_URL, backend=RESULT_BACKEND)
    broker_client = redis.Redis.from_url(BROKER_URL, decode_responses=True)
    result_client = redis.Redis.from_url(RESULT_BACKEND, decode_responses=True)

    with gr.Accordion("Admin & Worker", open=False):
        queue_names = gr.Textbox(
            label="Queues to purge",
            value=",".join(name.strip() for name in QUEUE_NAMES if name.strip()),
        )
        clear_cache = gr.Button("Clear Cache")
        purge_queues = gr.Button("Purge Queues")
        admin_status = gr.Markdown()

    def clear_all():
        result = admin_usecase.clear_cache()
        return f"Cache clear result: {result}"

    def purge(queue_text: str) -> str:
        queues = [name.strip() for name in queue_text.split(",") if name.strip()]
        result = admin_usecase.purge_queues(queues)
        return f"Purge result: {result}"

    clear_cache.click(clear_all, outputs=[admin_status])
    purge_queues.click(purge, inputs=[queue_names], outputs=[admin_status])

    with gr.Accordion("Worker Emergency Operations", open=False):
        status_button = gr.Button("Refresh worker status", variant="primary")
        queue_status = gr.Dataframe(
            headers=["Queue", "Length"],
            datatype=["str", "number"],
            row_count=0,
            column_count=(2, "fixed"),
            interactive=False,
        )
        counts_status = gr.Markdown()
        alerts_status = gr.Markdown()

        def load_worker_status():
            status = worker_emergency_usecase.load_status(
                celery_app,
                broker_client,
                result_client,
                [name.strip() for name in QUEUE_NAMES if name.strip()],
            )
            queue_rows = [
                {"Queue": name, "Length": status.queue_lengths.get(name, -1)}
                for name in status.queue_lengths.keys()
            ]
            counts = (
                f"**Active:** {status.active} | "
                f"**Reserved:** {status.reserved} | "
                f"**Scheduled:** {status.scheduled}"
            )
            alerts = []
            if not status.broker_ok:
                alerts.append(f"Broker connectivity failed: {status.broker_error}")
            if not status.result_ok:
                alerts.append(f"Result backend connectivity failed: {status.result_error}")
            if status.celery_error:
                alerts.append(f"Celery inspection failed: {status.celery_error}")
            alert_text = "\n".join(f"- {entry}" for entry in alerts) if alerts else "All checks OK."
            return pd.DataFrame(queue_rows), counts, alert_text

        status_button.click(load_worker_status, outputs=[queue_status, counts_status, alerts_status])

        with gr.Accordion("Pause/Resume Queue Consumption", open=False):
            queue_name = gr.Textbox(label="Queue name", value=QUEUE_NAMES[0] if QUEUE_NAMES else "ingest")
            pause_button = gr.Button("Cancel consumer (pause)")
            resume_button = gr.Button("Add consumer (resume)")
            pause_status = gr.Markdown()

            def pause_queue(name: str) -> str:
                if not name:
                    return "Enter a queue name."
                result = celery_app.control.cancel_consumer(name)
                return f"Cancelled consumer for queue {name}. Result: {result}"

            def resume_queue(name: str) -> str:
                if not name:
                    return "Enter a queue name."
                result = celery_app.control.add_consumer(name)
                return f"Added consumer for queue {name}. Result: {result}"

            pause_button.click(pause_queue, inputs=[queue_name], outputs=[pause_status])
            resume_button.click(resume_queue, inputs=[queue_name], outputs=[pause_status])

        with gr.Accordion("Autoscale Worker Pool", open=False):
            autoscale_max = gr.Number(label="Max processes", value=8, precision=0)
            autoscale_min = gr.Number(label="Min processes", value=0, precision=0)
            autoscale_button = gr.Button("Apply autoscale")
            autoscale_status = gr.Markdown()

            def apply_autoscale(max_val: float, min_val: float) -> str:
                if max_val is None or min_val is None:
                    return "Enter max and min values."
                max_int = int(max_val)
                min_int = int(min_val)
                result = celery_app.control.autoscale(max_int, min_int)
                return f"Autoscale applied: max={max_int}, min={min_int}. Result: {result}"

            autoscale_button.click(
                apply_autoscale,
                inputs=[autoscale_max, autoscale_min],
                outputs=[autoscale_status],
            )

        with gr.Accordion("Stop/Start Worker via Docker Compose", open=False):
            stop_button = gr.Button("Stop worker container", variant="primary")
            start_button = gr.Button("Start worker container")
            compose_status = gr.Markdown()

            def stop_worker() -> str:
                result = worker_emergency_usecase.run_compose(
                    ["stop", CELERY_SERVICE],
                    compose_dir=COMPOSE_DIR,
                    compose_project=COMPOSE_PROJECT,
                )
                return result.stdout or result.stderr or "(no output)"

            def start_worker() -> str:
                result = worker_emergency_usecase.run_compose(
                    ["up", "-d", CELERY_SERVICE],
                    compose_dir=COMPOSE_DIR,
                    compose_project=COMPOSE_PROJECT,
                )
                return result.stdout or result.stderr or "(no output)"

            stop_button.click(stop_worker, outputs=[compose_status])
            start_button.click(start_worker, outputs=[compose_status])

        with gr.Accordion("Rate Limit & Revoke Active Tasks", open=False):
            rate_button = gr.Button("Apply 0/m rate limit")
            revoke_term = gr.Button("Revoke active (SIGTERM)")
            revoke_kill = gr.Button("Revoke active (SIGKILL)")
            revoke_status = gr.Markdown()

            def rate_limit() -> str:
                worker_emergency_usecase.rate_limit_zero(celery_app, DEFAULT_TASK)
                return f"Rate limit applied: {DEFAULT_TASK} set to 0/m."

            def revoke_active(signal: str) -> str:
                count = worker_emergency_usecase.revoke_all_active(celery_app, signal=signal)
                return f"Revoked {count} active task(s) with {signal}."

            rate_button.click(rate_limit, outputs=[revoke_status])
            revoke_term.click(lambda: revoke_active("SIGTERM"), outputs=[revoke_status])
            revoke_kill.click(lambda: revoke_active("SIGKILL"), outputs=[revoke_status])

        with gr.Accordion("Purge Queues", open=False):
            purge_text = gr.Textbox(
                label="Queues to purge (comma-separated)",
                value=",".join(name.strip() for name in QUEUE_NAMES if name.strip()),
            )
            purge_button = gr.Button("Purge queues")
            purge_status = gr.Markdown()

            def purge_selected(queue_text: str) -> str:
                queues = [name.strip() for name in queue_text.split(",") if name.strip()]
                result = worker_emergency_usecase.purge_queues(broker_client, queues)
                return f"Purge result: {result}"

            purge_button.click(purge_selected, inputs=[purge_text], outputs=[purge_status])

        with gr.Accordion("Flush Redis DBs", open=False):
            flush_broker = gr.Button("Flush broker DB (DB 0)")
            flush_results = gr.Button("Flush result DB (DB 1)")
            flush_status = gr.Markdown()

            def flush_broker_db() -> str:
                broker_client.flushdb()
                return "Broker FLUSHDB completed (DB 0)."

            def flush_results_db() -> str:
                result_client.flushdb()
                return "Result FLUSHDB completed (DB 1)."

            flush_broker.click(flush_broker_db, outputs=[flush_status])
            flush_results.click(flush_results_db, outputs=[flush_status])
