"""Watchlist tab for tracking folder indexing progress."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import gradio as gr
import pandas as pd

from config import WATCH_INVENTORY_INDEX, WATCHLIST_INDEX
from app.usecases import watchlist_usecase
from ui.task_status import add_records
from utils.opensearch.indexes import ensure_index_exists


def build_watchlist_tab() -> None:
    ensure_index_exists(index=WATCH_INVENTORY_INDEX)
    ensure_index_exists(index=WATCHLIST_INDEX)

    initial_prefixes = watchlist_usecase.load_watchlist_prefixes()
    watchlist_state = gr.State(initial_prefixes)
    task_state = gr.State([])
    initial_message = "" if initial_prefixes else "No tracked folders yet. Add one above."

    def _format_task_rows(records: list[dict[str, Any]] | None) -> pd.DataFrame:
        rows = []
        for record in records or []:
            timestamp = record.get("enqueued_at")
            if timestamp:
                readable = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            else:
                readable = ""
            rows.append(
                {
                    "Path": record.get("path"),
                    "Task ID": record.get("task_id"),
                    "Action": record.get("action"),
                    "Enqueued": readable,
                }
            )
        return pd.DataFrame(rows)

    def _build_status_frame(prefixes: list[str]) -> pd.DataFrame:
        rows = []
        for pref in prefixes:
            status = watchlist_usecase.get_status(pref)
            meta = status.meta or {}
            rows.append(
                {
                    "Prefix": pref,
                    "Unindexed": status.remaining,
                    "Quick wins": status.quick_wins,
                    "Indexed": status.indexed,
                    "Total": status.total,
                    "Progress": f"{status.percent_indexed:.1%}" if status.total else "0%",
                    "Last refreshed": meta.get("last_refreshed") or "",
                    "Last scan": meta.get("last_scanned") or "",
                }
            )
        return pd.DataFrame(rows)

    def _build_preview_frame(prefix: str | None) -> pd.DataFrame:
        if not prefix:
            return pd.DataFrame({"Unindexed preview": []})
        status = watchlist_usecase.get_status(prefix)
        preview = status.preview or []
        if not preview:
            preview = watchlist_usecase.list_unindexed_paths(prefix, limit=10)
        return pd.DataFrame({"Unindexed preview": preview})

    def _load_watchlist() -> tuple[list[str], gr.update, pd.DataFrame, pd.DataFrame, str]:
        prefixes = watchlist_usecase.load_watchlist_prefixes()
        message = "" if prefixes else "No tracked folders yet. Add one above."
        selected = prefixes[0] if prefixes else None
        return (
            prefixes,
            gr.update(choices=prefixes, value=selected),
            _build_status_frame(prefixes),
            _build_preview_frame(selected),
            message,
        )

    def _add_prefix(
        prefix: str, prefixes: list[str]
    ) -> tuple[list[str], gr.update, pd.DataFrame, pd.DataFrame, str]:
        cleaned = (prefix or "").strip()
        if not cleaned:
            return (
                prefixes,
                gr.update(),
                _build_status_frame(prefixes),
                _build_preview_frame(prefixes[0] if prefixes else None),
                "Enter a valid path prefix.",
            )
        if cleaned in prefixes:
            return (
                prefixes,
                gr.update(value=cleaned),
                _build_status_frame(prefixes),
                _build_preview_frame(cleaned),
                "Already tracked.",
            )
        if watchlist_usecase.add_prefix(cleaned):
            updated = prefixes + [cleaned]
            return (
                updated,
                gr.update(choices=updated, value=cleaned),
                _build_status_frame(updated),
                _build_preview_frame(cleaned),
                "Added to watchlist.",
            )
        return (
            prefixes,
            gr.update(),
            _build_status_frame(prefixes),
            _build_preview_frame(prefixes[0] if prefixes else None),
            "Could not add prefix.",
        )

    def _refresh_all(prefixes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        if not prefixes:
            return _build_status_frame(prefixes), _build_preview_frame(None), "No tracked folders to refresh."
        result = watchlist_usecase.refresh_all_status(prefixes)
        message = (
            "Refreshed all. Imported known files: "
            f"{result.total_fulltext}, updated chunk counts: {result.total_chunks}."
        )
        if result.errors:
            message += "\nSome folders could not be refreshed."
        return _build_status_frame(prefixes), _build_preview_frame(prefixes[0]), message

    def _refresh_status(prefix: str, prefixes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        if not prefix:
            return _build_status_frame(prefixes), _build_preview_frame(None), "Select a folder to refresh."
        result = watchlist_usecase.refresh_status(prefix)
        message = (
            "Imported known files: "
            f"{result.imported}, updated chunk counts: {result.chunk_counts}. "
            f"Unindexed now: {result.remaining}."
        )
        if result.errors:
            message += "\nSome refresh steps failed."
        return _build_status_frame(prefixes), _build_preview_frame(prefix), message

    def _scan_disk(prefix: str, prefixes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        if not prefix:
            return _build_status_frame(prefixes), _build_preview_frame(None), "Select a folder to scan."
        result = watchlist_usecase.scan_folder(prefix)
        message = f"Scan complete. Found: {result.found}, marked missing: {result.marked_missing}."
        if result.errors:
            message += "\nSome scan steps failed."
        return _build_status_frame(prefixes), _build_preview_frame(prefix), message

    def _compute_remaining(prefix: str, prefixes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        if not prefix:
            return _build_status_frame(prefixes), _build_preview_frame(None), "Select a folder to check."
        remaining, errors = watchlist_usecase.get_remaining(prefix)
        message = f"Remaining unindexed files: {remaining}."
        if errors:
            message += "\nUnable to compute remaining files."
        return _build_status_frame(prefixes), _build_preview_frame(prefix), message

    def _import_known(prefix: str, prefixes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        if not prefix:
            return _build_status_frame(prefixes), _build_preview_frame(None), "Select a folder to import."
        count, errors = watchlist_usecase.import_known_files(prefix)
        message = f"Imported {count} files from index."
        if errors:
            message += "\nSome files could not be imported."
        return _build_status_frame(prefixes), _build_preview_frame(prefix), message

    def _import_chunks(prefix: str, prefixes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        if not prefix:
            return _build_status_frame(prefixes), _build_preview_frame(None), "Select a folder to import."
        count, errors = watchlist_usecase.import_chunk_counts(prefix)
        message = f"Updated chunk counts for {count} files."
        if errors:
            message += "\nSome chunk counts could not be updated."
        return _build_status_frame(prefixes), _build_preview_frame(prefix), message

    def _sync_indices(prefix: str, prefixes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        if not prefix:
            return _build_status_frame(prefixes), _build_preview_frame(None), "Select a folder to sync."
        errors = watchlist_usecase.sync_from_indices(prefix)
        message = "Synced from indices."
        if errors:
            message += "\nSome sync steps failed."
        return _build_status_frame(prefixes), _build_preview_frame(prefix), message

    def _reingest_changed(
        prefix: str,
        prefixes: list[str],
        records: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame, str]:
        if not prefix:
            return (
                _build_status_frame(prefixes),
                records,
                _format_task_rows(records),
                _build_preview_frame(None),
                "Select a folder to re-ingest.",
            )
        re_files = watchlist_usecase.list_reingest_paths(prefix, limit=2000)
        if not re_files:
            return (
                _build_status_frame(prefixes),
                records,
                _format_task_rows(records),
                _build_preview_frame(prefix),
                "No changed files found under this folder.",
            )
        result = watchlist_usecase.queue_ingest(re_files, mode="reingest")
        updated_records = add_records(records, re_files, result.task_ids, action="reingest")
        message = f"Queued {len(result.task_ids)} file(s) for re-ingestion."
        if result.errors:
            message += "\nSome re-ingest steps failed."
        return (
            _build_status_frame(prefixes),
            updated_records,
            _format_task_rows(updated_records),
            _build_preview_frame(prefix),
            message,
        )

    def _remove_prefix(
        prefix: str, prefixes: list[str]
    ) -> tuple[list[str], gr.update, pd.DataFrame, pd.DataFrame, str]:
        if not prefix:
            return (
                prefixes,
                gr.update(),
                _build_status_frame(prefixes),
                _build_preview_frame(prefixes[0] if prefixes else None),
                "Select a folder to remove.",
            )
        if watchlist_usecase.remove_prefix(prefix):
            updated = [item for item in prefixes if item != prefix]
            message = "Folder removed from watchlist."
            return (
                updated,
                gr.update(choices=updated, value=updated[0] if updated else None),
                _build_status_frame(updated),
                _build_preview_frame(updated[0] if updated else None),
                message,
            )
        return (
            prefixes,
            gr.update(),
            _build_status_frame(prefixes),
            _build_preview_frame(prefix),
            "Failed to remove. Try again.",
        )

    def _ingest_remaining(
        prefix: str,
        prefixes: list[str],
        records: list[dict[str, Any]],
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame, str]:
        if not prefix:
            return (
                _build_status_frame(prefixes),
                records,
                _format_task_rows(records),
                _build_preview_frame(None),
                "Select a folder to ingest.",
            )
        sync_errors = watchlist_usecase.sync_from_indices(prefix)
        paths = watchlist_usecase.list_unindexed_paths(prefix, limit=2000)
        if not paths:
            return (
                _build_status_frame(prefixes),
                records,
                _format_task_rows(records),
                _build_preview_frame(prefix),
                "No unindexed files found under this folder.",
            )
        result = watchlist_usecase.queue_ingest(paths, mode="ingest")
        updated_records = add_records(records, paths, result.task_ids, action="ingest")
        message = f"Queued {len(result.task_ids)} file(s) for ingestion."
        if sync_errors or result.errors:
            message += "\nSome ingestion steps failed."
        return (
            _build_status_frame(prefixes),
            updated_records,
            _format_task_rows(updated_records),
            _build_preview_frame(prefix),
            message,
        )

    gr.Markdown(
        "Track folders and see how many files still need indexing. "
        "Use actions to sync, scan, and queue ingest or re-ingest jobs."
    )

    with gr.Row():
        prefix_input = gr.Textbox(label="Path prefix (folder)", placeholder="/data/docs")
        add_button = gr.Button("Add folder", variant="primary")
        reload_button = gr.Button("Reload list")
        refresh_all_button = gr.Button("Refresh all status")

    action_status = gr.Markdown(value=initial_message)

    status_table = gr.Dataframe(
        headers=[
            "Prefix",
            "Unindexed",
            "Quick wins",
            "Indexed",
            "Total",
            "Progress",
            "Last refreshed",
            "Last scan",
        ],
        datatype=["str", "number", "number", "number", "number", "str", "str", "str"],
        row_count=0,
        column_count=(8, "fixed"),
        interactive=False,
        value=_build_status_frame(initial_prefixes),
    )

    with gr.Row():
        prefix_select = gr.Dropdown(
            choices=initial_prefixes,
            label="Tracked folder",
            interactive=True,
            allow_custom_value=False,
            value=initial_prefixes[0] if initial_prefixes else None,
        )

    with gr.Row():
        refresh_button = gr.Button("Refresh status")
        scan_button = gr.Button("Scan disk for changes")
        remaining_button = gr.Button("Compute remaining")
        import_known_button = gr.Button("Import known files")
        import_chunks_button = gr.Button("Import chunk counts")
        sync_button = gr.Button("Sync from indices")
        ingest_button = gr.Button("Ingest remaining")
        reingest_button = gr.Button("Reingest changed")
        remove_button = gr.Button("Remove from watchlist", variant="stop")

    preview_table = gr.Dataframe(
        headers=["Unindexed preview"],
        datatype=["str"],
        row_count=0,
        column_count=(1, "fixed"),
        interactive=False,
        value=_build_preview_frame(initial_prefixes[0] if initial_prefixes else None),
    )

    task_table = gr.Dataframe(
        headers=["Path", "Task ID", "Action", "Enqueued"],
        datatype=["str", "str", "str", "str"],
        row_count=0,
        column_count=(4, "fixed"),
        interactive=False,
    )

    reload_button.click(
        _load_watchlist,
        outputs=[watchlist_state, prefix_select, status_table, preview_table, action_status],
    )
    add_button.click(
        _add_prefix,
        inputs=[prefix_input, watchlist_state],
        outputs=[watchlist_state, prefix_select, status_table, preview_table, action_status],
    )
    refresh_all_button.click(
        _refresh_all,
        inputs=[watchlist_state],
        outputs=[status_table, preview_table, action_status],
    )
    refresh_button.click(
        _refresh_status,
        inputs=[prefix_select, watchlist_state],
        outputs=[status_table, preview_table, action_status],
    )
    scan_button.click(
        _scan_disk,
        inputs=[prefix_select, watchlist_state],
        outputs=[status_table, preview_table, action_status],
    )
    remaining_button.click(
        _compute_remaining,
        inputs=[prefix_select, watchlist_state],
        outputs=[status_table, preview_table, action_status],
    )
    import_known_button.click(
        _import_known,
        inputs=[prefix_select, watchlist_state],
        outputs=[status_table, preview_table, action_status],
    )
    import_chunks_button.click(
        _import_chunks,
        inputs=[prefix_select, watchlist_state],
        outputs=[status_table, preview_table, action_status],
    )
    sync_button.click(
        _sync_indices,
        inputs=[prefix_select, watchlist_state],
        outputs=[status_table, preview_table, action_status],
    )
    ingest_button.click(
        _ingest_remaining,
        inputs=[prefix_select, watchlist_state, task_state],
        outputs=[status_table, task_state, task_table, preview_table, action_status],
    )
    reingest_button.click(
        _reingest_changed,
        inputs=[prefix_select, watchlist_state, task_state],
        outputs=[status_table, task_state, task_table, preview_table, action_status],
    )
    remove_button.click(
        _remove_prefix,
        inputs=[prefix_select, watchlist_state],
        outputs=[watchlist_state, prefix_select, status_table, preview_table, action_status],
    )

    prefix_select.change(
        _build_preview_frame,
        inputs=[prefix_select],
        outputs=[preview_table],
    )
