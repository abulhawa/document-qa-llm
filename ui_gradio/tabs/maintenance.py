"""Knowledge base maintenance tab."""

from __future__ import annotations

import math

import os
from pathlib import Path

from celery import Celery
import gradio as gr
import pandas as pd
import redis

from app.schemas import FileResyncScanRequest
from app.usecases import (
    admin_usecase,
    duplicates_usecase,
    file_resync_usecase,
    index_viewer_usecase,
    worker_emergency_usecase,
    watchlist_usecase,
)
from utils.file_utils import format_file_size
from utils.time_utils import format_timestamp, format_timestamp_ampm


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
    duplicate_selection = gr.State(None)
    celery_app = Celery(broker=BROKER_URL, backend=RESULT_BACKEND)
    broker_client = redis.Redis.from_url(BROKER_URL, decode_responses=True)
    result_client = redis.Redis.from_url(RESULT_BACKEND, decode_responses=True)

    with gr.Accordion("Index Viewer", open=True):
        files_state = gr.State([])
        page_state = gr.State(0)
        total_pages_state = gr.State(1)
        qdrant_memo_state = gr.State({})

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
    )

    with gr.Accordion("Duplicate Files", open=False):
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

    with gr.Accordion("Watchlist & File Resync", open=False):
        watchlist_prefix = gr.Textbox(label="Watchlist prefix")
        add_watchlist = gr.Button("Add to Watchlist")
        resync_button = gr.Button("Scan & Plan Resync")
        watchlist_status = gr.Markdown()

    def add_prefix(prefix: str) -> str:
        if not prefix:
            return "Enter a prefix to add."
        added = watchlist_usecase.add_prefix(prefix)
        return "Prefix added." if added else "Prefix already exists or failed."

    def scan_resync(prefix: str) -> str:
        if not prefix:
            return "Enter a prefix to scan."
        response, meta = file_resync_usecase.scan_and_plan(
            FileResyncScanRequest(roots=[prefix])
        )
        return (
            f"Plan generated with {len(response.items)} items. "
            f"Buckets: {response.counts}. "
            f"Scanned: {meta.get('scanned_roots', [])}."
        )

    add_watchlist.click(add_prefix, inputs=[watchlist_prefix], outputs=[watchlist_status])
    resync_button.click(scan_resync, inputs=[watchlist_prefix], outputs=[watchlist_status])

    with gr.Accordion("Admin & Worker", open=False):
        queue_names = gr.Textbox(
            label="Queues to purge",
            value="ingest,celery",
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
