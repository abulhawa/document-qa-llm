"""Knowledge base maintenance tab."""

from __future__ import annotations

import math

import gradio as gr
import pandas as pd

from app.schemas import FileResyncScanRequest
from app.usecases import (
    admin_usecase,
    duplicates_usecase,
    file_resync_usecase,
    index_viewer_usecase,
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


def build_maintenance_tab() -> None:
    duplicate_selection = gr.State(None)

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
