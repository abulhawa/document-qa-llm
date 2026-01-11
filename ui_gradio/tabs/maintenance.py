"""Knowledge base maintenance tab."""

from __future__ import annotations

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


def build_maintenance_tab() -> None:
    duplicate_selection = gr.State(None)

    with gr.Accordion("Index Viewer", open=True):
        index_button = gr.Button("Load indexed files", variant="primary")
        index_table = gr.Dataframe(
            headers=["Path", "Filetype", "Modified", "Created"],
            datatype=["str", "str", "str", "str"],
            row_count=0,
            col_count=(4, "fixed"),
            interactive=False,
        )

    def load_indexed():
        files = index_viewer_usecase.fetch_indexed_files()
        rows = [
            {
                "Path": file_meta.get("path"),
                "Filetype": file_meta.get("filetype"),
                "Modified": file_meta.get("modified_at"),
                "Created": file_meta.get("created_at"),
            }
            for file_meta in files
        ]
        return pd.DataFrame(rows)

    index_button.click(load_indexed, outputs=[index_table])

    with gr.Accordion("Duplicate Files", open=False):
        dup_button = gr.Button("Load duplicates", variant="primary")
        dup_table = gr.Dataframe(
            headers=["Checksum", "Location", "Filetype", "Created", "Modified"],
            datatype=["str", "str", "str", "str", "str"],
            row_count=0,
            col_count=(5, "fixed"),
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
