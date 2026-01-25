"""Intelligent search tab."""

from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import gradio as gr

from app.gradio_utils import (
    DEFAULT_FILETYPES,
    normalize_date_input,
    render_search_snippet,
    search_hits_to_rows,
)
from app.schemas import IngestRequest, SearchRequest
from app.usecases import ingest_usecase, search_usecase
from utils.file_utils import open_file_local, show_in_folder


RESULT_HEADERS = ["Filename", "Path", "Score", "Date"]
PAGE_SIZE_OPTIONS = [5, 25, 50, 100]


def build_search_tab() -> None:
    hits_state = gr.State([])
    missing_state = gr.State([])
    total_pages_state = gr.State(1)
    selected_index_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            query = gr.Textbox(label="Query", placeholder="Search documents...")
            path_contains = gr.Textbox(label="Path contains")
            sort = gr.Dropdown(
                choices=["relevance", "modified"],
                value="relevance",
                label="Sort",
            )
            page_size = gr.Dropdown(
                choices=PAGE_SIZE_OPTIONS,
                value=PAGE_SIZE_OPTIONS[1],
                label="Page size",
            )
            page_number = gr.Number(label="Page", precision=0, value=1, minimum=1)
            date_from = gr.DateTime(label="Modified from", include_time=False)
            date_to = gr.DateTime(label="Modified to", include_time=False)
            created_from = gr.DateTime(label="Created from", include_time=False)
            created_to = gr.DateTime(label="Created to", include_time=False)
            filetypes = gr.CheckboxGroup(
                choices=list(DEFAULT_FILETYPES),
                label="File types",
            )
            run_button = gr.Button("Search", variant="primary")
            refresh_button = gr.Button("Refresh indices and cache")
            refresh_status = gr.Markdown()
            summary = gr.Markdown()
            with gr.Accordion("Full-text index health", open=False):
                missing_button = gr.Button("Check for missing full-text")
                missing_status = gr.Markdown()
                missing_list = gr.Code(label="Missing paths", language="markdown")
                reingest_button = gr.Button("Rebuild full-text (reingest)")
        with gr.Column(scale=2):
            with gr.Row():
                prev_button = gr.Button("Previous")
                next_button = gr.Button("Next")
            results = gr.Dataframe(
                headers=RESULT_HEADERS,
                datatype=["str", "str", "number", "str"],
                row_count=0,
                column_count=(len(RESULT_HEADERS), "fixed"),
                interactive=False,
                label="Results",
            )
            with gr.Row():
                open_selected_button = gr.Button("Open selected", variant="primary")
                show_folder_button = gr.Button("Show in folder")
            open_status = gr.Markdown()
            snippet = gr.Markdown(label="Selected document")

    def run_search(
        query_value: str,
        path_contains_value: str,
        sort_value: str,
        page_size_value: int,
        page_value: float,
        date_from_value: object,
        date_to_value: object,
        created_from_value: object,
        created_to_value: object,
        filetypes_value: list[str],
    ):
        safe_page_size = int(page_size_value or PAGE_SIZE_OPTIONS[1])
        safe_page = max(int(page_value or 1), 1)
        base_request = SearchRequest(
            query=query_value or "",
            filetypes=filetypes_value or [],
            page=safe_page - 1,
            page_size=safe_page_size,
            sort=sort_value or "relevance",
            path_contains=path_contains_value or None,
            modified_from=normalize_date_input(date_from_value),
            modified_to=normalize_date_input(date_to_value),
            created_from=normalize_date_input(created_from_value),
            created_to=normalize_date_input(created_to_value),
        )
        response = search_usecase.search(base_request)
        total_pages = max(1, math.ceil(response.total / safe_page_size))
        if safe_page > total_pages:
            safe_page = total_pages
            request = replace(base_request, page=safe_page - 1)
            response = search_usecase.search(request)
        rows = search_hits_to_rows(response.hits)
        summary_text = (
            f"Found {response.total} results in {response.took_ms} ms. "
            f"Page {safe_page} of {total_pages}."
        )
        return (
            pd.DataFrame(rows),
            response.hits,
            summary_text,
            "",
            "",
            total_pages,
            safe_page,
        )

    def _get_hit_row_index(evt: gr.SelectData) -> int | None:
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if row_index is None:
            return None
        return int(row_index)

    def _extract_hit_path(hit) -> tuple[str, str]:
        if isinstance(hit, dict):
            path = hit.get("path") or ""
            filename = hit.get("filename") or path
            return path, filename
        path = getattr(hit, "path", "") or ""
        filename = getattr(hit, "filename", "") or path
        return path, filename

    def show_snippet(evt: gr.SelectData, hits):
        if not hits:
            return "", None
        row_index = _get_hit_row_index(evt)
        if row_index is None or row_index >= len(hits):
            return "", None
        return render_search_snippet(hits[row_index]), row_index

    def open_selected(hits, selected_index: int | None, show_folder: bool = False) -> str:
        if not hits:
            return "No results to open."
        if selected_index is None:
            return "Select a row to open."
        if selected_index < 0 or selected_index >= len(hits):
            return "Selected row is out of range."
        path, filename = _extract_hit_path(hits[selected_index])
        if not path:
            return "Selected row has no file path."
        try:
            if show_folder:
                show_in_folder(path)
                return f"Opened folder for {filename}."
            open_file_local(path)
            return f"Opened {filename}."
        except Exception as exc:  # noqa: BLE001
            action = "show folder" if show_folder else "open"
            return f"Failed to {action} {filename}: {exc}"

    def refresh_indices() -> str:
        refreshed = search_usecase.refresh_search_index()
        return "Indices refreshed." if refreshed else "Index refresh failed."

    def find_missing_fulltext() -> tuple[str, list[str], str]:
        missing_paths = search_usecase.find_missing_files(limit=10000)
        if not missing_paths:
            return "All indexed files are present in the full-text index.", [], ""
        listing = "\n".join(missing_paths)
        return (
            f"{len(missing_paths)} file(s) missing from full-text index.",
            missing_paths,
            listing,
        )

    def reingest_missing(missing_paths: list[str]) -> str:
        if not missing_paths:
            return "No missing files to reingest."
        response = ingest_usecase.ingest(
            IngestRequest(paths=missing_paths, mode="reingest")
        )
        message = f"Queued reingest for {response.queued_count} file(s)."
        if response.errors:
            message += "\n" + "\n".join(response.errors)
        return message

    def next_page(page_value: float, total_pages: int) -> int:
        return min(int(page_value or 1) + 1, total_pages or 1)

    def prev_page(page_value: float) -> int:
        return max(int(page_value or 1) - 1, 1)

    run_button.click(
        run_search,
        inputs=[
            query,
            path_contains,
            sort,
            page_size,
            page_number,
            date_from,
            date_to,
            created_from,
            created_to,
            filetypes,
        ],
        outputs=[
            results,
            hits_state,
            summary,
            snippet,
            open_status,
            total_pages_state,
            page_number,
        ],
    )
    query.submit(
        run_search,
        inputs=[
            query,
            path_contains,
            sort,
            page_size,
            page_number,
            date_from,
            date_to,
            created_from,
            created_to,
            filetypes,
        ],
        outputs=[
            results,
            hits_state,
            summary,
            snippet,
            open_status,
            total_pages_state,
            page_number,
        ],
    )
    results.select(
        show_snippet, inputs=[hits_state], outputs=[snippet, selected_index_state]
    )
    open_selected_button.click(
        lambda hits, selected_index: open_selected(hits, selected_index, False),
        inputs=[hits_state, selected_index_state],
        outputs=[open_status],
    )
    show_folder_button.click(
        lambda hits, selected_index: open_selected(hits, selected_index, True),
        inputs=[hits_state, selected_index_state],
        outputs=[open_status],
    )
    refresh_button.click(refresh_indices, outputs=[refresh_status])
    missing_button.click(
        find_missing_fulltext,
        outputs=[missing_status, missing_state, missing_list],
    )
    reingest_button.click(
        reingest_missing, inputs=[missing_state], outputs=[missing_status]
    )
    next_button.click(
        next_page, inputs=[page_number, total_pages_state], outputs=[page_number]
    ).then(
        run_search,
        inputs=[
            query,
            path_contains,
            sort,
            page_size,
            page_number,
            date_from,
            date_to,
            created_from,
            created_to,
            filetypes,
        ],
        outputs=[
            results,
            hits_state,
            summary,
            snippet,
            open_status,
            total_pages_state,
            page_number,
        ],
    )
    prev_button.click(prev_page, inputs=[page_number], outputs=[page_number]).then(
        run_search,
        inputs=[
            query,
            path_contains,
            sort,
            page_size,
            page_number,
            date_from,
            date_to,
            created_from,
            created_to,
            filetypes,
        ],
        outputs=[
            results,
            hits_state,
            summary,
            snippet,
            open_status,
            total_pages_state,
            page_number,
        ],
    )
