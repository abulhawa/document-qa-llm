"""Intelligent search tab."""

from __future__ import annotations

import pandas as pd
import gradio as gr

from app.gradio_utils import (
    DEFAULT_FILETYPES,
    normalize_date_input,
    render_search_snippet,
    search_hits_to_rows,
)
from app.schemas import SearchRequest
from app.usecases import search_usecase


RESULT_HEADERS = ["Filename", "Score", "Date"]


def build_search_tab() -> None:
    hits_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            query = gr.Textbox(label="Query", placeholder="Search documents...")
            date_from = gr.DateTime(label="Modified from", include_time=False)
            date_to = gr.DateTime(label="Modified to", include_time=False)
            filetypes = gr.CheckboxGroup(
                choices=list(DEFAULT_FILETYPES),
                label="File types",
            )
            run_button = gr.Button("Search", variant="primary")
            summary = gr.Markdown()
        with gr.Column(scale=2):
            results = gr.Dataframe(
                headers=RESULT_HEADERS,
                datatype=["str", "number", "str"],
                row_count=0,
                column_count=(len(RESULT_HEADERS), "fixed"),
                interactive=False,
                label="Results",
            )
            snippet = gr.Markdown(label="Selected document")

    def run_search(
        query_value: str,
        date_from_value: object,
        date_to_value: object,
        filetypes_value: list[str],
    ):
        request = SearchRequest(
            query=query_value or "",
            filetypes=filetypes_value or [],
            modified_from=normalize_date_input(date_from_value),
            modified_to=normalize_date_input(date_to_value),
        )
        response = search_usecase.search(request)
        rows = search_hits_to_rows(response.hits)
        summary_text = f"Found {response.total} results in {response.took_ms} ms."
        return pd.DataFrame(rows), response.hits, summary_text, ""

    def show_snippet(evt: gr.SelectData, hits):
        if not hits:
            return ""
        row_index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        if row_index is None or row_index >= len(hits):
            return ""
        return render_search_snippet(hits[row_index])

    run_button.click(
        run_search,
        inputs=[query, date_from, date_to, filetypes],
        outputs=[results, hits_state, summary, snippet],
    )
    query.submit(
        run_search,
        inputs=[query, date_from, date_to, filetypes],
        outputs=[results, hits_state, summary, snippet],
    )
    results.select(show_snippet, inputs=[hits_state], outputs=[snippet])
