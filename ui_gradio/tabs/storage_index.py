"""Storage and index hub tab for Gradio."""

from __future__ import annotations

import gradio as gr

from ui_gradio.tabs.ingest import build_ingest_logs_section
from ui_gradio.tabs.maintenance import (
    build_duplicates_section,
    build_file_resync_section,
    build_index_viewer_section,
    build_maintenance_tab,
    build_watchlist_section,
)
from ui_gradio.tabs.search import build_search_tab


def build_storage_index_tab() -> None:
    with gr.Tabs():
        with gr.Tab("Search"):
            build_search_tab()
        with gr.Tab("Index Viewer"):
            build_index_viewer_section(use_accordion=False)
        with gr.Tab("Ingestion Logs"):
            build_ingest_logs_section()
        with gr.Tab("File Resync"):
            build_file_resync_section(use_accordion=False)
        with gr.Tab("Duplicate Files"):
            build_duplicates_section(use_accordion=False)
        with gr.Tab("Watchlist"):
            build_watchlist_section(use_accordion=False)
        with gr.Tab("Maintenance"):
            build_maintenance_tab()
