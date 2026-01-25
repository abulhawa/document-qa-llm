"""Storage and index hub tab for Gradio."""

from __future__ import annotations

import gradio as gr

from ui_gradio.tabs.ingest import build_ingest_logs_section
from ui_gradio.tabs.maintenance import build_maintenance_tab
from ui_gradio.tabs.search import build_search_tab


def build_storage_index_tab() -> None:
    with gr.Tabs():
        with gr.Tab("Search"):
            build_search_tab()
        with gr.Tab("Ingestion Logs"):
            build_ingest_logs_section()
        with gr.Tab("Maintenance"):
            build_maintenance_tab()
