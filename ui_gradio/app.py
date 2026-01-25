"""Gradio entry point for document-qa-llm."""

from __future__ import annotations

import gradio as gr

from ui_gradio.tabs.chat import build_chat_tab
from ui_gradio.tabs.ingest import build_ingest_tab
from ui_gradio.tabs.maintenance import build_maintenance_tab
from ui_gradio.tabs.search import build_search_tab
from ui_gradio.tabs.topics import build_topics_tab
from ui_gradio.tabs.watchlist import build_watchlist_tab
from ui_gradio.theme import build_theme


def create_app() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# Document QA Workspace")
        cluster_state = gr.State(None)

        with gr.Tabs():
            with gr.Tab("Chat Assistant"):
                build_chat_tab()
            with gr.Tab("Intelligent Search"):
                build_search_tab()
            with gr.Tab("Ingestion Pipeline"):
                build_ingest_tab()
            with gr.Tab("Topic Discovery & Naming"):
                build_topics_tab(cluster_state)
            with gr.Tab("Watchlist"):
                build_watchlist_tab()
            with gr.Tab("Knowledge Base Maintenance"):
                build_maintenance_tab()

    return demo


if __name__ == "__main__":
    theme = build_theme()
    create_app().launch(theme=theme)
