"""Gradio entry point for document-qa-llm."""

from __future__ import annotations

import gradio as gr

from ui_gradio.tabs.chat import build_chat_tab
from ui_gradio.tabs.ingest import build_ingest_tab
from ui_gradio.tabs.maintenance import build_maintenance_tab
from ui_gradio.tabs.running_tasks import build_running_tasks_tab
from ui_gradio.tabs.search import build_search_tab
from ui_gradio.tabs.tools_file_sorter import build_tools_file_sorter_tab
from ui_gradio.tabs.storage_index import build_storage_index_tab
from ui_gradio.tabs.topics import build_topics_tab
from ui_gradio.tabs.watchlist import build_watchlist_tab
from ui_gradio.theme import build_theme


def create_app() -> gr.Blocks:
    css = """
    .gradio-container {
        max-width: 1300px;
        margin: 0 auto;
    }
    .app-header {
        margin: 0.5rem 0 1rem 0;
    }
    .card {
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 14px;
        padding: 16px;
        background: var(--background-fill-secondary);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    .card .gradio-row,
    .card .gradio-column {
        gap: 0.75rem;
    }
    .block-label {
        font-size: 0.85rem;
        font-weight: 500;
    }
    .gradio-container .form {
        gap: 0.75rem;
    }
    """

    with gr.Blocks(theme=build_theme(), css=css) as demo:
        gr.Markdown("# Document QA Workspace", elem_classes=["app-header"])
        cluster_state = gr.State(None)
        session_tasks_state = gr.State([])

        with gr.Tabs():
            with gr.Tab("Chat Assistant"):
                build_chat_tab()
            with gr.Tab("Intelligent Search"):
                build_search_tab()
            with gr.Tab("Storage & Index"):
                with gr.Group(elem_classes=["card"]):
                    build_storage_index_tab()
            with gr.Tab("Ingestion Pipeline"):
                with gr.Group(elem_classes=["card"]):
                    build_ingest_tab(session_tasks_state)
            with gr.Tab("Topic Discovery & Naming"):
                with gr.Group(elem_classes=["card"]):
                    build_topics_tab(cluster_state)
            with gr.Tab("Watchlist"):
                with gr.Group(elem_classes=["card"]):
                    build_watchlist_tab()
            with gr.Tab("Knowledge Base Maintenance"):
                with gr.Group(elem_classes=["card"]):
                    build_maintenance_tab()
            with gr.Tab("Task Administration"):
                with gr.Group(elem_classes=["card"]):
                    build_running_tasks_tab(session_tasks_state)
            with gr.Tab("Tools - Smart File Sorter"):
                with gr.Group(elem_classes=["card"]):
                    build_tools_file_sorter_tab()

    return demo


demo = create_app()


if __name__ == "__main__":
    demo.launch()
