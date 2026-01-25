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
    :root {
        --app-surface: #f8fafc;
        --app-surface-muted: #f1f5f9;
        --app-border: rgba(15, 23, 42, 0.08);
        --app-text: #0f172a;
        --app-text-muted: #475569;
        --app-accent: #3b82f6;
    }
    .gradio-container {
        max-width: 1240px;
        margin: 0 auto;
        padding: 1.5rem 1.5rem 2.5rem;
        font-family: "Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
        color: var(--app-text);
    }
    .gradio-container .app-shell {
        background: var(--app-surface);
        border: 1px solid var(--app-border);
        border-radius: 20px;
        padding: 1.75rem 1.75rem 2rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
    }
    .app-header {
        margin: 0 0 1rem 0;
    }
    .app-header h1 {
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }
    .app-header p {
        color: var(--app-text-muted);
        margin-top: 0.25rem;
        font-size: 0.95rem;
    }
    .tabs {
        gap: 0.5rem;
    }
    .tab-nav {
        background: var(--app-surface-muted);
        border-radius: 14px;
        padding: 0.4rem;
        border: 1px solid var(--app-border);
    }
    .tab-nav button {
        border-radius: 12px;
        padding: 0.5rem 0.9rem;
        font-weight: 600;
        color: var(--app-text-muted);
    }
    .tab-nav button[aria-selected="true"] {
        background: #ffffff;
        color: var(--app-text);
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.08);
        border: 1px solid rgba(15, 23, 42, 0.06);
    }
    .card {
        border: 1px solid var(--app-border);
        border-radius: 16px;
        padding: 18px;
        background: #ffffff;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06);
    }
    .card .gradio-row,
    .card .gradio-column {
        gap: 0.75rem;
    }
    .block-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--app-text-muted);
    }
    .gradio-container .form {
        gap: 0.85rem;
    }
    """

    with gr.Blocks(theme=build_theme(), css=css) as demo:
        with gr.Group(elem_classes=["app-shell"]):
            gr.Markdown(
                "# Document QA Workspace\n"
                "A unified hub for chatting with documents, indexing sources, and maintaining your knowledge base.",
                elem_classes=["app-header"],
            )
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
