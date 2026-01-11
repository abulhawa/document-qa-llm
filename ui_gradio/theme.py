"""Custom Gradio theme helpers."""

from __future__ import annotations

import gradio as gr


def build_theme() -> gr.themes.Theme:
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="gray",
    ).set(
        block_title_text_weight="600",
        block_label_text_weight="600",
    )
