"""Chat assistant tab."""

from __future__ import annotations

import gradio as gr

from app.gradio_utils import stream_text, to_chat_history
from app.schemas import QARequest
from app.usecases import qa_usecase


DEFAULT_MODEL = "default"


def build_chat_tab() -> None:
    chat_state = gr.State([])

    with gr.Accordion("Advanced Settings", open=False):
        temperature = gr.Slider(
            0.0,
            1.0,
            value=0.2,
            step=0.05,
            label="Temperature",
        )
        model = gr.Dropdown(
            choices=[DEFAULT_MODEL],
            value=DEFAULT_MODEL,
            allow_custom_value=True,
            label="Model",
        )

    def respond(
        message: str,
        history: list[list[str]],
        temperature_value: float,
        model_value: str,
        state: list[dict[str, str]],
    ):
        chat_history = to_chat_history(history, message)
        model_name = None if model_value == DEFAULT_MODEL else model_value
        req = QARequest(
            question=message,
            temperature=temperature_value,
            model=model_name,
            chat_history=chat_history,
        )
        response = qa_usecase.answer(req)
        answer = response.answer or ""
        if response.error:
            answer = f"Error: {response.error}"
        updated_state = chat_history + ([{"role": "assistant", "content": answer}] if answer else [])
        for chunk in stream_text(answer):
            yield chunk, updated_state

    gr.ChatInterface(
        fn=respond,
        additional_inputs=[temperature, model, chat_state],
        additional_outputs=[chat_state],
        title="Chat Assistant",
        description="Ask questions across your indexed documents.",
    )
