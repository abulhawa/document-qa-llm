"""Chat assistant tab."""

from __future__ import annotations

import gradio as gr

from app.gradio_utils import to_chat_history
from app.schemas import QARequest
from app.usecases import qa_usecase
from core.llm import check_llm_status, get_available_models, load_model


def _format_llm_status(llm_status: dict) -> str:
    if llm_status.get("active"):
        return f"✅ {llm_status.get('status_message')}"
    return f"⚠️ {llm_status.get('status_message')}"


def _format_model_status(llm_status: dict) -> str:
    if llm_status.get("model_loaded"):
        return f"**Loaded model**: `{llm_status.get('current_model')}`"
    if llm_status.get("model_loaded") is False:
        return "⚠️ No model is currently loaded."
    return f"⚠️ Model status error: {llm_status.get('status_message')}"


def _format_sources(sources: list[str]) -> str:
    if not sources:
        return ""
    items = "\n".join(f"- {src}" for src in sources)
    return f"#### 📁 Sources:\n{items}"


def _sync_llm_state() -> tuple[dict, list[str], str | None]:
    llm_status = check_llm_status()
    llm_models: list[str] = []
    loaded_llm_model: str | None = None
    if llm_status.get("server_online"):
        llm_models = get_available_models()
        loaded_llm_model = llm_status.get("current_model")
    return llm_status, llm_models, loaded_llm_model


def build_chat_tab() -> None:
    llm_status, llm_models, loaded_llm_model = _sync_llm_state()

    llm_status_state = gr.State(llm_status)
    llm_models_state = gr.State(llm_models)
    loaded_llm_model_state = gr.State(loaded_llm_model)

    with gr.Accordion("🧠 LLM Settings", open=True):
        llm_status_md = gr.Markdown(_format_llm_status(llm_status))
        model_status_md = gr.Markdown(_format_model_status(llm_status))
        model_feedback_md = gr.Markdown()

        model = gr.Dropdown(
            choices=llm_models,
            value=loaded_llm_model if loaded_llm_model in llm_models else None,
            allow_custom_value=False,
            label="Choose a model to load",
        )
        with gr.Row():
            load_button = gr.Button("📦 Load Model", variant="primary")
            refresh_button = gr.Button("🔄 Refresh")

        gr.Markdown("---")
        mode = gr.Radio(
            ["completion", "chat"],
            value="completion",
            label="LLM Mode",
            interactive=bool(llm_status.get("active")),
        )
        temperature = gr.Slider(
            0.0,
            1.5,
            value=0.7,
            step=0.05,
            label="Temperature",
            interactive=bool(llm_status.get("active")),
        )
        use_cache = gr.Checkbox(
            label="Use LLM cache",
            value=True,
            interactive=bool(llm_status.get("active")),
        )

    gr.Markdown("## Chat Assistant")
    gr.Markdown("Ask questions across your indexed documents.")

    with gr.Group(visible=False) as chat_group:
        chatbot = gr.Chatbot(label="Conversation")
        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask a question...",
                label="Message",
                interactive=bool(llm_status.get("active")),
            )
            chat_send = gr.Button(
                "Send",
                variant="primary",
                interactive=bool(llm_status.get("active")),
            )
            clear_chat = gr.Button(
                "🗑️ Clear Chat",
                interactive=bool(llm_status.get("active")),
            )

    with gr.Group(visible=True) as completion_group:
        completion_input = gr.Textbox(
            label="Your question",
            interactive=bool(llm_status.get("active")),
        )
        completion_submit = gr.Button(
            "Get Answer",
            variant="primary",
            interactive=bool(llm_status.get("active")),
        )
        completion_output = gr.Markdown()

    def update_visibility(selected_mode: str):
        return (
            gr.update(visible=selected_mode == "chat"),
            gr.update(visible=selected_mode != "chat"),
        )

    def apply_llm_status(llm_status: dict):
        interactive = bool(llm_status.get("active"))
        return (
            gr.update(interactive=interactive),
            gr.update(interactive=interactive),
            gr.update(interactive=interactive),
            gr.update(interactive=interactive),
            gr.update(interactive=interactive),
            gr.update(interactive=interactive),
            gr.update(interactive=interactive),
            gr.update(interactive=interactive),
        )

    def refresh_llm_state():
        llm_status, llm_models, loaded_llm_model = _sync_llm_state()
        status_md = _format_llm_status(llm_status)
        model_md = _format_model_status(llm_status)
        selected_model = loaded_llm_model if loaded_llm_model in llm_models else None
        dropdown_update = gr.update(choices=llm_models, value=selected_model)
        (
            mode_update,
            temperature_update,
            cache_update,
            chat_input_update,
            chat_send_update,
            completion_input_update,
            completion_submit_update,
            clear_chat_update,
        ) = apply_llm_status(llm_status)
        return (
            llm_status,
            llm_models,
            loaded_llm_model,
            status_md,
            model_md,
            dropdown_update,
            mode_update,
            temperature_update,
            cache_update,
            chat_input_update,
            chat_send_update,
            completion_input_update,
            completion_submit_update,
            clear_chat_update,
        )

    def load_selected_model(selected_model: str | None):
        llm_status, llm_models, loaded_llm_model = _sync_llm_state()
        feedback = ""
        if not llm_status.get("server_online"):
            feedback = "⚠️ The LLM server is unreachable or offline."
        elif not selected_model:
            feedback = "⚠️ Please select a model first."
        elif selected_model == loaded_llm_model:
            feedback = "🎉 Model selected is already loaded."
        else:
            feedback = "❌ Failed to load model."
            if load_model(selected_model):
                llm_status, llm_models, loaded_llm_model = _sync_llm_state()
                feedback = f"✅ Loaded: {loaded_llm_model}"
        status_md = _format_llm_status(llm_status)
        model_md = _format_model_status(llm_status)
        selected_model_value = (
            loaded_llm_model if loaded_llm_model in llm_models else selected_model
        )
        dropdown_update = gr.update(choices=llm_models, value=selected_model_value)
        (
            mode_update,
            temperature_update,
            cache_update,
            chat_input_update,
            chat_send_update,
            completion_input_update,
            completion_submit_update,
            clear_chat_update,
        ) = apply_llm_status(llm_status)
        return (
            llm_status,
            llm_models,
            loaded_llm_model,
            status_md,
            model_md,
            feedback,
            dropdown_update,
            mode_update,
            temperature_update,
            cache_update,
            chat_input_update,
            chat_send_update,
            completion_input_update,
            completion_submit_update,
            clear_chat_update,
        )

    def respond_chat(
        message: str,
        history: list[list[str]],
        temperature_value: float,
        model_value: str | None,
        use_cache_value: bool,
    ):
        llm_status = check_llm_status()
        status_md = _format_llm_status(llm_status)
        model_md = _format_model_status(llm_status)
        if not llm_status.get("active"):
            warning = f"⚠️ {llm_status.get('status_message')}"
            return "", history, status_md, model_md, warning
        chat_history = to_chat_history(history, message)
        req = QARequest(
            question=message,
            mode="chat",
            temperature=temperature_value,
            model=model_value,
            chat_history=chat_history,
            use_cache=use_cache_value,
        )
        response = qa_usecase.answer(req)
        answer = response.answer or response.error or ""
        sources_md = _format_sources(response.sources)
        if sources_md:
            answer = f"{answer}\n\n{sources_md}"
        if response.error and not response.answer:
            answer = f"⚠️ {response.error}"
        updated_history = history + [[message, answer]]
        return "", updated_history, status_md, model_md, ""

    def respond_completion(
        message: str,
        temperature_value: float,
        model_value: str | None,
        use_cache_value: bool,
    ):
        llm_status = check_llm_status()
        status_md = _format_llm_status(llm_status)
        model_md = _format_model_status(llm_status)
        if not llm_status.get("active"):
            warning = f"⚠️ {llm_status.get('status_message')}"
            return warning, status_md, model_md
        req = QARequest(
            question=message,
            mode="completion",
            temperature=temperature_value,
            model=model_value,
            use_cache=use_cache_value,
        )
        response = qa_usecase.answer(req)
        answer = response.answer or response.error or ""
        sources_md = _format_sources(response.sources)
        if sources_md:
            answer = f"{answer}\n\n{sources_md}"
        if response.error and not response.answer:
            answer = f"⚠️ {response.error}"
        return answer, status_md, model_md

    mode.change(update_visibility, inputs=[mode], outputs=[chat_group, completion_group])

    load_button.click(
        load_selected_model,
        inputs=[model],
        outputs=[
            llm_status_state,
            llm_models_state,
            loaded_llm_model_state,
            llm_status_md,
            model_status_md,
            model_feedback_md,
            model,
            mode,
            temperature,
            use_cache,
            chat_input,
            chat_send,
            completion_input,
            completion_submit,
            clear_chat,
        ],
    )
    refresh_button.click(
        refresh_llm_state,
        outputs=[
            llm_status_state,
            llm_models_state,
            loaded_llm_model_state,
            llm_status_md,
            model_status_md,
            model,
            mode,
            temperature,
            use_cache,
            chat_input,
            chat_send,
            completion_input,
            completion_submit,
            clear_chat,
        ],
    )

    chat_send.click(
        respond_chat,
        inputs=[chat_input, chatbot, temperature, loaded_llm_model_state, use_cache],
        outputs=[chat_input, chatbot, llm_status_md, model_status_md, model_feedback_md],
    )
    chat_input.submit(
        respond_chat,
        inputs=[chat_input, chatbot, temperature, loaded_llm_model_state, use_cache],
        outputs=[chat_input, chatbot, llm_status_md, model_status_md, model_feedback_md],
    )
    clear_chat.click(lambda: [], outputs=[chatbot])

    completion_submit.click(
        respond_completion,
        inputs=[completion_input, temperature, loaded_llm_model_state, use_cache],
        outputs=[completion_output, llm_status_md, model_status_md],
    )
