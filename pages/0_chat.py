import streamlit as st
from typing import List, Optional
from config import logger
from core.llm import get_available_models, load_model, check_llm_status
from qa_pipeline import answer_question

st.set_page_config(page_title="Ask a Question", layout="wide")
st.title("💬 Talk to Your Documents")

st.markdown(
    """
    Upload and index your documents using the sidebar.
    Then ask natural-language questions powered by a **local, private LLM**.

    **No cloud, no leaks - just answers.**
    """
)
st.markdown("---")

llm_status = check_llm_status()
if not llm_status["active"]:
    st.error(llm_status["status_message"], icon="⚠️")
# ───────────────────────────────────────
# 🔸 Sidebar: LLM Settings
# ───────────────────────────────────────
with st.sidebar.expander("🧠 LLM Settings", expanded=True):
    llm_models: List[str] = []
    loaded_llm_model: Optional[str] = None
    if not llm_status["server_online"]:
        st.info("The LLM server is unreachable or offline.")
    else:
        llm_models = get_available_models()
        loaded_llm_model = llm_status["current_model"]

    st.markdown("---")

    loaded_model_index: Optional[int] = None
    if loaded_llm_model in llm_models:
        loaded_model_index = llm_models.index(loaded_llm_model)

    selected_model = st.selectbox(
        "Choose a model to load",
        llm_models,
        index=loaded_model_index,
        placeholder="Select a model..." if llm_models else "No models available...",
    )

    if st.button(
        "📦 Load Model",
        disabled=not selected_model,
        help=None if llm_status["server_online"] else llm_status["status_message"],
    ):
        if selected_model:
            if selected_model != loaded_llm_model:
                with st.spinner("Loading model..."):
                    if load_model(selected_model):
                        llm_status = check_llm_status()
                        loaded_llm_model = llm_status["current_model"]
                        st.toast(f"✅ Loaded: {loaded_llm_model}")
                    else:
                        st.error("❌ Failed to load model.")
            else:
                st.toast("Model selected is already loaded", icon="🎉")
        else:
            st.warning("⚠️ Please select a model first.")

    if llm_status["model_loaded"]:
        st.info(f"**Loaded model**: `{llm_status['current_model']}`", icon="🧠")
    elif llm_status["model_loaded"] is False:
        st.warning("No model is currently loaded.", icon="⚠️")
    else:
        st.warning(f"Model status error: {llm_status['status_message']}", icon="⚠️")

    st.markdown("---")
    mode = st.radio(
        "LLM Mode",
        ["completion", "chat"],
        index=0,
        disabled=not llm_status["active"],
        help=None if llm_status["active"] else llm_status["status_message"],
    )
    temperature = st.slider(
        "Temperature",
        0.0,
        1.5,
        0.7,
        step=0.05,
        disabled=not llm_status["active"],
        help=None if llm_status["active"] else llm_status["status_message"],
    )

with st.container():
    # ───────────────────────────────────────
    # 🔹 Chat History
    # ───────────────────────────────────────
    st.session_state.setdefault("chat_history", [])

    if mode == "chat":
        st.markdown("### 💬 Conversation Mode")

        col1, col2 = st.columns([5, 1])
        with col2:
            if st.button(
                "🗑️ Clear Chat",
                disabled=not llm_status["active"],
                help=None if llm_status["active"] else llm_status["status_message"],
            ):
                st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input(
            "Ask a question...", disabled=not llm_status["active"]
        )
        if user_input:
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            with st.chat_message("user"):
                st.markdown(user_input)

            result = answer_question(
                question=user_input,
                mode="chat",
                temperature=temperature,
                model=loaded_llm_model,
                chat_history=st.session_state.chat_history,
            )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": result.answer or ""}
            )

            with st.chat_message("assistant"):
                st.markdown(result.answer or "")
                if result.sources:
                    st.markdown("#### 📁 Sources:")
                    for src in result.sources:
                        st.markdown(f"- {src}")

    # ───────────────────────────────────────
    # 🔹 Completion Mode UI
    # ───────────────────────────────────────
    else:
        st.markdown("### 🧠 Ask a Question")
        with st.form("completion_form"):
            query = st.text_input(
                "Your question",
                disabled=not llm_status["active"],
            )
            submitted = st.form_submit_button(
                "Get Answer",
                disabled=not llm_status["active"],
                help=None if llm_status["active"] else llm_status["status_message"],
            )

        if submitted and query:
            with st.spinner("🧠 Thinking..."):
                result = answer_question(
                    question=query,
                    mode="completion",
                    temperature=temperature,
                    model=loaded_llm_model,
                )

            st.subheader("📝 Answer")
            st.markdown(result.answer or "")
            logger.info(f"LLM Answer:\n{result.answer}")

            if result.sources:
                st.markdown("#### 📁 Sources:")
                for src in result.sources:
                    st.markdown(f"- {src}")

            st.caption(
                f"Mode: {mode} | Temp: {temperature} | Model: {loaded_llm_model}"
            )
