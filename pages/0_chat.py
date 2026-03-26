import streamlit as st
import uuid
from typing import List, Optional

from app.schemas import QARequest
from app.usecases.qa_usecase import answer as answer_usecase
from config import logger
from core.llm import get_available_models, load_model, check_llm_status
from utils.timing import set_run_id, timed_block


def _source_label(path: str, page: Optional[int], location_percent: Optional[float]) -> str:
    if page is not None:
        return f"{path} (Page {page})"
    if location_percent is not None:
        return f"{path} (~{location_percent}%)"
    return path


def _source_rows(result) -> List[str]:
    documents = getattr(result, "documents", []) or []
    if not documents:
        return list(getattr(result, "sources", []) or [])

    ordered_labels: List[str] = []
    best_score_by_label: dict[str, Optional[float]] = {}
    for doc in documents:
        label = _source_label(
            path=getattr(doc, "path", ""),
            page=getattr(doc, "page", None),
            location_percent=getattr(doc, "location_percent", None),
        )
        score = getattr(doc, "score", None)
        if label not in best_score_by_label:
            ordered_labels.append(label)
            best_score_by_label[label] = score
            continue
        best_score = best_score_by_label[label]
        if score is not None and (best_score is None or score > best_score):
            best_score_by_label[label] = score

    rows: List[str] = []
    for label in ordered_labels:
        score = best_score_by_label[label]
        if score is None:
            rows.append(label)
        else:
            rows.append(f"{label} | score: {score:.3f}")
    return rows


st.set_page_config(page_title="Ask Your Documents", layout="wide")
st.title("Ask Your Documents")

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
    if st.button(
        "Reconnect LLM",
        help="Re-check LLM connectivity and model availability.",
    ):
        llm_status = check_llm_status()
        if llm_status["active"]:
            st.success("LLM connection restored.")
        else:
            st.warning(llm_status["status_message"])

    st.markdown("---")

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
        0.1,
        step=0.05,
        disabled=not llm_status["active"],
        help=None if llm_status["active"] else llm_status["status_message"],
    )
    use_cache = st.checkbox(
        "Use LLM cache",
        value=True,
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
            run_id = uuid.uuid4().hex[:8]
            st.session_state["_run_id"] = run_id
            set_run_id(run_id)
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )

            with st.chat_message("user"):
                st.markdown(user_input)

            with timed_block(
                "action.chat.chat_input",
                extra={
                    "run_id": run_id,
                    "mode": "chat",
                    "model": loaded_llm_model,
                },
                logger=logger,
            ):
                result = answer_usecase(
                    QARequest(
                        question=user_input,
                        mode="chat",
                        temperature=temperature,
                        model=loaded_llm_model,
                        chat_history=st.session_state.chat_history,
                        use_cache=use_cache,
                    )
                )

            assistant_message = result.answer or result.error or ""
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_message}
            )

            with st.chat_message("assistant"):
                st.markdown(assistant_message)
                source_rows = _source_rows(result)
                if source_rows:
                    st.markdown("#### 📁 Sources:")
                    for src in source_rows:
                        st.markdown(f"- {src}")
                if result.error and not result.answer:
                    st.error(result.error)

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
            run_id = uuid.uuid4().hex[:8]
            st.session_state["_run_id"] = run_id
            set_run_id(run_id)
            with st.spinner("🧠 Thinking..."):
                with timed_block(
                    "action.chat.get_answer",
                    extra={
                        "run_id": run_id,
                        "mode": "completion",
                        "model": loaded_llm_model,
                    },
                    logger=logger,
                ):
                    result = answer_usecase(
                        QARequest(
                            question=query,
                            mode="completion",
                            temperature=temperature,
                            model=loaded_llm_model,
                            use_cache=use_cache,
                        )
                    )

            st.subheader("📝 Answer")
            answer_text = result.answer or result.error or ""
            st.markdown(answer_text)
            logger.info(f"LLM Answer:\n{answer_text}")

            source_rows = _source_rows(result)
            if source_rows:
                st.markdown("#### 📁 Sources:")
                for src in source_rows:
                    st.markdown(f"- {src}")
            if result.error and not result.answer:
                st.error(result.error)

            st.caption(
                f"Mode: {mode} | Temp: {temperature} | Model: {loaded_llm_model}"
            )
