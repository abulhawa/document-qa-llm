import os
import subprocess
import json
from typing import List

import pandas as pd

from tracing import start_span, CHAIN, TOOL, INPUT_VALUE, OUTPUT_VALUE
import streamlit as st
from config import logger
from core.ingestion import ingest_paths
from core.query import answer_question
from core.llm import get_available_models, load_model

# ───────────────────────────────────────
# 🔹 Setup
# ───────────────────────────────────────

st.set_page_config(page_title="Document QA", layout="wide")
st.title("📄 Document Q&A")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ───────────────────────────────────────
# 🔹 Sidebar – LLM Settings
# ───────────────────────────────────────

with st.sidebar.expander("🧠 LLM Settings", expanded=True):
    models = get_available_models()
    if models:
        selected_model = st.selectbox("Choose model", models, index=0)
        if st.button("Load model"):
            if load_model(selected_model):
                st.success(f"✅ Loaded: {selected_model}")
            else:
                st.error("❌ Failed to load model.")
    else:
        st.warning("⚠️ No models available on server.")

    st.markdown("---")
    mode = st.radio("LLM Mode", ["completion", "chat"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, step=0.05)


# ───────────────────────────────────────
# 🔹 File/Folder Picker
# ───────────────────────────────────────


def run_picker(mode: str) -> List[str]:
    """Run external file/folder picker script."""
    try:
        result = subprocess.run(
            ["python", "file_picker.py", mode],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except Exception as e:
        st.error(f"❌ Picker failed: {e}")
        return []


# ───────────────────────────────────────
# 🔹 Ingest Documents
# ───────────────────────────────────────

st.caption("Ingest Documents from File(s) or Folder")
col1, col2 = st.columns([1, 1], gap="small")

selected_files: List[str] = []
with col1:
    if st.button("📄 Select File(s)"):
        selected_files = run_picker("files")
with col2:
    if st.button("📂 Select Folder"):
        selected_files = run_picker("folder")

if selected_files:
    status_table = st.empty()
    status_line = st.empty()
    status_line.success(f"Found {len(selected_files)} path(s).")
    df = pd.DataFrame({"Selected Path": [p.replace("\\", "/") for p in selected_files]})
    status_table.dataframe(df, height=300)

    with start_span("index_chain", CHAIN) as span:
        if len(selected_files) > 5:
            preview = selected_files[:5] + [
                f"... and {len(selected_files) - 5} more not shown here"
            ]
        else:
            preview = selected_files

        span.set_attribute(INPUT_VALUE, preview)
        with st.spinner("🔄 Processing files and folders..."):
            results = ingest_paths(selected_files)

        successes = [r for r in results if r["success"]]
        failures = [(r["path"], r["reason"]) for r in results if not r["success"]]

        span.set_attribute("indexed_files", len(successes))
        span.set_attribute("failed_files", len(failures))
        span.set_attribute(
            OUTPUT_VALUE, f"{len(successes)} indexed, {len(failures)} failed"
        )

        status_line.success(
            f"✅ Indexed {len(successes)} out of {len(results)} file(s)."
        )

        if failures:
            span.set_attribute("failed_files_details", str(failures))

        summary_df = pd.DataFrame(
            [
                {
                    "File": r["path"],
                    "Status": "✅ Success" if r["success"] else f"❌ {r['reason']}",
                }
                for r in results
            ]
        )
        status_table.dataframe(summary_df, height=300)


# ───────────────────────────────────────
# 🔹 Chat Mode UI
# ───────────────────────────────────────

if mode == "chat":
    st.markdown("### 💬 Conversation")
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("💬 Ask a question about your document")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with start_span("qa_chain", CHAIN) as span:
            span.set_attribute("mode", "chat")
            span.set_attribute("model", selected_model)
            span.set_attribute("temperature", temperature)
            span.set_attribute("question_length", len(user_input))
            span.set_attribute(INPUT_VALUE, user_input)

            with st.spinner("🧠 Thinking..."):
                answer, sources = answer_question(
                    question=user_input,
                    mode="chat",
                    temperature=temperature,
                    model=selected_model,
                    chat_history=st.session_state.chat_history,
                )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                st.markdown("#### 📁 Sources:")
                for src in sources:
                    st.markdown(f"- {src}")

# ───────────────────────────────────────
# 🔹 Completion Mode UI
# ───────────────────────────────────────

else:
    with st.form("completion_form"):
        query = st.text_input("💬 Ask a question about your document")
        submitted = st.form_submit_button("Get Answer")

    if submitted and query:
        with start_span("qa_chain", CHAIN) as span:
            span.set_attribute("mode", "completion")
            span.set_attribute("model", selected_model)
            span.set_attribute("temperature", temperature)
            span.set_attribute("question_length", len(query))
            span.set_attribute(INPUT_VALUE, query)

            with st.spinner("🧠 Thinking..."):
                answer, sources = answer_question(
                    question=query,
                    mode="completion",
                    temperature=temperature,
                    model=selected_model,
                )
            span.set_attribute(OUTPUT_VALUE, answer)
        st.subheader("📝 Answer")
        st.markdown(answer)
        logger.info(f"LLM Answer:\n{answer}")

        if sources:
            st.markdown("#### 📁 Sources:")
            for src in sources:
                st.markdown(f"- {src}")

        st.caption(f"Mode: completion | Temp: {temperature} | Model: {selected_model}")
