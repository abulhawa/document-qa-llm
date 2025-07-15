import os
import streamlit as st

from config import logger, TEMP_DIR
from ingest import ingest
from query import answer_question
from llm import get_available_models, load_model

# ───────────────────────────────────────
# 🔹 Setup
# ───────────────────────────────────────
os.makedirs(TEMP_DIR, exist_ok=True)
st.set_page_config(page_title="Document QA", layout="wide")
st.title("📄 Document Q&A")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ───────────────────────────────────────
# 🔹 Sidebar – Model, Mode, Temperature
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
# 🔹 Document Upload & Ingestion
# ───────────────────────────────────────
uploaded_file = st.file_uploader("📎 Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
if uploaded_file:
    temp_path = os.path.join(TEMP_DIR, uploaded_file.name)

    # Save uploaded file to temp path
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info("Saved uploaded file: %s", temp_path)

    # Ingest and embed into Qdrant
    ingest(temp_path)

    # Delete after ingestion
    try:
        os.remove(temp_path)
        logger.info("Deleted temp file: %s", temp_path)
    except Exception as e:
        logger.warning("Failed to delete temp file: %s", e)

# ───────────────────────────────────────
# 🔹 Render Chat History (chat mode only)
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

# ───────────────────────────────────────
# 🔹 Input + Answer
# ───────────────────────────────────────
if mode == "chat":
    user_input = st.chat_input("💬 Ask a question about your document")
    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🧠 Thinking..."):
            response = answer_question(
                question=user_input,
                mode="chat",
                temperature=temperature,
                model=selected_model,
                chat_history=st.session_state.chat_history,
            )

        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

else:  # mode == "completion"
    with st.form("completion_form"):
        query = st.text_input("💬 Ask a question about your document")
        submitted = st.form_submit_button("Get Answer")

    if submitted and query:
        with st.spinner("🧠 Thinking..."):
            answer = answer_question(
                question=query,
                mode="completion",
                temperature=temperature,
                model=selected_model,
            )
        st.subheader("📝 Answer")
        st.markdown(answer)
        st.caption(f"Mode: completion, Temperature: {temperature}, Model: {selected_model}")
