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

# ───────────────────────────────────────
# 🔹 Sidebar – Model Loader
# ───────────────────────────────────────
with st.sidebar.expander("🧠 Load LLM Model", expanded=True):
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

# ───────────────────────────────────────
# 🔹 Main Interface – Upload & Ask
# ───────────────────────────────────────
uploaded_file = st.file_uploader("📎 Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
query = st.text_input("💬 Ask a question about your document")

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
# 🔹 Answer Section
# ───────────────────────────────────────
if query:
    with st.spinner("🧠 Thinking..."):
        answer = answer_question(query)
        st.subheader("📝 Answer")
        st.markdown(answer)
