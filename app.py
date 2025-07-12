import os
import tempfile
import streamlit as st
import requests

from config import logger
from db import create_tables, get_indexed_chunk_count
from ingest import ingest
from query import answer_question
from faiss_store import rebuild_faiss_index, clear_faiss_index
from llm import get_model_status, get_available_models, load_model

# Create tables on startup
create_tables()

st.set_page_config(page_title="Document QA", layout="wide")
st.title("📚 Document Q&A System")

# --- Sidebar: Model Manager ---
with st.sidebar.expander("🧠 Model Manager", expanded=True):
    models = get_available_models()
    if models:
        selected_model = st.selectbox("Choose a model to load", models, index=0)
        if st.button("Load Selected Model"):
            if load_model(selected_model):
                st.success(f"✅ Model loaded: {selected_model}")
            else:
                st.error("❌ Failed to load model.")
    else:
        st.warning("⚠️ No models found in TGW's model folder.")

# --- Sidebar: FAISS Admin ---
st.sidebar.subheader("🧠 FAISS Index")

# Show current chunk count
try:
    chunk_count = get_indexed_chunk_count()
    st.sidebar.markdown(f"**Indexed Chunks:** {chunk_count}")
except Exception as e:
    st.sidebar.warning("Could not retrieve chunk count.")
    logger.exception("Chunk count failed: %s", e)

# Rebuild index
if st.sidebar.button("Rebuild FAISS Index"):
    try:
        rebuild_faiss_index()
        st.sidebar.success("✅ FAISS index rebuilt.")
    except Exception as e:
        st.sidebar.error("❌ Failed to rebuild index.")
        logger.exception("Rebuild failed: %s", e)

# Clear index with confirmation
confirm_clear = st.sidebar.checkbox("⚠️ Confirm clear index")
if st.sidebar.button("Clear FAISS Index"):
    if confirm_clear:
        try:
            clear_faiss_index()
            st.sidebar.success("🗑️ FAISS index cleared.")
        except Exception as e:
            st.sidebar.error("❌ Failed to clear index.")
            logger.exception("Clear failed: %s", e)
    else:
        st.sidebar.warning("Please confirm before clearing the index.")

# --- Main UI Tabs ---
tab_ingest, tab_query = st.tabs(["📄 Ingest Documents", "❓ Ask a Question"])

# --- Ingest Tab ---
# Ensure temp_docs folder exists
TEMP_DOCS_DIR = "temp_docs"
os.makedirs(TEMP_DOCS_DIR, exist_ok=True)

with tab_ingest:
    st.header("Ingest documents and ask")

    st.subheader("Upload a file (auto-ingested, not stored)")

    uploaded_file = st.file_uploader("Choose a file (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

    if uploaded_file:
        try:
            # Save temporarily
            file_path = os.path.join(TEMP_DOCS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            # Ingest immediately
            ingest(file_path)
            st.success(f"✅ File ingested: {uploaded_file.name}")

            # Delete right after ingestion
            os.remove(file_path)
            logger.info(f"🗑️ Deleted temporary file: {file_path}")
        except Exception as e:
            st.error("❌ Failed to ingest uploaded file.")
            logger.exception("Upload ingest error: %s", e)

    st.markdown("---")

    st.subheader("Ingest a folder")

    folder_path = st.text_input("Enter full path to folder (must exist on this machine):")

    if folder_path:
        try:
            ingest(folder_path)
            st.success("✅ Folder ingested successfully.")
        except Exception as e:
            st.error("❌ Failed to ingest folder.")
            logger.exception("Folder ingest error: %s", e)

    st.markdown("---")

    st.subheader("Ask a question about the documents")

    query = st.text_area("Your question:", height=100)

    if st.button("Get Answer"):
        if query.strip():
            with st.spinner("Thinking..."):
                answer = answer_question(query)
                st.markdown("### 💡 Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a question.")