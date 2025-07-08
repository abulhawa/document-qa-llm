import requests
import json

import streamlit as st
import os
from ingest import embed_and_store
from ask import get_top_chunks, ask_llm
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import config

st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("📄 Ask Your Document")

uploaded_file = st.file_uploader("Upload a document (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

if uploaded_file:
    path = os.path.join("docs", uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")

    with st.spinner("Processing document..."):
        if uploaded_file.name.endswith(".pdf"):
            doc = PyPDFLoader(path).load()
        elif uploaded_file.name.endswith(".docx"):
            doc = Docx2txtLoader(path).load()
        elif uploaded_file.name.endswith(".txt"):
            doc = TextLoader(path).load()
        else:
            st.error("Unsupported file format")
            st.stop()
        embed_and_store(doc)
    st.success("Document indexed successfully!")

    question = st.text_input("Ask a question about this document:")
    if question:
        with st.spinner("Thinking..."):
            context = get_top_chunks(question, config.TOP_K)
            answer = ask_llm(context, question)
        st.markdown("### Answer")
        st.write(answer)

def get_model_status():
    try:
        r = requests.get("http://localhost:5000/v1/models")
        data = r.json()
        if data.get("data"):
            return data["data"][0]["id"]  # currently loaded model
        else:
            return None
    except Exception:
        return None

def get_available_models():
    try:
        res = requests.get("http://localhost:5000/v1/internal/model/list")
        return res.json() if res.status_code == 200 else []
    except Exception:
        return []

def load_model(model_name):
    try:
        res = requests.post("http://localhost:5000/v1/internal/model/load", json={"model_name": model_name})
        return res.status_code == 200
    except Exception:
        return False
            
with st.sidebar.expander("🧠 Model Manager", expanded=True):
    current_model = get_model_status()
    if current_model:
        st.success(f"Loaded: `{current_model}`")
    else:
        st.warning("🚨 No model loaded")

    models = get_available_models()
    if models:
        selected_model = st.selectbox("Choose model", models['model_names'], index=0)
        if st.button("Load model"):
            if load_model(selected_model):
                st.success("✅ Model loaded! Refresh main view.")
            else:
                st.error("❌ Failed to load model.")
    else:
        st.warning("⚠️ No models found in the server model folder.")