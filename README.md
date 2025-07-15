# Document QA System (Local LLM + Qdrant)

This is a local document Q&A system powered by a locally running LLM and Qdrant for fast vector search and metadata storage. It supports ingestion of documents from file uploads or folder paths and lets you ask natural language questions via a Streamlit UI.

## 🔧 Features

- Local document ingestion and chunking
- Embedding generation using Sentence Transformers
- Qdrant used for **both vector storage and metadata tracking** (replaces FAISS and SQLite)
- Duplicate file detection using SHA256 checksum
- Skips re-indexing already indexed documents
- Streamlit-based interface with:
   - File upload tab
   - Folder path ingestion tab
   - Question answering interface
- Integrated with a locally running LLM (e.g., text-generation-webui via OpenAI-compatible API)

## 🗂️ File Structure

```bash
.
├── app.py                 # Streamlit app
├── config.py              # Configuration (paths, settings)
├── ingest.py              # Document ingestion, chunking, embedding, and Qdrant storage
├── llm.py                 # Interface with local LLM
├── qdrant_store.py        # Qdrant-based storage and retrieval
├── utils.py               # Utility functions (e.g., checksum)
├── README.md              # This file

## 🛠️ Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure Qdrant is running locally on `http://localhost:6333`.

3. Ensure your LLM API (e.g., text-generation-webui) is available at `http://localhost:5000`.

## ▶️ Running the App

```bash
streamlit run app.py
```

Use the web interface to ingest documents or folders and start asking questions.

## 🧠 How It Works

1. **Ingestion**:
   - Supported formats: `.pdf`, `.docx`, `.txt`
   - Documents are chunked and embedded
   - Stored in Qdrant

2. **Querying**:
   - Retrieves top-k most relevant chunks from Qdrant
   - Passes context to LLM and returns answer
   - Future TODO: Hybrid mode — use full document text if only one file involved
