# 🧠 Local Document Q&A System

This project is a **fully local, privacy-first system** for question answering over your documents (PDF, DOCX, TXT). It uses semantic search, chunked embeddings, and a local LLM to answer questions accurately — either in single-turn (completion) or multi-turn (chat) modes.

---

## 🔧 Architecture Overview

The system is built from modular components:

### ✅ 1. **Embedding Service** (Dockerized or local)
- Uses `intfloat/multilingual-e5-base` or similar model
- Converts document chunks into vector embeddings

### ✅ 2. **Qdrant** (Vector Store)
- Stores and indexes semantic embeddings
- Supports fast top-k retrieval for chunked search

### ✅ 3. **Text-Generation-WebUI (TGW)**
- Hosts your local LLM (e.g., Mistral, GPTQ, GGUF models)
- Accessible via OpenAI-compatible API endpoints
- Supports both chat and completion-style models

### ✅ 4. **Streamlit Frontend**
- Upload and ingest documents
- Ask questions in chat or single-turn mode
- Switch models, temperatures, and view responses

---

## 🚀 Features

- 🔍 **Semantic Search** over your own documents
- 💬 **Chat Mode** with memory of prior turns
- 🧠 **Completion Mode** for single-shot Q&A
- 📎 Supports PDF, DOCX, and TXT input
- 📁 Real-time ingestion + deduplication
- 🛠️ Plug-and-play backend: switch LLMs or embedding models

---

## 🧪 Usage

### 📥 Upload Documents
- Upload one file at a time
- Automatically chunked, embedded, and stored in Qdrant

### 💬 Ask Questions
- Choose chat or completion mode
- Ask questions about the content
- In chat mode, you can follow up with contextual questions

### 🧠 LLM Settings
- Select model, temperature, and mode from the sidebar

---

## 🧰 Requirements

- Python 3.10+
- Running Qdrant (Docker or local)
- Running Text-Generation-WebUI with model loaded
- Optional: Dockerized embedding service (can run standalone as well)

---

## 📌 Current Status

- ✅ MVP completed with working ingestion, retrieval, LLM connection
- 🔄 Chat and completion modes supported
- 🔍 Prompt building and chunk retrieval working
- ⚠️ Streaming is deprioritized for now

---

## 🛣️ Roadmap

- [ ] Show source attribution (filename + page) with answers
- [ ] Multi-file ingestion (folder support)
- [ ] View/manage indexed files
- [ ] Session save/load
- [ ] Offline Docker bundle (Qdrant + Embedding + Streamlit)
