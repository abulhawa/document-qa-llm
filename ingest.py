import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import config

def load_documents(folder):
    docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.endswith(".pdf"):
            docs += PyPDFLoader(path).load()
        elif file.endswith(".docx"):
            docs += Docx2txtLoader(path).load()
        elif file.endswith(".txt"):
            docs += TextLoader(path).load()
    return docs

def embed_and_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)

    with open("data_store/chunks.pkl", "wb") as f:
        pickle.dump(texts, f)
    faiss.write_index(index, config.FAISS_INDEX_PATH)

if __name__ == "__main__":
    docs = load_documents(config.DOCS_FOLDER)
    embed_and_store(docs)
