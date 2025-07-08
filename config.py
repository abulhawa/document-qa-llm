EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
FAISS_INDEX_PATH = "data_store/index.faiss"
DOCS_FOLDER = "docs"
LLM_API_BASE = "http://localhost:5000/v1"
LLM_MODEL_NAME = "local-mixtral"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
PROMPT_TEMPLATE = """Use the following context to answer the question:

{context}

Question: {question}
Answer:
"""