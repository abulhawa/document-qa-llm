import faiss
import pickle
import requests
from sentence_transformers import SentenceTransformer
import config

def get_top_chunks(question, top_k):
    with open("data_store/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    index = faiss.read_index(config.FAISS_INDEX_PATH)
    embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    q_embedding = embed_model.encode([question])
    scores, indices = index.search(q_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def ask_llm(context, question):
    prompt = config.PROMPT_TEMPLATE.format(context="\n\n".join(context), question=question)
    payload = {
        "model": config.LLM_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": 0.3
    }
    response = requests.post(f"{config.LLM_API_BASE}/completions", json=payload)
    return response.json()["choices"][0]["text"].strip()
