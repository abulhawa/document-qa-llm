from typing import List
from vector_store import query_top_k
from llm import ask_llm
from config import logger


def build_prompt(context_chunks: List[str], question: str) -> str:
    context_text = "\n\n".join(context_chunks)
    return f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"


def answer_question(question: str, top_k: int = 5) -> str:
    logger.info("🔍 Running semantic search for: %s", question)
    top_chunks = query_top_k(query=question, top_k=top_k)

    if not top_chunks:
        logger.warning("No relevant chunks found.")
        return "No relevant context found to answer the question."

    context = [chunk["content"] for chunk in top_chunks]
    prompt = build_prompt(context, question)

    logger.info("🧠 Sending prompt to LLM...")
    answer = ask_llm(prompt)
    return answer
