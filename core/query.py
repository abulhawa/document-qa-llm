from typing import List, Union, Dict, Optional, Tuple
import os

from core.embeddings import embed_texts
from core.vector_store import retrieve_top_k
from core.llm import ask_llm
from config import logger
from tracing import get_tracer

tracer = get_tracer(__name__)


@tracer.chain
def build_prompt(
    context_chunks: List[str], question: str, mode: str = "completion"
) -> Union[str, List[Dict[str, str]]]:
    context_text = "\n\n".join(context_chunks)

    if mode == "chat":
        system_msg = (
            "You are a helpful and fact-based assistant. Only answer the specific question asked, using only the provided documents. "
            "Do not answer related questions, and do not include extra information that was not explicitly requested. "
            "If the answer is not clearly present, say 'I don't know.' Avoid assumptions, commentary, or elaboration. "
            "Keep your response concise and directly focused on the user's question."
        )

        user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    else:
        return f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"


def answer_question(
    question: str,
    top_k: int = 3,
    mode: str = "completion",
    temperature: float = 0.7,
    model: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, List[str]]:
    logger.info("🔍 Embedding query and retrieving top-k results...")

    try:
        query_embedding = embed_texts([question])[0]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return "Failed to process query.", []

    top_chunks = retrieve_top_k(query_embedding=query_embedding, top_k=top_k)

    if not top_chunks:
        logger.warning("No relevant chunks found.")
        return "No relevant context found to answer the question.", []

    context = [chunk["content"] for chunk in top_chunks]

    # Format source references
    seen = set()
    sources = []
    for chunk in top_chunks:
        meta = chunk.get("metadata", {})
        path = os.path.basename(meta.get("path", ""))
        if meta.get("page") is not None:
            label = f"{path} (Page {meta['page']})"
        elif meta.get("location_percent") is not None:
            label = f"{path} (~{meta['location_percent']}%)"
        else:
            label = path
        if label not in seen:
            sources.append(label)
            seen.add(label)

    if mode == "chat":
        logger.info("💬 Sending chat prompt to LLM...")
        new_turn = build_prompt(context, question, mode="chat")

        # Always start with system prompt
        full_history = [{"role": "system", "content": "You are a helpful assistant."}]
        if chat_history:
            full_history.extend(chat_history)

        if isinstance(new_turn, list):
            full_history.extend(new_turn[1:])  # skip the new system message
        else:
            logger.error("Expected chat prompt as a list, got string.")
            return "Internal error: invalid chat prompt format.", sources

        answer = ask_llm(
            prompt=full_history,
            mode="chat",
            temperature=temperature,
            model=model,
        )
    else:
        logger.info("🧠 Sending completion prompt to LLM...")
        prompt = build_prompt(context, question, mode="completion")
        answer = ask_llm(
            prompt=prompt,
            mode="completion",
            temperature=temperature,
            model=model,
        )

    return answer, sources
