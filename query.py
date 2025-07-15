from typing import List, Union, Dict, Optional
from vector_store import query_top_k
from llm import ask_llm
from config import logger


def build_prompt(
    context_chunks: List[str], question: str, mode: str = "completion"
) -> Union[str, List[Dict[str, str]]]:
    context_text = "\n\n".join(context_chunks)

    if mode == "chat":
        system_msg = "You are a helpful assistant. Use the provided context to answer the question accurately."
        user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
    else:
        return f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"


def answer_question(
    question: str,
    top_k: int = 5,
    mode: str = "completion",
    temperature: float = 0.7,
    model: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    logger.info("🔍 Running semantic search for: %s", question)
    top_chunks = query_top_k(query=question, top_k=top_k)

    if not top_chunks:
        logger.warning("No relevant chunks found.")
        return "No relevant context found to answer the question."

    context = [chunk["content"] for chunk in top_chunks]

    if mode == "chat":
        logger.info("💬 Sending chat prompt to LLM with history...")
        # Build system + user message based on current question and top chunks
        new_turn = build_prompt(context, question, mode="chat")
        # Combine with existing history (if any)
        full_history = [{"role": "system", "content": "You are a helpful assistant."}]
        if chat_history:
            full_history.extend(chat_history)

        if isinstance(new_turn, list):
            full_history.extend(new_turn[1:])  # only user message, not system again
        else:
            logger.error(
                "Expected chat prompt as list of messages, got string instead."
            )
            return "Internal error: invalid prompt format."

        return ask_llm(
            prompt=full_history,
            mode="chat",
            temperature=temperature,
            model=model,
        )
    else:
        logger.info("🧠 Sending completion prompt to LLM...")
        prompt = build_prompt(context, question, mode="completion")
        return ask_llm(
            prompt=prompt,
            mode="completion",
            temperature=temperature,
            model=model,
        )
