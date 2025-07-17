from typing import List, Union, Dict, Optional, Tuple
import os
from vector_store import query_top_k
from llm import ask_llm
from config import logger


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
    logger.info("🔍 Running semantic search for: %s", question)
    top_chunks = query_top_k(query=question, top_k=top_k)

    if not top_chunks:
        logger.warning("No relevant chunks found.")
        return "No relevant context found to answer the question.", []

    context = [chunk["content"] for chunk in top_chunks]

    # Format source references
    seen = set()
    sources = []
    for chunk in top_chunks:
        meta = chunk["metadata"]
        path = os.path.basename(meta.get("path", ""))
        if meta.get("page") is not None:
            label = f"{path} (Page. {meta['page']})"
        elif meta.get("location_percent") is not None:
            label = f"{path} (~{meta['location_percent']}%)"
        else:
            label = path

        if label not in seen:
            sources.append(label)
            seen.add(label)

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
            return "Internal error: invalid prompt format.", sources

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
