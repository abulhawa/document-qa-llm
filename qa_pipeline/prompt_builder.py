from typing import Dict, List

from qa_pipeline.types import PromptRequest, RetrievalResult


CHAT_SYSTEM_PROMPT = (
    "You are a helpful and fact-based assistant. Only answer the specific question asked, "
    "using only the provided documents. Do not answer related questions, and do not include "
    "extra information that was not explicitly requested. If the answer is not clearly present, "
    "say 'I don't know.' Avoid assumptions, commentary, or elaboration. Keep your response concise "
    "and directly focused on the user's question."
)


def build_prompt(
    retrieval: RetrievalResult, question: str, mode: str = "completion", chat_history: List[Dict[str, str]] | None = None
) -> PromptRequest:
    context_text = "\n\n".join(retrieval.context_chunks)

    if mode == "chat":
        system_msg = CHAT_SYSTEM_PROMPT
        user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"
        full_history: List[Dict[str, str]] = [{"role": "system", "content": "You are a helpful assistant."}]
        if chat_history:
            full_history.extend(chat_history)
        full_history.append({"role": "user", "content": user_msg})
        return PromptRequest(prompt=full_history, mode="chat")

    prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    return PromptRequest(prompt=prompt, mode="completion")
