from typing import Dict, List

from qa_pipeline.types import PromptRequest, RetrievalResult


STRICT_QA_INSTRUCTIONS = (
    "You are a fact-based QA assistant. Answer only the specific question asked using only "
    "the provided context. If the answer is not clearly present in the context, respond exactly "
    "with: I don't know. Do not use outside knowledge, assumptions, or extra commentary. Keep "
    "the response concise and directly focused on the user's question."
)
CHAT_SYSTEM_PROMPT = STRICT_QA_INSTRUCTIONS


def build_prompt(
    retrieval: RetrievalResult, question: str, mode: str = "completion", chat_history: List[Dict[str, str]] | None = None
) -> PromptRequest:
    context_text = "\n\n".join(retrieval.context_chunks)

    if mode == "chat":
        system_msg = CHAT_SYSTEM_PROMPT
        user_msg = f"Context:\n{context_text}\n\nQuestion: {question}"
        full_history: List[Dict[str, str]] = [{"role": "system", "content": system_msg}]
        if chat_history:
            full_history.extend(chat_history)
        full_history.append({"role": "user", "content": user_msg})
        return PromptRequest(prompt=full_history, mode="chat")

    prompt = (
        f"Instructions:\n{STRICT_QA_INSTRUCTIONS}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return PromptRequest(prompt=prompt, mode="completion")
