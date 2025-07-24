import os
from typing import List, Union, Dict, Optional, Tuple

from tracing import (
    start_span,
    EMBEDDING,
    RETRIEVER,
    LLM,
    INPUT_VALUE,
    OUTPUT_VALUE,
    record_span_error,
    get_current_span,
)


from core.embeddings import embed_texts
from core.vector_store import retrieve_top_k
from core.llm import ask_llm
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

    with start_span("Embed query", EMBEDDING) as span:
        logger.info("🔍 Running semantic search for user question...")
        try:
            span.set_attribute(INPUT_VALUE, question)
            span.set_attribute("question_length", len(question))
            query_embedding = embed_texts([question])[0]
            span.set_attribute(
                OUTPUT_VALUE, f"{len(query_embedding)} dimensional vector"
            )

        except Exception as e:
            logger.error(f"❌ Query embedding failed: {e}")
            record_span_error(span, e)
            return "❌ Failed to embed query.", []

    with start_span("Retriever", RETRIEVER) as span:
        try:
            span.set_attribute(INPUT_VALUE, question)
            span.set_attribute("embedding_dim", len(query_embedding))
            span.set_attribute("top_k", top_k)
            top_chunks = retrieve_top_k(query_embedding=query_embedding, top_k=top_k)
            span.set_attribute("results_found", len(top_chunks))

            retrieved_summary = [
                f"{chunk.get('path', '')} | idx={chunk.get('chunk_index')} | "
                f"score={chunk.get('score'):.4f} | page={chunk.get('page')} | ~{chunk.get('location_percent')}%"
                for chunk in top_chunks
            ]
            span.set_attribute("retrieved_summary", retrieved_summary)            
            span.set_attribute(OUTPUT_VALUE, retrieved_summary)

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            record_span_error(span, e)
            return "❌ Retrieval failed.", []

    if not top_chunks:
        logger.warning("⚠️ No relevant chunks found.")
        return "No relevant context found to answer the question.", []

    context = [chunk["content"] for chunk in top_chunks]

    seen = set()
    sources = []
    for chunk in top_chunks:
        path = chunk.get("path", "")
        if "page" in chunk and chunk["page"] is not None:
            label = f"{path} (Page {chunk['page']})"
        elif "location_percent" in chunk:
            label = f"{path} (~{chunk['location_percent']}%)"
        else:
            label = path
        if label not in seen:
            sources.append(label)
            seen.add(label)

    with start_span("LLM", LLM) as span:
        try:
            span.set_attribute("model", model or "unknown")
            span.set_attribute("temperature", temperature)
            span.set_attribute("mode", mode)

            if mode == "chat":
                new_turn = build_prompt(context, question, mode="chat")
                full_history = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]
                if chat_history:
                    full_history.extend(chat_history)
                if isinstance(new_turn, list):
                    full_history.extend(new_turn[1:])
                else:
                    raise ValueError("Invalid chat prompt format")

                span.set_attribute(INPUT_VALUE, str(full_history)[:1000])
                answer = ask_llm(
                    prompt=full_history,
                    mode="chat",
                    temperature=temperature,
                    model=model,
                )
            else:
                prompt = build_prompt(context, question, mode="completion")
                span.set_attribute(INPUT_VALUE, str(prompt[:1000]))
                answer = ask_llm(
                    prompt=prompt,
                    mode="completion",
                    temperature=temperature,
                    model=model,
                )

            span.set_attribute(OUTPUT_VALUE, answer[:1000])
            logger.info("✅ LLM answered the question.")
            return answer, sources

        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            record_span_error(span, e)
            return "❌ LLM call failed.", []
