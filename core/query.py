from typing import List, Union, Dict, Optional, Tuple
from tracing import (
    start_span,
    RETRIEVER,
    LLM,
    INPUT_VALUE,
    OUTPUT_VALUE,
    record_span_error,
    STATUS_OK,
)

from core.hybrid_search import retrieve_hybrid
from core.llm import ask_llm
from config import logger
from core.query_rewriter import rewrite_query


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

    rewritten_query = rewrite_query(question, temperature=0.15)
    logger.info(f"Rewritten query: {rewritten_query}")

    if "clarify" in rewritten_query:
        # Early exit — clarification needed
        return (
            f'**Clarify**:  \n   -  {rewritten_query["clarify"]}.  \n\nTry again!',
            [],
        )
    elif "rewritten" in rewritten_query:
        rewritten_query = rewritten_query["rewritten"]
        # Proceed to embed and search...
    else:
        return "❌ Unexpected error occurred... ERR-QRWR", []

    logger.info("🔍 Running semantic search for user question...")

    with start_span("Retriever", RETRIEVER) as span:
        try:
            span.set_attribute(INPUT_VALUE, rewritten_query)
            top_results = retrieve_hybrid(
                rewritten_query, top_k_each=20, final_k=top_k
            )
            span.set_attribute("top_k", top_k)
            span.set_attribute("results_found", len(top_results))

            retrieved_summary = [
                f"{result.get('path', '')} | idx={result.get('chunk_index')} | "
                f"score={result.get('score'):.4f} | page={result.get('page')} | ~{result.get('location_percent')}%"
                for result in top_results
            ]
            span.set_attribute("retrieved_summary", retrieved_summary)
            span.set_attribute(OUTPUT_VALUE, retrieved_summary)
            span.set_status(STATUS_OK)

        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            record_span_error(span, e)
            return "❌ Retrieval failed.", []

    if not top_results:
        logger.warning("⚠️ No relevant results found.")
        return "No relevant context found to answer the question.", []

    context = [result["text"] for result in top_results]

    seen = set()
    sources = []
    for result in top_results:
        path = result.get("path", "")
        if "page" in result and result["page"] is not None:
            label = f"{path} (Page {result['page']})"
        elif "location_percent" in result:
            label = f"{path} (~{result['location_percent']}%)"
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
                # Use original query
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

                span.set_attribute(INPUT_VALUE, str(full_history)[:2000])
                answer = ask_llm(
                    prompt=full_history,
                    mode="chat",
                    temperature=temperature,
                    model=model,
                )
            else:
                # Use original query
                prompt = build_prompt(context, question, mode="completion")
                span.set_attribute(INPUT_VALUE, str(prompt[:2000]))
                answer = ask_llm(
                    prompt=prompt,
                    mode="completion",
                    temperature=temperature,
                    model=model,
                )

            span.set_attribute(OUTPUT_VALUE, answer[:1000])
            span.set_status(STATUS_OK)
            logger.info("✅ LLM answered the question.")
            return answer, sources

        except Exception as e:
            logger.error(f"❌ LLM call failed: {e}")
            record_span_error(span, e)
            return "❌ LLM call failed.", []
