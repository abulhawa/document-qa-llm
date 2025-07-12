from config import logger
from faiss_store import query_faiss
from llm import ask_llm  # Make sure this exists

def build_context(chunks, max_chars=3000):
    context = ""
    for item in chunks:
        text = item["chunk"]["content"]
        if len(context) + len(text) > max_chars:
            break
        context += text + "\n---\n"
    return context.strip()


def build_prompt(question, context):
    return f"""You are a helpful assistant answering questions based on the provided context.

Context:
{context}

Question:
{question}

Answer:"""


def answer_question(query, top_k=5):
    logger.info("Received query: %s", query)
    top_chunks = query_faiss(query, top_k)

    if not top_chunks:
        logger.warning("No relevant chunks found for query.")
        return "Sorry, I couldn't find relevant information to answer your question."

    context = build_context(top_chunks)
    prompt = build_prompt(query, context)

    logger.info("Sending prompt to LLM (context length: %d chars)", len(context))
    answer = ask_llm(prompt)
    logger.info("LLM response generated.")
    return answer
