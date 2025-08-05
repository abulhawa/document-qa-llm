from worker.celery_worker import app
from core.vector_store import index_chunks
from config import logger

@app.task(name="core.embedding_tasks.embed_and_index_chunks", bind=True)
def embed_and_index_chunks(self, chunks: list[dict]):
    task_id = self.request.id
    logger.info(f"🚀 [Task {task_id}] Received batch of {len(chunks)} chunks.")

    try:
        index_chunks(chunks)
        logger.info(f"✅ [Task {task_id}] Indexed {len(chunks)} chunks to Qdrant.")
    except Exception as e:
        logger.exception(f"❌ [Task {task_id}] Embedding/indexing failed: {e}")
        raise