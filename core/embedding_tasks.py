from worker.celery_worker import app
from core.vector_store import index_chunks
from utils.opensearch_utils import set_has_embedding_true_by_ids
from config import logger
import time
from celery import Task


class LoggedTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"✅ task={self.name} id={task_id} ok")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"❌ task={self.name} id={task_id} failed: {exc}")


@app.task(
    name="core.embedding_tasks.embed_and_index_chunks",
    bind=True,
    base=LoggedTask,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=5,
    acks_late=True,
)
def embed_and_index_chunks(self, chunks: list[dict]):
    t0 = time.perf_counter()
    task_id = self.request.id
    batch_size = len(chunks)
    logger.info(f"🚀 [Task {task_id}] Received batch of {batch_size} chunks.")

    try:
        # 1) Embed + upsert to Qdrant (idempotent)
        index_chunks(chunks)
        logger.info(f"✅ [Task {task_id}] Indexed {batch_size} chunks to Qdrant.")

        # 2) Flip has_embedding in OpenSearch by _id (which == chunk['id'])
        os_ids = [c["id"] for c in chunks]
        if os_ids:
            updated, errors = set_has_embedding_true_by_ids(os_ids)
            logger.info(
                f"🔖 [Task {task_id}] has_embedding=true for {updated} docs (errors={errors})"
            )
        else:
            logger.warning(
                f"ℹ️ [Task {task_id}] No OpenSearch IDs found in chunk batch; skipped OS flip"
            )

    except Exception as e:
        logger.warning(
            "Transient failure in embed_and_index_chunks; will retry", exc_info=True
        )
        raise
    finally:
        dt = time.perf_counter() - t0
        logger.info(f"🏁 [Task {task_id}] done in {dt:.2f}s (batch={batch_size})")
