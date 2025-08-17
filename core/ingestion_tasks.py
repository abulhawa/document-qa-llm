from celery import shared_task
from typing import List, Dict, Any
from config import logger
from utils.file_utils import normalize_path
from utils.opensearch_utils import index_documents, set_has_embedding_true_by_ids
from utils import qdrant_utils


@shared_task(
    name="core.ingestion_tasks.index_and_embed_chunks",
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    retry_backoff=True,
    retry_backoff_max=60,
)
def index_and_embed_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    One Celery task per batch of chunks.

    Steps:
      1) Index to OpenSearch (uses _id = chunk['id'])
      2) Embed + upsert to Qdrant
      3) Flip has_embedding=true in OpenSearch for those ids

    Progress reporting (coarse, low overhead):
      - state="PROGRESS", meta={"stage": "...", "progress": 0.33/0.66/0.95, "processed": n, "total": N}
    """
    if not chunks:
        return {"ok": True, "indexed": 0, "upserted": 0, "flipped": 0}

    for c in chunks:
        if c.get("path"):
            c["path"] = normalize_path(c["path"])

    total = len(chunks)
    try:
        # 1) Index to OpenSearch (chunks only)
        os_chunks = [
            {
                "id": c["id"],
                "parent_id": c["parent_id"],
                "join_field": {"name": "chunk", "parent": c["parent_id"]},
                "text": c["text"],
                "page": c.get("page"),
                "chunk_index": c["chunk_index"],
                "location_percent": c.get("location_percent"),
                "has_embedding": c.get("has_embedding", False),
                "path": c.get("path"),
            }
            for c in chunks
        ]
        index_documents(os_chunks)
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "indexed_os",
                "progress": 0.33,
                "processed": 0,
                "total": total,
            },
        )

        # 2) Embed + upsert to Qdrant
        ok = qdrant_utils.index_chunks(chunks)
        if not ok:
            raise RuntimeError("Qdrant upsert failed for batch")
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "upserted_qdrant",
                "progress": 0.66,
                "processed": total,
                "total": total,
            },
        )

        # 3) Flip OS flag
        ids = [c["id"] for c in chunks if c.get("id")]
        updated, errors = set_has_embedding_true_by_ids(ids)
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "flipped_flags",
                "progress": 0.95,
                "processed": total,
                "total": total,
            },
        )

        return {
            "ok": True,
            "indexed": total,
            "upserted": total,
            "flipped": updated,
            "flip_errors": errors,
        }

    except Exception as e:
        logger.exception("❌ index_and_embed_chunks failed: %s", e)
        raise self.retry(exc=e)
