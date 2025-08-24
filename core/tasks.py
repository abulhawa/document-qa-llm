from typing import Dict, Any

from celery import shared_task
from celery.exceptions import Ignore

from core.paths import to_worker_path
from core.job_guard import ensure_job_can_start
from core.job_control import incr_stat
from core.job_queue import rem_active, add_retry
from core.checksum import file_checksum, chunk_id
from core.qdrant_bootstrap import ensure_doc_collection, COLLECTION, QDRANT_URL
from dq_loaders_langchain.loader import load_with_langchain
from dq_splitters_langchain.splitter import split_with_langchain
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from config import CHUNK_SIZE, CHUNK_OVERLAP, logger


@shared_task(bind=True, acks_late=True, autoretry_for=(), retry_backoff=True, max_retries=None)
def ingest_file_task(self, *, job_id: str, path: str) -> Dict[str, Any]:
    ensure_job_can_start(job_id, task=self)   # boundary gate only
    real_path = to_worker_path(path)
    try:
        ensure_doc_collection()
        checksum = file_checksum(real_path)

        docs = load_with_langchain(real_path)
        if not docs:
            incr_stat(job_id, "failed", 1); raise Ignore()

        chunks = split_with_langchain(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        texts = [c.text for c in chunks]

        from core.embedder import embed_texts
        embeds = []
        for i in range(0, len(texts), 32):
            embeds.extend(embed_texts(texts[i:i+32]))

        client = QdrantClient(url=QDRANT_URL)
        points = []
        for idx, (c, v) in enumerate(zip(chunks, embeds)):
            meta = dict(c.metadata)
            meta["path"] = meta.get("path") or meta.get("source") or path
            meta["checksum"] = checksum
            meta["chunk_index"] = idx
            meta["text"] = c.text
            points.append(PointStruct(id=chunk_id(checksum, idx), vector=v, payload=meta))
        client.upsert(collection_name=COLLECTION, points=points)

        incr_stat(job_id, "done", 1)
        return {"path": path, "chunks": len(chunks)}
    except Exception as e:
        add_retry(job_id, path)    # restart the whole file on resume
        logger.warning("Ingest interrupted; will restart on resume: %s (%s)", path, e)
        raise
    finally:
        rem_active(job_id, path)
