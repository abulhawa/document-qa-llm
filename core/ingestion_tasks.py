from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone
from typing import Iterable

from celery import group

from worker.celery_worker import app
from core.file_loader import load_documents
from core.chunking import split_documents
from core.opensearch_store import index_documents
from core.embedding_tasks import embed_and_index_chunks, LoggedTask
from utils.file_utils import compute_checksum, get_file_timestamps
from config import logger


@app.task(
    name="core.ingestion_tasks.ingest_paths",
    bind=True,
    base=LoggedTask,
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=600,
    retry_jitter=True,
    max_retries=3,
    soft_time_limit=1800,
    time_limit=2100,
)
def ingest_paths(
    self,
    paths: Iterable[str],
    force: bool = False,
    replace: bool = False,
    batch_size: int = 32,
):
    """Ingest one or more file paths and dispatch embedding jobs."""
    t0 = time.perf_counter()
    task_id = self.request.id
    paths = list(paths)
    total_files = len(paths)
    logger.info(f"🚀 [Task {task_id}] ingesting {total_files} file(s)")

    embed_jobs = []

    try:
        for idx, path in enumerate(paths):
            file_start = time.perf_counter()
            logger.info(f"📥 [Task {task_id}] processing {path}")
            normalized = os.path.normpath(path).replace("\\", "/")
            checksum = compute_checksum(normalized)
            timestamps = get_file_timestamps(normalized)

            docs = load_documents(normalized)
            chunks = split_documents(docs)
            indexed_at = datetime.now(timezone.utc).isoformat()

            for i, chunk in enumerate(chunks):
                chunk["id"] = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{checksum}-{i}"))
                chunk["chunk_index"] = i
                chunk["path"] = normalized
                chunk["checksum"] = checksum
                chunk["indexed_at"] = indexed_at
                chunk["created_at"] = timestamps.get("created")
                chunk["modified_at"] = timestamps.get("modified")

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                batch_start = time.perf_counter()
                index_documents(batch)
                batch_dt = time.perf_counter() - batch_start
                logger.info(
                    f"📝 [Task {task_id}] Indexed batch of {len(batch)} chunks in {batch_dt:.2f}s"
                )
                embed_jobs.append(embed_and_index_chunks.s(batch))

            files_done = idx + 1
            self.update_state(
                state="PROGRESS",
                meta={
                    "msg": f"Processed {normalized}",
                    "files_done": files_done,
                    "total_files": total_files,
                    "queued_jobs": len(embed_jobs),
                },
            )
            file_dt = time.perf_counter() - file_start
            logger.info(f"🏁 [Task {task_id}] Finished {normalized} in {file_dt:.2f}s")

        if embed_jobs:
            group(embed_jobs).apply_async()

        return {
            "status": "ok",
            "files_done": total_files,
            "queued_jobs": len(embed_jobs),
        }
    except Exception:
        logger.warning("Transient failure in ingest_paths; will retry", exc_info=True)
        raise
    finally:
        dt = time.perf_counter() - t0
        logger.info(
            f"🏁 [Task {task_id}] ingest_paths done in {dt:.2f}s (files={total_files})"
        )
