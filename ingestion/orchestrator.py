import os
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Protocol, Sequence

from config import logger
from ingestion import io_loader, preprocess, storage
from utils.ingest_logging import IngestLogEmitter


class InventoryWriter(Protocol):
    def set_number_of_chunks(self, path: str, count: int) -> None: ...

    def set_last_indexed(self, path: str, indexed_at: str) -> None: ...


class IngestLogger(AbstractContextManager, Protocol):
    def set(self, **fields: Any) -> None: ...

    def fail(self, *, stage: str, error_type: str, reason: str) -> None: ...

    def done(self, *, status: str) -> None: ...


IngestLogFactory = Callable[[str, str, str], IngestLogger]


@dataclass
class DefaultInventoryWriter:
    def set_number_of_chunks(self, path: str, count: int) -> None:
        from utils.inventory import set_inventory_number_of_chunks

        try:
            set_inventory_number_of_chunks(path, count)
        except Exception:  # noqa: BLE001
            pass

    def set_last_indexed(self, path: str, indexed_at: str) -> None:
        from utils.inventory import set_inventory_last_indexed

        try:
            set_inventory_last_indexed(path, indexed_at)
        except Exception:  # noqa: BLE001
            pass


class IngestLogAdapter(IngestLogger):
    def __init__(self, emitter: IngestLogEmitter):
        self._emitter = emitter

    def __enter__(self):
        self._emitter.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._emitter.__exit__(exc_type, exc, tb)

    def set(self, **fields: Any) -> None:  # noqa: D401
        self._emitter.set(**fields)

    def fail(self, *, stage: str, error_type: str, reason: str) -> None:  # noqa: D401
        self._emitter.fail(stage=stage, error_type=error_type, reason=reason)

    def done(self, *, status: str) -> None:  # noqa: D401
        self._emitter.done(status=status)


def default_log_factory(path: str, op: str, source: str) -> IngestLogger:
    return IngestLogAdapter(IngestLogEmitter(path=path, op=op, source=source))


def ingest_one(
    path: str,
    *,
    fs_path: Optional[str] = None,
    force: bool = False,
    replace: bool = True,
    total_files: int = 1,
    op: str = "ingest",
    source: str = "ingest_page",
    inventory_writer: Optional[InventoryWriter] = None,
    log_factory: Optional[IngestLogFactory] = None,
) -> Dict[str, Any]:
    """
    Ingest a single file path:
      1) load + split locally
      2) build chunk metadata (UUIDv5 ids)
      3) vectors first (embed + Qdrant), then OpenSearch (chunks + full text)
    """

    normalized_path, io_path = io_loader.normalize_paths(path, fs_path)
    logger.info("📥 Starting ingestion for: %s", path)
    ext = os.path.splitext(normalized_path)[1].lower().lstrip(".")
    log_factory = log_factory or default_log_factory
    inventory_writer = inventory_writer or DefaultInventoryWriter()
    log = log_factory(normalized_path, op, source)

    with log:
        checksum, size_bytes, timestamps = io_loader.file_fingerprint(io_path)
        log.set(
            checksum=checksum,
            path_hash=io_loader.hash_path(normalized_path),
            bytes=size_bytes,
            size=io_loader.format_file_size(size_bytes),
        )

        # Skip only if same path and checksum; allow duplicates across paths
        if not force and storage.is_file_up_to_date(checksum, normalized_path):
            logger.info("✅ File already indexed and unchanged: %s", normalized_path)
            log.done(status="Already indexed")
            return {
                "success": True,
                "status": "Already indexed",
                "path": normalized_path,
            }

        is_dup = False
        if not force and storage.is_duplicate_checksum(checksum, normalized_path):
            logger.info("♻️ Duplicate file detected: %s", normalized_path)
            is_dup = True

        created = timestamps.get("created")
        modified = timestamps.get("modified")
        indexed_at = datetime.now().astimezone().isoformat()

        logger.info("📄 Loading: %s (fs: %s)", normalized_path, io_path)
        try:
            docs = io_loader.load_file_documents(io_path)
        except Exception as e:  # noqa: BLE001
            logger.error("❌ Failed to load document: %s", e)
            log.fail(stage="load", error_type=e.__class__.__name__, reason=str(e))
            raise RuntimeError(f"Failed to load document {normalized_path}: {e}") from e

        logger.info("🧼 Preprocessing %s documents", len(docs))
        docs_list = preprocess.preprocess_documents(docs, normalized_path, ext)

        logger.info("📝 Indexing full document text")
        full_text = preprocess.build_full_text(docs_list)
        if not full_text:
            logger.warning("⚠️ No valid content found in: %s", normalized_path)
            log.done(status="No valid content found")
            return {
                "success": True,
                "status": "No valid content found",
                "path": normalized_path,
            }

        full_doc = {
            "id": io_loader.hash_path(normalized_path),
            "path": normalized_path,
            "filename": os.path.basename(normalized_path),
            "filetype": ext,
            "modified_at": modified,
            "created_at": created,
            "indexed_at": indexed_at,
            "size_bytes": size_bytes,
            "checksum": checksum,
            "text_full": full_text,
        }

        logger.info("✂️ Splitting document into chunks")
        try:
            chunks = preprocess.chunk_documents(docs_list)
        except Exception as e:  # noqa: BLE001
            logger.error("❌ Failed to split document: %s", e)
            log.fail(stage="extract", error_type=e.__class__.__name__, reason=str(e))
            raise RuntimeError(
                f"Failed to split document {normalized_path}: {e}"
            ) from e

        if not chunks:
            logger.warning("⚠️ No chunks generated from: %s", normalized_path)
            log.done(status="No valid content found")
            return {
                "success": True,
                "status": "No valid content found",
                "path": normalized_path,
                "num_chunks": 0,
            }

        logger.info("🧩 Split into %s chunks", len(chunks))

    for i, chunk in enumerate(chunks):
        chunk["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{normalized_path}-{i}"))
        chunk["chunk_index"] = i
        chunk["path"] = normalized_path
        chunk["checksum"] = checksum
        chunk["filetype"] = ext
        chunk["indexed_at"] = indexed_at
        chunk["created_at"] = created
        chunk["modified_at"] = modified
        chunk["bytes"] = size_bytes
        chunk["size"] = io_loader.format_file_size(size_bytes)
        chunk["page"] = chunk.get("page", None)
        chunk["location_percent"] = round((i / max(len(chunks) - 1, 1)) * 100)

    if force and replace:
        storage.replace_existing_artifacts(normalized_path)

    os_acc: Dict[str, Any] = {"indexed": 0, "errors": []}

    def _os_index_batch(group: Sequence[Dict[str, Any]]) -> None:
        n, errs = storage.index_chunk_batch(group)
        os_acc["indexed"] += int(n)
        if errs:
            os_acc["errors"].extend(errs)

    try:
        logger.info(
            "Embedding + upserting %s chunks to Qdrant in batches (wait=True).",
            len(chunks),
        )
        ok = storage.embed_and_store(chunks, os_index_batch=_os_index_batch)
        if not ok:
            raise RuntimeError("Qdrant upsert returned falsy")
    except Exception as e:  # noqa: BLE001
        logger.error("❌ Vector indexing failed: %s", e)
        log.fail(stage="index_vec", error_type=e.__class__.__name__, reason=str(e))
        raise RuntimeError(f"Vector indexing failed for {normalized_path}: {e}") from e

    try:
        storage.index_fulltext(full_doc)
    except Exception as e:  # noqa: BLE001
        logger.warning("OpenSearch full-text indexing failed: %s", e)
        log.fail(stage="index_fulltext", error_type=e.__class__.__name__, reason=str(e))
        raise RuntimeError(
            f"OpenSearch full-text indexing failed for {normalized_path}: {e}"
        ) from e

    inventory_writer.set_number_of_chunks(normalized_path, len(chunks))
    inventory_writer.set_last_indexed(normalized_path, indexed_at)

    final_status = "Duplicate & Indexed" if is_dup else "Success"
    log.done(status=final_status)
    return {
        "success": True,
        "num_chunks": len(chunks),
        "path": normalized_path,
        "status": final_status,
    }


__all__ = [
    "InventoryWriter",
    "IngestLogFactory",
    "IngestLogger",
    "ingest_one",
]
