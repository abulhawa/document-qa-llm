"""
ingestion/orchestrator.py
=========================
The single-document ingestion pipeline. This is the main coordinator — every
other ingestion module is called from here in a strict sequence.

Entry point: ingest_one(path, ...)

Pipeline stages (in order)
--------------------------
1. FINGERPRINT    — compute checksum + file size + timestamps (io_loader)
2. DEDUP CHECK    — skip if already indexed at same path+checksum, or handle
                    cross-path duplicates (same checksum, different path)
3. LOAD           — parse raw file into LangChain Document objects (io_loader)
4. PREPROCESS     — clean text: strip headers/footers, fix hyphenation,
                    remove junk lines (preprocess / core.document_preprocessor)
5. CLASSIFY       — infer doc_type, person_name, authority_rank (doc_classifier)
6. CHUNK          — split into overlapping chunks (core.chunking)
7. ENRICH         — extract financial metadata + transaction records (financial_extractor)
8. EMBED + STORE  — embed chunks → Qdrant (dense vectors)
                    index chunks + full text → OpenSearch (BM25)
9. FINANCIAL UPSERT — write transaction records to financial_records index
10. INVENTORY     — update per-path stats (chunk count, last indexed timestamp)

Metadata flow
-------------
classify_document() returns a dict with doc_type, person_name, etc.
  └─ _merge_identity_metadata() copies it into full_doc AND every chunk.

extract_financial_enrichment() returns document_metadata + records.
  └─ _merge_financial_metadata() copies metadata into full_doc AND every chunk.
  └─ financial_records are upserted separately via financial_records_store.

All chunks therefore carry the same doc_type and financial metadata,
which enables filtering in retrieval (e.g. filter by doc_type="invoice").

Observability
-------------
Each ingest_one() call is wrapped in an IngestLogger context (log_factory).
Stages that fail call log.fail() with a stage label before re-raising.
Successful completion calls log.done(status=...). This feeds the
Ingestion Logs tab in the Streamlit UI.
"""

import os
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Protocol, Sequence

from config import logger
from ingestion import io_loader, preprocess, storage
from ingestion.doc_classifier import classify_document
from ingestion.financial_extractor import extract_financial_enrichment
from ingestion.financial_records_store import upsert_financial_records
from utils.opensearch_utils import ensure_financial_metadata_mappings
from utils.ingest_logging import IngestLogEmitter
from utils.file_utils import choose_canonical_path
from utils.timing import timed_block


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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        return self._emitter.__exit__(exc_type, exc, tb)

    def set(self, **fields: Any) -> None:  # noqa: D401
        self._emitter.set(**fields)

    def fail(self, *, stage: str, error_type: str, reason: str) -> None:  # noqa: D401
        self._emitter.fail(stage=stage, error_type=error_type, reason=reason)

    def done(self, *, status: str) -> None:  # noqa: D401
        self._emitter.done(status=status)


def default_log_factory(path: str, op: str, source: str) -> IngestLogger:
    return IngestLogAdapter(IngestLogEmitter(path=path, op=op, source=source))


def _merge_identity_metadata(target: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """
    Copy doc classification fields (doc_type, person_name, etc.) into a chunk
    or full-text document dict. Skips keys with None values to avoid overwriting
    existing data with nulls.
    """
    for key in (
        "doc_type",
        "doc_type_confidence",
        "doc_type_source",
        "person_name",
        "authority_rank",
    ):
        value = metadata.get(key)
        if value is not None:
            target[key] = value


def _merge_financial_metadata(target: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """
    Copy financial enrichment fields (is_financial_document, transaction_dates, etc.)
    into a chunk or full-text document dict. Same None-skip behaviour as above.
    Both functions exist (rather than one generic merge) to make the field sets
    explicit and auditable.
    """
    for key in (
        "is_financial_document",
        "document_date",
        "mentioned_years",
        "transaction_dates",
        "tax_years_referenced",
        "amounts",
        "counterparties",
        "tax_relevance_signals",
        "expense_category",
        "financial_record_type",
        "financial_metadata_version",
        "financial_metadata_source",
    ):
        value = metadata.get(key)
        if value is not None:
            target[key] = value


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
      2) build chunk metadata (deterministic checksum-based ids)
      3) vectors first (embed + Qdrant), then OpenSearch (chunks + full text)
    """

    normalized_path, io_path = io_loader.normalize_paths(path, fs_path)
    logger.info("📥 Starting ingestion for: %s", path)
    ext = os.path.splitext(normalized_path)[1].lower().lstrip(".")
    log_factory = log_factory or default_log_factory
    inventory_writer = inventory_writer or DefaultInventoryWriter()
    log = log_factory(normalized_path, op, source)
    doc_metadata: Dict[str, Any] = {}
    financial_metadata: Dict[str, Any] = {}
    financial_records: list[Dict[str, Any]] = []

    with log:
        checksum, size_bytes, timestamps = io_loader.file_fingerprint(io_path)
        path_fulltext = storage.get_fulltext_for_path(normalized_path)
        existing_fulltext = storage.get_existing_fulltext(checksum)

        if path_fulltext and path_fulltext.get("checksum") != checksum:
            stale_checksum = path_fulltext.get("checksum")
            stale_path = path_fulltext.get("path") or normalized_path
            logger.info(
                "♻️ Path previously indexed with different checksum. "
                "Removing stale artifacts for path=%s (old_checksum=%s, new_checksum=%s)",
                normalized_path,
                stale_checksum,
                checksum,
            )
            try:
                storage.replace_existing_artifacts(stale_path)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to replace artifacts for stale path=%s: %s",
                    stale_path,
                    e,
                )

        existing_path = existing_fulltext.get("path") if existing_fulltext else None
        existing_aliases: list[str] = []
        if existing_fulltext:
            existing_aliases = existing_fulltext.get("aliases") or []
        all_paths = [normalized_path]
        if existing_path:
            all_paths.append(existing_path)
        all_paths.extend(existing_aliases)
        canonical_path = choose_canonical_path(all_paths)
        aliases = sorted({p for p in all_paths if p and p != canonical_path})
        existing_paths = {p for p in [existing_path, *existing_aliases] if p}

        log.set(
            checksum=checksum,
            path_hash=io_loader.hash_path(canonical_path),
            bytes=size_bytes,
            size=io_loader.format_file_size(size_bytes),
        )

        created = timestamps.get("created")
        modified = timestamps.get("modified")
        indexed_at = datetime.now().astimezone().isoformat()

        if not force and existing_fulltext:
            if normalized_path in existing_paths:
                logger.info("Checksum already indexed: %s", normalized_path)
                log.done(status="Already indexed")
                return {
                    "success": True,
                    "status": "Already indexed",
                    "path": normalized_path,
                }
            logger.info(
                "Checksum already indexed under a different path; skipping ingest for %s",
                normalized_path,
            )
            full_doc_id = existing_fulltext.get("id") or checksum
            full_doc = existing_fulltext or {}
            full_doc.update(
                {
                    "id": full_doc_id,
                    "path": canonical_path,
                    "aliases": aliases,
                    "filename": os.path.basename(canonical_path),
                    "filetype": ext,
                    "modified_at": modified,
                    "created_at": created,
                    "indexed_at": indexed_at,
                    "size_bytes": size_bytes,
                    "checksum": checksum,
                }
            )
            try:
                storage.index_fulltext(full_doc)
            except Exception as e:  # noqa: BLE001
                logger.warning("OpenSearch full-text indexing failed: %s", e)
                log.fail(stage="index_fulltext", error_type=e.__class__.__name__, reason=str(e))
                raise RuntimeError(
                    f"OpenSearch full-text indexing failed for {normalized_path}: {e}"
                ) from e
            log.done(status="Duplicate checksum")
            return {
                "success": True,
                "status": "Duplicate checksum",
                "path": normalized_path,
            }

        # Skip only if same path and checksum; allow duplicates across paths
        if not force and storage.is_file_up_to_date(checksum, normalized_path):
            logger.info("File already indexed and unchanged: %s", normalized_path)
            log.done(status="Already indexed")
            return {
                "success": True,
                "status": "Already indexed",
                "path": normalized_path,
            }

        is_dup = False
        if not force and storage.is_duplicate_checksum(checksum, normalized_path):
            logger.info("Duplicate file detected: %s", normalized_path)
            is_dup = True

        with timed_block(
            "step.content.parse",
            extra={"path": normalized_path, "extension": ext},
            logger=logger,
        ):
            logger.info("📄 Loading: %s (fs: %s)", normalized_path, io_path)
            try:
                docs = io_loader.load_file_documents(io_path)
            except Exception as e:  # noqa: BLE001
                logger.error("❌ Failed to load document: %s", e)
                log.fail(stage="load", error_type=e.__class__.__name__, reason=str(e))
                raise RuntimeError(f"Failed to load document {normalized_path}: {e}") from e

            logger.info("🧼 Preprocessing %s documents", len(docs))
            docs_list = list(preprocess.preprocess_documents(docs, normalized_path, ext))

            logger.info("📝 Indexing full document text")
            full_text = preprocess.build_full_text(docs_list)
            doc_metadata = classify_document(
                path=canonical_path,
                filetype=ext,
                full_text=full_text,
            )
            if not full_text:
                full_doc_id = checksum
                if existing_fulltext and existing_fulltext.get("id"):
                    full_doc_id = existing_fulltext["id"]
                full_doc = existing_fulltext or {}
                full_doc.update(
                    {
                        "id": full_doc_id,
                        "path": canonical_path,
                        "aliases": aliases,
                        "filename": os.path.basename(canonical_path),
                        "filetype": ext,
                        "modified_at": modified,
                        "created_at": created,
                        "indexed_at": indexed_at,
                        "size_bytes": size_bytes,
                        "checksum": checksum,
                        "text_full": "",
                    }
                )
                _merge_identity_metadata(full_doc, doc_metadata)
                try:
                    storage.index_fulltext(full_doc)
                except Exception as e:  # noqa: BLE001
                    logger.warning("OpenSearch full-text indexing failed: %s", e)
                    log.fail(stage="index_fulltext", error_type=e.__class__.__name__, reason=str(e))
                    raise RuntimeError(
                        f"OpenSearch full-text indexing failed for {normalized_path}: {e}"
                    ) from e
                inventory_writer.set_number_of_chunks(canonical_path, 0)
                inventory_writer.set_last_indexed(canonical_path, indexed_at)
                logger.warning("No valid content found in: %s", normalized_path)
                log.done(status="No valid content found")
                return {
                    "success": True,
                    "status": "No valid content found",
                    "path": normalized_path,
                    "num_chunks": 0,
                }

            full_doc_id = checksum
            if existing_fulltext and existing_fulltext.get("id"):
                full_doc_id = existing_fulltext["id"]

            full_doc = existing_fulltext or {}
            full_doc.update(
                {
                    "id": full_doc_id,
                    "path": canonical_path,
                    "aliases": aliases,
                    "filename": os.path.basename(canonical_path),
                    "filetype": ext,
                    "modified_at": modified,
                    "created_at": created,
                    "indexed_at": indexed_at,
                    "size_bytes": size_bytes,
                    "checksum": checksum,
                    "text_full": full_text,
                }
            )
            _merge_identity_metadata(full_doc, doc_metadata)

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

    def _chunk_id(checksum_val: str, index: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{checksum_val}:{index}"))

    for i, chunk in enumerate(chunks):
        chunk["id"] = _chunk_id(checksum, i)
        chunk["chunk_index"] = i
        chunk["path"] = canonical_path
        chunk["checksum"] = checksum
        chunk["chunk_char_len"] = len(chunk.get("text") or "")
        chunk["filetype"] = ext
        chunk["indexed_at"] = indexed_at
        chunk["created_at"] = created
        chunk["modified_at"] = modified
        chunk["bytes"] = size_bytes
        chunk["size"] = io_loader.format_file_size(size_bytes)
        chunk["page"] = chunk.get("page", None)
        chunk["location_percent"] = round((i / max(len(chunks) - 1, 1)) * 100)
        _merge_identity_metadata(chunk, doc_metadata)

    try:
        ensure_financial_metadata_mappings()
        financial_result = extract_financial_enrichment(
            path=canonical_path,
            full_text=full_text,
            chunks=chunks,
            doc_type=str(doc_metadata.get("doc_type") or ""),
            checksum=checksum,
            document_id=str(full_doc.get("id") or checksum),
            enable_llm_fallback=True,
        )
        financial_metadata = financial_result.document_metadata
        financial_records = list(financial_result.records)
        _merge_financial_metadata(full_doc, financial_metadata)
        for chunk in chunks:
            _merge_financial_metadata(chunk, financial_metadata)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Financial enrichment skipped for %s: %s", normalized_path, exc)
        financial_metadata = {}
        financial_records = []

    if force and replace:
        storage.replace_existing_artifacts(canonical_path)

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

    if financial_records:
        try:
            upsert_stats = upsert_financial_records(financial_records)
            log.set(
                financial_records_processed=upsert_stats.get("processed", 0),
                financial_records_created=upsert_stats.get("created", 0),
                financial_records_updated=upsert_stats.get("updated", 0),
                financial_records_errors=upsert_stats.get("errors", 0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Financial sidecar upsert failed for path=%s checksum=%s: %s",
                normalized_path,
                checksum,
                exc,
            )

    inventory_writer.set_number_of_chunks(canonical_path, len(chunks))
    inventory_writer.set_last_indexed(canonical_path, indexed_at)

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
