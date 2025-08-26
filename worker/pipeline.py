from typing import Dict
from worker.file_loader import load_document_bytes, sniff_mime
from worker.chunking import split_into_chunks
from worker.embeddings import embed_chunks_batched
from worker.qdrant_io import upsert_vectors
from worker.opensearch_io import index_fulltext, mark_has_embedding
from worker.utils import file_checksum

def ingest_one(fs_path: str) -> Dict:
    # 1) Read bytes (from /host-c or /staging etc.)
    data = load_document_bytes(fs_path)
    checksum = file_checksum(data)

    # 2) Idempotency gate (skip if already embedded)
    if mark_has_embedding(checksum, check_only=True):  # returns True if already done
        return {"checksum": checksum, "skipped": True, "reason": "already_embedded"}

    mime = sniff_mime(fs_path, data)
    chunks = split_into_chunks(fs_path, data, mime)  # [{text, meta}, ...]

    # 3) Embed
    vectors = embed_chunks_batched([c["text"] for c in chunks])

    # 4) Index
    upsert_vectors(checksum, chunks, vectors)     # Qdrant
    index_fulltext(checksum, chunks)              # OpenSearch
    mark_has_embedding(checksum, check_only=False)

    return {"checksum": checksum, "n_chunks": len(chunks), "skipped": False}
