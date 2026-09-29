"""Seed reusable benchmark cache indexes from immutable HF bucket artifacts.

The indexes are content addressed; data files remain in their frozen v1 locations.
Each reference is backed by a checked artifact file hash. This command never deletes
or rewrites a bucket object. Run without --apply to validate and preview first.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import struct
from collections import defaultdict
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any


TRACKS = ("open_ragbench", "officeqa", "nfcorpus", "miracl_de", "miracl_ar")
DEFAULT_BUCKET = "abulhawa/document-qa-artifacts"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def signature(manifest: dict[str, Any], stage: str) -> str:
    fields = {
        "parsed": ("parser_fingerprint", "parser_config", "package_versions", "python_version"),
        "chunks": ("chunker_fingerprint", "chunker_config", "package_versions", "python_version"),
        "embeddings": ("embedder_fingerprint", "model", "package_versions", "python_version"),
    }[stage]
    return digest({"stage": stage, **{key: manifest.get(key) for key in fields}})


def checked_path(path: str) -> str:
    parts = PurePosixPath(path).parts
    if not parts or path.startswith("/") or any(part in (".", "..") for part in parts):
        raise ValueError(f"unsafe bucket path: {path!r}")
    return path


def npy_shape(data: bytes) -> tuple[int, int, int]:
    if not data.startswith(b"\x93NUMPY"):
        raise ValueError("embedding file is not NPY")
    major = data[6]
    if major == 1:
        header_size = struct.unpack_from("<H", data, 8)[0]
        start = 10
    elif major in (2, 3):
        header_size = struct.unpack_from("<I", data, 8)[0]
        start = 12
    else:
        raise ValueError("unsupported NPY version")
    header = ast.literal_eval(data[start : start + header_size].decode("latin1"))
    shape = header["shape"]
    if header["descr"] not in ("<f4", "=f4") or header["fortran_order"] or len(shape) != 2:
        raise ValueError("expected row-major float32 embedding matrix")
    if len(data) != start + header_size + shape[0] * shape[1] * 4:
        raise ValueError("embedding matrix byte count does not match NPY header")
    return int(shape[0]), int(shape[1]), start + header_size


class BucketReader:
    def __init__(self, bucket: str, directory: Path):
        self.bucket = bucket
        self.directory = directory
        self.counter = 0
        from huggingface_hub import download_bucket_files

        self.download = download_bucket_files

    def read(self, path: str) -> bytes:
        self.counter += 1
        destination = self.directory / str(self.counter)
        self.download(self.bucket, files=[(checked_path(path), str(destination))])
        try:
            return destination.read_bytes()
        finally:
            destination.unlink(missing_ok=True)

    def verified(self, root: str, manifest: dict[str, Any], relative: str) -> bytes:
        entries = {item["path"]: item for item in manifest["files"]}
        if relative not in entries:
            raise ValueError(f"{relative} is absent from the artifact manifest")
        data = self.read(f"{root}/{relative}")
        item = entries[relative]
        if len(data) != item["bytes"] or sha256(data) != item["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {root}/{relative}")
        return data


def jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def artifact_root(benchmark: str, stage: str, fingerprint: str) -> str:
    return f"{benchmark}/{stage}/{fingerprint}"


def load_lineage(reader: BucketReader, benchmark: str) -> tuple[dict[str, Any], dict[str, tuple[str, dict[str, Any], str]]]:
    pointers = {
        stage: json.loads(reader.read(f"{benchmark}/{stage}/current.json"))
        for stage in ("source", "chunks", "embeddings")
    }
    fingerprints = {
        "source": pointers["source"]["artifact_fingerprint"],
        "parsed": pointers["chunks"]["parent_parsed_artifact_fingerprint"],
        "chunks": pointers["chunks"]["artifact_fingerprint"],
        "embeddings": pointers["embeddings"]["artifact_fingerprint"],
    }
    result = {}
    for stage, fingerprint in fingerprints.items():
        root = artifact_root(benchmark, stage, fingerprint)
        raw = reader.read(f"{root}/manifest.json")
        manifest = json.loads(raw)
        if manifest.get("artifact_fingerprint") != fingerprint or manifest.get("artifact_type") != stage:
            raise ValueError(f"invalid {stage} artifact manifest")
        if manifest.get("benchmark_id") != benchmark or not manifest.get("persistence", {}).get("persisted"):
            raise ValueError(f"{stage} artifact is not a persisted {benchmark} artifact")
        result[stage] = (root, manifest, sha256(raw))
    source, parsed, chunks, embeddings = (result[stage][1] for stage in result)
    if parsed["parent_source_artifact_fingerprint"] != source["artifact_fingerprint"]:
        raise ValueError("parsed/source lineage mismatch")
    if chunks["parent_parsed_artifact_fingerprint"] != parsed["artifact_fingerprint"]:
        raise ValueError("chunks/parsed lineage mismatch")
    if embeddings["parent_chunks_artifact_fingerprint"] != chunks["artifact_fingerprint"]:
        raise ValueError("embeddings/chunks lineage mismatch")
    return fingerprints, result


def document_digest(document: dict[str, Any], pages: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "page_index": row["page_index"],
            "text": row["text"],
            "metadata": {key: value for key, value in row["metadata"].items() if key != "source"},
        }
        for row in sorted(pages, key=lambda row: row["page_index"])
    ]
    return digest({"filetype": document["filetype"], "pages": normalized})


def chunks_digest(rows: list[dict[str, Any]]) -> str:
    fields = ("chunk_index", "text", "page", "location_percent", "filetype")
    return digest([{key: row.get(key) for key in fields} for row in sorted(rows, key=lambda row: row["chunk_index"])])


def index_bytes(rows: list[dict[str, Any]]) -> bytes:
    lines = b"".join(canonical(row) + b"\n" for row in rows)
    return gzip.compress(lines, compresslevel=9, mtime=0)


def build_indexes(reader: BucketReader, benchmark: str) -> tuple[dict[str, bytes], dict[str, Any]]:
    fingerprints, artifacts = load_lineage(reader, benchmark)
    source_root, source_manifest, _ = artifacts["source"]
    parsed_root, parsed_manifest, _ = artifacts["parsed"]
    chunks_root, chunks_manifest, _ = artifacts["chunks"]
    embeddings_root, embeddings_manifest, _ = artifacts["embeddings"]
    signatures = {stage: signature(artifacts[stage][1], stage) for stage in ("parsed", "chunks", "embeddings")}
    source_files = {item["path"]: item for item in source_manifest["files"]}
    parsed_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    vector_rows: list[dict[str, Any]] = []
    global_chunks: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    parsed_keys: dict[str, str] = {}
    chunk_keys: dict[str, str] = {}
    vector_keys: dict[str, str] = {}
    ambiguous_vector_keys: set[str] = set()

    for track in TRACKS:
        source_documents = jsonl(reader.verified(source_root, source_manifest, f"{track}/documents.jsonl"))
        parsed_documents = jsonl(reader.verified(parsed_root, parsed_manifest, f"{track}/documents.jsonl"))
        pages = jsonl(reader.verified(parsed_root, parsed_manifest, f"{track}/pages.jsonl"))
        chunk_documents = jsonl(reader.verified(chunks_root, chunks_manifest, f"{track}/documents.jsonl"))
        chunks = jsonl(reader.verified(chunks_root, chunks_manifest, f"{track}/chunks.jsonl"))
        sources = {row["source_document_id"]: row for row in source_documents}
        parsed = {row["source_document_id"]: row for row in parsed_documents}
        chunked = {row["source_document_id"]: row for row in chunk_documents}
        if len(sources) != len(source_documents) or len(parsed) != len(parsed_documents) or len(chunked) != len(chunk_documents):
            raise ValueError(f"duplicate document identity in {track}")
        if sources.keys() != parsed.keys() or sources.keys() != chunked.keys():
            raise ValueError(f"document sets differ in {track}")
        pages_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        chunks_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pages:
            pages_by_id[row["source_document_id"]].append(row)
        for row in chunks:
            chunks_by_id[row["source_document_id"]].append(row)
        for doc_id in sorted(sources):
            src, doc, chunk_doc = sources[doc_id], parsed[doc_id], chunked[doc_id]
            source_file = source_files.get(f"{track}/{src['path']}")
            if not source_file or source_file["sha256"] != src["sha256"] or source_file["bytes"] != src["bytes"]:
                raise ValueError(f"source manifest disagrees with a {track} document")
            if doc["status"] != "parsed" or chunk_doc["status"] != "chunked":
                raise ValueError(f"failed legacy document in {track}")
            if doc["source_sha256"] != src["sha256"] or chunk_doc["source_sha256"] != src["sha256"]:
                raise ValueError(f"source hash lineage mismatch in {track}")
            doc_pages = pages_by_id.get(doc_id, [])
            doc_chunks = chunks_by_id.get(doc_id, [])
            if len(doc_pages) != doc["page_count"] or len(doc_chunks) != chunk_doc["chunk_count"]:
                raise ValueError(f"document row count mismatch in {track}")
            parsed_hash = document_digest(doc, doc_pages)
            chunk_hash = chunks_digest(doc_chunks)
            parsed_key = f"parsed/{signatures['parsed']}/{doc['filetype']}/{src['sha256']}"
            chunk_key = f"chunks/{signatures['chunks']}/{parsed_hash}"
            if parsed_key in parsed_keys and parsed_keys[parsed_key] != parsed_hash:
                raise ValueError(f"conflicting parsed cache key: {parsed_key}")
            if chunk_key in chunk_keys and chunk_keys[chunk_key] != chunk_hash:
                raise ValueError(f"conflicting chunks cache key: {chunk_key}")
            parsed_keys[parsed_key] = parsed_hash
            chunk_keys[chunk_key] = chunk_hash
            parsed_rows.append({
                "cache_key": parsed_key,
                "parsed_sha256": parsed_hash, "source_sha256": src["sha256"],
                "track": track, "source_document_id": doc_id, "page_count": len(doc_pages),
                "artifact_uri": f"hf://buckets/{reader.bucket}/{parsed_root}",
                "documents_file": f"{track}/documents.jsonl", "pages_file": f"{track}/pages.jsonl",
            })
            chunk_rows.append({
                "cache_key": chunk_key,
                "chunks_sha256": chunk_hash, "parsed_sha256": parsed_hash,
                "track": track, "source_document_id": doc_id, "chunk_count": len(doc_chunks),
                "artifact_uri": f"hf://buckets/{reader.bucket}/{chunks_root}",
                "documents_file": f"{track}/documents.jsonl", "chunks_file": f"{track}/chunks.jsonl",
            })
        for row in chunks:
            global_chunks.append({
                "id": row["id"], "text_sha256": sha256(row["text"].encode("utf-8")),
                "track": track, "source_document_id": row["source_document_id"],
                "chunk_index": row["chunk_index"],
            })
        counts[track] = {"documents": len(sources), "pages": len(pages), "chunks": len(chunks)}

    shard_count = int(embeddings_manifest["shard_count"])
    seen: set[int] = set()
    for shard in range(shard_count):
        prefix = f"shards/{shard:03d}"
        records_file = f"{prefix}/records.jsonl"
        matrix_file = f"{prefix}/embeddings.npy"
        records = jsonl(reader.verified(embeddings_root, embeddings_manifest, records_file))
        matrix = reader.verified(embeddings_root, embeddings_manifest, matrix_file)
        rows, dimensions, data_offset = npy_shape(matrix)
        if rows != len(records) or dimensions != embeddings_manifest["vector_dimension"]:
            raise ValueError(f"embedding shape mismatch in shard {shard}")
        for position, record in enumerate(records):
            global_index = int(record["global_index"])
            if global_index in seen or global_index >= len(global_chunks) or global_index < 0:
                raise ValueError("duplicate or out-of-range embedding global index")
            seen.add(global_index)
            chunk = global_chunks[global_index]
            if record["row_index"] != position or global_index % shard_count != shard or any(
                record[field] != chunk[field] for field in ("id", "source_document_id", "chunk_index")
            ) or record["source_benchmark"] != chunk["track"]:
                raise ValueError(f"embedding/chunk row mismatch at {global_index}")
            vector_hash = sha256(matrix[data_offset + position * dimensions * 4:
                                        data_offset + (position + 1) * dimensions * 4])
            vector_key = f"embeddings/{signatures['embeddings']}/{chunk['text_sha256']}"
            if vector_key in vector_keys and vector_keys[vector_key] != vector_hash:
                ambiguous_vector_keys.add(vector_key)
            vector_keys.setdefault(vector_key, vector_hash)
            vector_rows.append({
                "cache_key": vector_key,
                "text_sha256": chunk["text_sha256"], "global_index": global_index,
                "artifact_uri": f"hf://buckets/{reader.bucket}/{embeddings_root}",
                "matrix_file": matrix_file, "matrix_sha256": next(
                    item["sha256"] for item in embeddings_manifest["files"] if item["path"] == matrix_file
                ),
                "row_index": position, "chunk_id": chunk["id"], "vector_sha256": vector_hash,
            })
        del matrix
    if len(seen) != len(global_chunks) or len(seen) != embeddings_manifest["chunk_embeddings"]:
        raise ValueError("not every chunk has exactly one embedding")

    lineage = digest(fingerprints)
    payloads = {
        f"cache/{stage}/{signatures[stage]}/{lineage}.jsonl.gz": index_bytes(rows)
        for stage, rows in (("parsed", parsed_rows), ("chunks", chunk_rows), ("embeddings", vector_rows))
    }
    report = {
        "schema_version": 1, "benchmark_id": benchmark, "lineage_sha256": lineage,
        "source_artifacts": {stage: {"fingerprint": fingerprints[stage], "manifest_sha256": artifacts[stage][2]}
                             for stage in fingerprints},
        "signatures": signatures, "track_counts": counts,
        "cache_records": {"parsed": len(parsed_rows), "chunks": len(chunk_rows), "embeddings": len(vector_rows)},
        "ambiguous_embedding_text_keys": len(ambiguous_vector_keys),
        "verified_input_scope": "source metadata and all consumed JSONL/NPY files against artifact manifests; raw source files were not redownloaded",
        "indexes": {path: {"sha256": sha256(data), "bytes": len(data)} for path, data in payloads.items()},
    }
    payloads[f"cache/migrations/{lineage}.json"] = canonical(report) + b"\n"
    return payloads, report


def persist(reader: BucketReader, payloads: dict[str, bytes]) -> dict[str, int]:
    from huggingface_hub import batch_bucket_files, get_bucket_paths_info

    result = {"created": 0, "reused": 0}
    for path, data in payloads.items():
        if list(get_bucket_paths_info(reader.bucket, [path])):
            if reader.read(path) != data:
                raise ValueError(f"existing cache object differs: {path}")
            result["reused"] += 1
            continue
        batch_bucket_files(reader.bucket, add=[(data, path)])
        if reader.read(path) != data:
            raise ValueError(f"uploaded cache checksum mismatch: {path}")
        result["created"] += 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--benchmark-id", default="composite-v1")
    parser.add_argument("--apply", action="store_true", help="write only missing, verified cache indexes")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="benchmark-cache-seed-") as temp:
        reader = BucketReader(args.bucket, Path(temp))
        payloads, report = build_indexes(reader, args.benchmark_id)
        print(json.dumps({"lineage_sha256": report["lineage_sha256"],
                          "track_counts": report["track_counts"], "cache_records": report["cache_records"],
                          "ambiguous_embedding_text_keys": report["ambiguous_embedding_text_keys"],
                          "index_bytes": sum(map(len, payloads.values())), "mode": "apply" if args.apply else "dry-run"},
                         sort_keys=True))
        if args.apply:
            print(json.dumps({"persistence": persist(reader, payloads)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

