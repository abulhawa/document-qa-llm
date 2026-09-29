"""Reuse verified document work from a persisted predecessor benchmark.

This is a wrapper around the frozen v1 stage implementations. Their source files
remain unchanged, so their code fingerprints retain their original meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts import chunk_benchmark_parsed as chunker
from scripts import checkpoint_benchmark_embeddings as checkpoint
from scripts import embed_benchmark_chunks as embedder
from scripts import parse_benchmark_source as parser
from scripts.seed_benchmark_cache import document_digest, signature


def _manifest(root: Path, stage: str, *, verify_files: bool = True) -> dict[str, Any]:
    value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if value.get("artifact_type") != stage or not value.get("persistence", {}).get("persisted"):
        raise ValueError(f"unverified {stage} base artifact: {root}")
    for file in value.get("files", []) if verify_files else []:
        path = root / file["path"]
        if not path.is_file() or path.stat().st_size != file["bytes"]:
            raise ValueError(f"missing or truncated base file: {path}")
        if parser._sha256_file(path) != file["sha256"]:
            raise ValueError(f"base file checksum mismatch: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return list(parser._jsonl_rows(path))


def _by_document(root: Path, record_name: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for track in parser.TRACKS:
        for row in _rows(root / track / record_name):
            result[(track, str(row["source_document_id"]))].append(row)
    return result


def _write_reuse(root: Path, counts: dict[str, int], base: dict[str, Any]) -> None:
    path = root / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["reuse"] = {
        **counts,
        "base_benchmark_id": base["benchmark_id"],
        "base_artifact_fingerprint": base["artifact_fingerprint"],
    }
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def parse(args: argparse.Namespace) -> None:
    base_source = _manifest(args.base_source_root, "source", verify_files=False)
    base = _manifest(args.base_root, "parsed")
    if base["parent_source_artifact_fingerprint"] != base_source["artifact_fingerprint"]:
        raise ValueError("base parsed/source lineage mismatch")
    compatible = (
        base["parser_fingerprint"] == parser._parser_fingerprint()
        and base["parser_config"]["pypdf_xform_maximum_invocations_per_extraction"] == args.pypdf_xform_limit
        and base["package_versions"] == parser._package_versions()
        and base["python_version"] == platform.python_version()
    )
    source_files = {item["path"]: item for item in base_source["files"]}
    old_sources = {
        (track, str(row["source_document_id"])): row
        for track in parser.TRACKS
        for row in _rows(args.base_source_root / track / "documents.jsonl")
    }
    old_docs = {
        (track, str(row["source_document_id"])): row
        for track in parser.TRACKS
        for row in _rows(args.base_root / track / "documents.jsonl")
    }
    old_pages = _by_document(args.base_root, "pages.jsonl")
    original = parser._parse_one
    counts = {"documents_reused": 0, "documents_computed": 0}

    def cached(task: dict[str, Any]) -> dict[str, Any]:
        track = str(task["track"])
        entry = task["entry"]
        key = (track, str(entry["source_document_id"]))
        old_source = old_sources.get(key)
        old_doc = old_docs.get(key)
        old_file = source_files.get(f"{track}/{old_source['path']}") if old_source else None
        if (
            compatible and old_source and old_doc and old_doc["status"] == "parsed"
            and old_file and old_file["sha256"] == old_source["sha256"]
            and old_file["bytes"] == old_source["bytes"]
            and old_source["sha256"] == entry["sha256"]
            and old_source["bytes"] == entry["bytes"]
            and old_doc["filetype"] == Path(str(entry["path"])).suffix.lower().lstrip(".")
            and len(old_pages.get(key, [])) == old_doc["page_count"]
        ):
            counts["documents_reused"] += 1
            return {"document": {**old_doc, "source_path": entry["path"]}, "pages": old_pages[key], "error": None}
        counts["documents_computed"] += 1
        return original(task)

    parser._parse_one = cached
    try:
        output = parser.parse_source(
            args.source_root, args.output_root, benchmark_id=args.benchmark_id,
            workers=args.workers, max_failures=args.max_failures,
            repo_revision=args.repo_revision,
            parent_source_fingerprint=args.parent_fingerprint,
            pypdf_xform_limit=args.pypdf_xform_limit,
        )
    finally:
        parser._parse_one = original
    _write_reuse(output, counts, base)
    print(json.dumps({"stage": "parsed", "compatible": compatible, **counts}))


def chunk(args: argparse.Namespace) -> None:
    base_parsed = _manifest(args.base_parsed_root, "parsed")
    base = _manifest(args.base_root, "chunks")
    if base["parent_parsed_artifact_fingerprint"] != base_parsed["artifact_fingerprint"]:
        raise ValueError("base chunks/parsed lineage mismatch")
    compatible = (
        base["chunker_fingerprint"] == chunker._chunker_fingerprint()
        and base["chunker_config"] == {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap}
        and base["package_versions"] == chunker._package_versions()
        and base["python_version"] == platform.python_version()
    )
    old_parsed_docs = {
        (track, str(row["source_document_id"])): row
        for track in chunker.TRACKS
        for row in _rows(args.base_parsed_root / track / "documents.jsonl")
    }
    old_pages = _by_document(args.base_parsed_root, "pages.jsonl")
    old_chunk_docs = {
        (track, str(row["source_document_id"])): row
        for track in chunker.TRACKS
        for row in _rows(args.base_root / track / "documents.jsonl")
    }
    old_chunks = _by_document(args.base_root, "chunks.jsonl")
    original = chunker._chunk_document
    counts = {"documents_reused": 0, "documents_computed": 0, "chunks_reused": 0}

    def cached(**kwargs: Any) -> list[dict[str, Any]]:
        track = kwargs["track"]
        document = kwargs["document"]
        key = (track, str(document["source_document_id"]))
        prior = old_parsed_docs.get(key)
        prior_chunked = old_chunk_docs.get(key)
        rows = old_chunks.get(key, [])
        if (
            compatible and prior and prior_chunked and prior_chunked["status"] == "chunked"
            and prior["source_sha256"] == document["source_sha256"]
            and document_digest(prior, old_pages.get(key, [])) == document_digest(document, list(kwargs["pages"]))
            and len(rows) == prior_chunked["chunk_count"]
        ):
            counts["documents_reused"] += 1
            counts["chunks_reused"] += len(rows)
            return rows
        counts["documents_computed"] += 1
        return original(**kwargs)

    chunker._chunk_document = cached
    try:
        output = chunker.chunk_parsed(
            args.parsed_root, args.output_root, benchmark_id=args.benchmark_id,
            parent_parsed_fingerprint=args.parent_fingerprint,
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            max_failures=args.max_failures, repo_revision=args.repo_revision,
        )
    finally:
        chunker._chunk_document = original
    _write_reuse(output, counts, base)
    print(json.dumps({"stage": "chunks", "compatible": compatible, **counts}))


def embed(args: argparse.Namespace) -> None:
    import numpy as np

    base_chunks = _manifest(args.base_chunks_root, "chunks")
    base = _manifest(args.base_root, "embeddings")
    if base["parent_chunks_artifact_fingerprint"] != base_chunks["artifact_fingerprint"]:
        raise ValueError("base embeddings/chunks lineage mismatch")
    expected_model = {
        "name": args.model_name, "revision": args.model_revision,
        "input_format": args.input_format, "document_input_type": "passage",
        "query_input_type": "query", "normalize_embeddings": True,
        "execution_device": args.execution_device, "output_dtype": "float32",
    }
    compatible = (
        base["embedder_fingerprint"] == embedder._embedder_fingerprint()
        and base["model"] == expected_model
        and base["package_versions"] == embedder._package_versions()
        and base["python_version"] == platform.python_version()
    )
    vectors: dict[tuple[str, str, int, str, str], np.ndarray] = {}
    query_vectors: dict[tuple[str, str, str], np.ndarray] = {}
    if compatible:
        chunks = [
            row
            for track in embedder.TRACKS
            for row in _rows(args.base_chunks_root / track / "chunks.jsonl")
        ]
        seen_global_indices: set[int] = set()
        for shard in range(int(base["shard_count"])):
            folder = args.base_root / "shards" / f"{shard:03d}"
            matrix = np.load(folder / "embeddings.npy", allow_pickle=False, mmap_mode="r")
            records = _rows(folder / "records.jsonl")
            if len(matrix) != len(records):
                raise ValueError("base vector row mismatch")
            for i, record in enumerate(records):
                global_index = int(record["global_index"])
                if global_index < 0 or global_index >= len(chunks) or global_index % int(base["shard_count"]) != shard:
                    raise ValueError("base vector global index mismatch")
                if global_index in seen_global_indices:
                    raise ValueError("duplicate base vector global index")
                seen_global_indices.add(global_index)
                row = chunks[global_index]
                if (
                    record["row_index"] != i or record["id"] != row["id"]
                    or record["source_benchmark"] != row["source_benchmark"]
                    or record["source_document_id"] != row["source_document_id"]
                    or record["chunk_index"] != row["chunk_index"]
                ):
                    raise ValueError("base embedding/chunk identity mismatch")
                key = (str(row["source_benchmark"]), str(row["source_document_id"]),
                       int(row["chunk_index"]), str(row["id"]),
                       hashlib.sha256(row["text"].encode()).hexdigest())
                vectors[key] = matrix[i]
        if len(seen_global_indices) != len(chunks):
            raise ValueError("base embedding artifact does not cover every chunk")
        for track in embedder.TRACKS:
            folder = args.base_root / "queries" / track
            if not folder.exists():
                continue
            old_queries = {row["query_id"]: row["text"] for row in _rows(args.base_chunks_root / track / "evaluation" / "queries.jsonl")}
            records = _rows(folder / "records.jsonl")
            matrix = np.load(folder / "embeddings.npy", allow_pickle=False, mmap_mode="r")
            if len(records) != len(matrix):
                raise ValueError("base query vector row mismatch")
            for i, record in enumerate(records):
                qid = record["query_id"]
                query_vectors[(track, qid, old_queries[qid])] = matrix[i]

    query_tracks = [track for track in embedder.TRACKS if (args.chunks_root / track / "evaluation" / "queries.jsonl").exists()]
    original = embedder._encode
    counts = {"vectors_reused": 0, "vectors_computed": 0, "queries_reused": 0, "queries_computed": 0}
    query_cursor = 0

    def passage_vectors(rows: list[dict[str, Any]], model: Any, *, batch_size: int, input_format: str) -> np.ndarray:
        texts = [str(row["text"]) for row in rows]
        keys = [(str(row["source_benchmark"]), str(row["source_document_id"]),
                 int(row["chunk_index"]), str(row["id"]),
                 hashlib.sha256(text.encode()).hexdigest()) for row, text in zip(rows, texts)]
        found = [vectors.get(key) for key in keys]
        missing = [i for i, value in enumerate(found) if value is None]
        if missing:
            computed = original(
                [texts[i] for i in missing], model=model, input_type="passage",
                input_format=input_format, batch_size=batch_size,
            )
            if len(computed) != len(missing):
                raise ValueError("new embedding count mismatch")
            for i, value in zip(missing, computed):
                found[i] = value
        counts["vectors_reused"] += len(keys) - len(missing)
        counts["vectors_computed"] += len(missing)
        return np.stack(found).astype(np.float32)

    def cached(texts: list[str], **kwargs: Any) -> np.ndarray:
        nonlocal query_cursor
        input_type = kwargs["input_type"]
        if input_type == "passage":
            raise ValueError("checkpoint must pass chunk identities to vector provider")
        track = query_tracks[query_cursor]
        query_cursor += 1
        rows = _rows(args.chunks_root / track / "evaluation" / "queries.jsonl")
        if [str(row["text"]) for row in rows] != texts:
            raise ValueError("query order changed during embedding")
        keys = [(track, str(row["query_id"]), str(row["text"])) for row in rows]
        found = [query_vectors.get(key) for key in keys]
        missing = [i for i, value in enumerate(found) if value is None]
        if missing:
            computed = original([texts[i] for i in missing], **kwargs)
            if len(computed) != len(missing):
                raise ValueError("new embedding count mismatch")
            for i, value in zip(missing, computed):
                found[i] = value
        reused = len(keys) - len(missing)
        counts["queries_reused"] += reused
        counts["queries_computed"] += len(missing)
        if found:
            return np.stack(found).astype(np.float32)
        return np.empty((0, int(base["vector_dimension"])), dtype=np.float32)

    embedder._encode = cached
    try:
        common = dict(
            benchmark_id=args.benchmark_id, parent_chunks_fingerprint=args.parent_fingerprint,
            model_name=args.model_name, model_revision=args.model_revision,
            input_format=args.input_format, execution_device=args.execution_device,
            batch_size=args.batch_size, shard_index=args.shard_index,
            shard_count=args.shard_count, repo_revision=args.repo_revision,
        )
        if args.fingerprint_only:
            output = embedder.build_shard(
                args.chunks_root, args.output_root, fingerprint_only=True, **common,
            )
        else:
            output = checkpoint.build_checkpointed_shard(
                args.chunks_root, args.output_root, args.checkpoint_root,
                artifact_fingerprint=args.artifact_fingerprint,
                checkpoint_size=args.checkpoint_size,
                remote_root=args.remote_root,
                vector_provider=passage_vectors,
                **common,
            )
    finally:
        embedder._encode = original
    if not args.fingerprint_only:
        _write_reuse(output / "shards" / f"{args.shard_index:03d}", counts, base)
    print(json.dumps({"stage": "embeddings", "compatible": compatible, **counts}))


def main() -> None:
    cli = argparse.ArgumentParser()
    commands = cli.add_subparsers(dest="stage", required=True)
    for stage in ("parse", "chunk", "embed"):
        command = commands.add_parser(stage)
        command.add_argument("--benchmark-id", required=True)
        command.add_argument("--parent-fingerprint", required=True)
        command.add_argument("--base-root", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--repo-revision", default="local")
        if stage == "parse":
            command.add_argument("--source-root", type=Path, required=True)
            command.add_argument("--base-source-root", type=Path, required=True)
            command.add_argument("--workers", type=int, default=4)
            command.add_argument("--max-failures", type=int, default=0)
            command.add_argument("--pypdf-xform-limit", type=int, default=50000)
        elif stage == "chunk":
            command.add_argument("--parsed-root", type=Path, required=True)
            command.add_argument("--base-parsed-root", type=Path, required=True)
            command.add_argument("--chunk-size", type=int, default=800)
            command.add_argument("--chunk-overlap", type=int, default=100)
            command.add_argument("--max-failures", type=int, default=0)
        else:
            command.add_argument("--chunks-root", type=Path, required=True)
            command.add_argument("--base-chunks-root", type=Path, required=True)
            command.add_argument("--model-name", default="intfloat/multilingual-e5-base")
            command.add_argument("--model-revision", required=True)
            command.add_argument("--input-format", default="e5")
            command.add_argument("--execution-device", default="cpu")
            command.add_argument("--batch-size", type=int, default=16)
            command.add_argument("--shard-index", type=int, required=True)
            command.add_argument("--shard-count", type=int, required=True)
            command.add_argument("--fingerprint-only", action="store_true")
            command.add_argument("--artifact-fingerprint")
            command.add_argument("--checkpoint-root", type=Path)
            command.add_argument("--checkpoint-size", type=int, default=2048)
            command.add_argument("--remote-root")
    args = cli.parse_args()
    {"parse": parse, "chunk": chunk, "embed": embed}[args.stage](args)


if __name__ == "__main__":
    main()

