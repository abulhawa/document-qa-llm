"""Chunk a persisted parsed benchmark artifact with production chunking.

This stage is deterministic and corpus-bearing. It restores no raw source files and
does not re-run parsing. Successful outputs are intended for private durable storage.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from langchain_core.documents import Document

TRACKS = ("open_ragbench", "officeqa", "nfcorpus", "miracl_de", "miracl_ar")
CHUNK_ARTIFACT_SCHEMA_VERSION = 1
CHUNKER_INPUTS = (
    Path("scripts/chunk_benchmark_parsed.py"),
    Path("core/chunking.py"),
    Path("requirements/chunk.txt"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield dict(json.loads(line))


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _chunker_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in CHUNKER_INPUTS:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("langchain-core", "langchain-text-splitters"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def _artifact_fingerprint(
    parent_parsed_fingerprint: str,
    chunker_fingerprint: str,
    package_versions: Mapping[str, str],
    python_version: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    payload = {
        "artifact_schema_version": CHUNK_ARTIFACT_SCHEMA_VERSION,
        "parent_parsed_artifact_fingerprint": parent_parsed_fingerprint,
        "chunker_fingerprint": chunker_fingerprint,
        "package_versions": dict(sorted(package_versions.items())),
        "python_version": python_version,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stable_chunk_id(source_sha256: str, logical_source: str, chunk_index: int) -> str:
    identity = source_sha256 or logical_source
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{identity}:{chunk_index}"))


def _chunk_document(
    *,
    benchmark_id: str,
    track: str,
    document: Mapping[str, Any],
    pages: Sequence[Mapping[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    from core.chunking import split_documents

    source_document_id = str(document["source_document_id"])
    source_sha256 = str(document.get("source_sha256") or "")
    logical_source = f"benchmark://{benchmark_id}/{track}/{source_document_id}"

    langchain_docs: list[Document] = []
    for page in sorted(pages, key=lambda row: int(row.get("page_index", 0))):
        page_index = int(page.get("page_index", 0))
        metadata = dict(page.get("metadata") or {})
        metadata["source"] = logical_source
        metadata["source_document_id"] = source_document_id
        metadata["source_benchmark"] = track
        metadata.setdefault("page", page_index)
        langchain_docs.append(
            Document(
                page_content=str(page.get("text") or ""),
                metadata=metadata,
            )
        )

    raw_chunks = split_documents(
        langchain_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    total = len(raw_chunks)
    output: list[dict[str, Any]] = []
    for index, chunk in enumerate(raw_chunks):
        text = str(chunk.get("text") or "")
        output.append(
            {
                "id": _stable_chunk_id(source_sha256, logical_source, index),
                "source_benchmark": track,
                "source_document_id": source_document_id,
                "source_sha256": source_sha256 or None,
                "text": text,
                "chunk_char_len": len(text),
                "chunk_index": index,
                "page": chunk.get("page"),
                "path": logical_source,
                "location_percent": round((index / max(total - 1, 1)) * 100),
                "filetype": document.get("filetype"),
            }
        )
    return output


def chunk_parsed(
    parsed_root: Path,
    output_root: Path,
    *,
    benchmark_id: str,
    parent_parsed_fingerprint: str,
    chunk_size: int,
    chunk_overlap: int,
    max_failures: int,
    repo_revision: str,
) -> Path:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    parent_manifest_path = parsed_root / "manifest.json"
    if not parent_manifest_path.exists():
        raise FileNotFoundError(f"missing parsed manifest: {parent_manifest_path}")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("artifact_fingerprint") != parent_parsed_fingerprint:
        raise ValueError("parsed manifest fingerprint does not match requested parent")
    if parent_manifest.get("artifact_type") != "parsed":
        raise ValueError("parent artifact is not a parsed benchmark artifact")
    if parent_manifest.get("benchmark_id") != benchmark_id:
        raise ValueError("parsed benchmark_id mismatch")
    if not parent_manifest.get("persistence", {}).get("persisted"):
        raise ValueError("parent parsed artifact is not marked persisted")

    output_dir = output_root / benchmark_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    all_errors: list[dict[str, Any]] = []
    track_stats: dict[str, dict[str, int | float]] = {}

    for track in TRACKS:
        parsed_track = parsed_root / track
        documents_path = parsed_track / "documents.jsonl"
        pages_path = parsed_track / "pages.jsonl"
        if not documents_path.exists() or not pages_path.exists():
            raise FileNotFoundError(f"missing parsed inputs for {track}")

        documents = sorted(
            _jsonl_rows(documents_path),
            key=lambda row: str(row["source_document_id"]),
        )
        pages_by_document: dict[str, list[dict[str, Any]]] = {}
        for page in _jsonl_rows(pages_path):
            doc_id = str(page["source_document_id"])
            pages_by_document.setdefault(doc_id, []).append(page)

        track_out = output_dir / track
        chunk_rows: list[dict[str, Any]] = []
        document_rows: list[dict[str, Any]] = []
        empty_input_pages = 0
        documents_with_zero_chunks = 0
        zero_length_chunks = 0
        total_chunk_chars = 0

        for document in documents:
            doc_id = str(document["source_document_id"])
            pages = pages_by_document.get(doc_id, [])
            empty_input_pages += sum(
                1 for page in pages if not str(page.get("text") or "").strip()
            )
            try:
                chunks = _chunk_document(
                    benchmark_id=benchmark_id,
                    track=track,
                    document=document,
                    pages=pages,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                if not chunks:
                    documents_with_zero_chunks += 1
                zero_length_chunks += sum(1 for row in chunks if not row["text"])
                total_chunk_chars += sum(int(row["chunk_char_len"]) for row in chunks)
                chunk_rows.extend(chunks)
                document_rows.append(
                    {
                        "source_benchmark": track,
                        "source_document_id": doc_id,
                        "source_sha256": document.get("source_sha256"),
                        "page_count": int(document.get("page_count") or len(pages)),
                        "empty_pages": int(document.get("empty_pages") or 0),
                        "chunk_count": len(chunks),
                        "status": "chunked",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                all_errors.append(
                    {
                        "source_benchmark": track,
                        "source_document_id": doc_id,
                        "error_type": exc.__class__.__name__,
                        "reason": str(exc),
                    }
                )
                document_rows.append(
                    {
                        "source_benchmark": track,
                        "source_document_id": doc_id,
                        "source_sha256": document.get("source_sha256"),
                        "page_count": int(document.get("page_count") or len(pages)),
                        "empty_pages": int(document.get("empty_pages") or 0),
                        "chunk_count": 0,
                        "status": "failed",
                    }
                )

        _jsonl_write(track_out / "chunks.jsonl", chunk_rows)
        _jsonl_write(track_out / "documents.jsonl", document_rows)

        evaluation_src = parsed_track / "evaluation"
        if evaluation_src.exists():
            shutil.copytree(evaluation_src, track_out / "evaluation")

        failures = sum(1 for row in document_rows if row["status"] == "failed")
        avg_chunk_chars = round(total_chunk_chars / len(chunk_rows), 2) if chunk_rows else 0.0
        track_stats[track] = {
            "documents_expected": len(documents),
            "documents_chunked": len(documents) - failures,
            "failures": failures,
            "pages": sum(int(row.get("page_count") or 0) for row in documents),
            "empty_input_pages": empty_input_pages,
            "documents_with_zero_chunks": documents_with_zero_chunks,
            "chunks": len(chunk_rows),
            "zero_length_chunks": zero_length_chunks,
            "chunk_characters": total_chunk_chars,
            "average_chunk_characters": avg_chunk_chars,
        }

    _jsonl_write(output_dir / "errors.jsonl", all_errors)

    files: list[dict[str, Any]] = []
    chunk_bytes = 0
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        size = path.stat().st_size
        chunk_bytes += size
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": size,
            }
        )

    chunker_fingerprint = _chunker_fingerprint()
    package_versions = _package_versions()
    python_version = platform.python_version()
    artifact_fingerprint = _artifact_fingerprint(
        parent_parsed_fingerprint,
        chunker_fingerprint,
        package_versions,
        python_version,
        chunk_size,
        chunk_overlap,
    )

    total_chunks = sum(int(stats["chunks"]) for stats in track_stats.values())
    total_zero_chunks = sum(
        int(stats["documents_with_zero_chunks"]) for stats in track_stats.values()
    )
    total_zero_length = sum(
        int(stats["zero_length_chunks"]) for stats in track_stats.values()
    )
    total_empty_pages = sum(
        int(stats["empty_input_pages"]) for stats in track_stats.values()
    )

    manifest = {
        "schema_version": CHUNK_ARTIFACT_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "artifact_type": "chunks",
        "artifact_fingerprint": artifact_fingerprint,
        "parent_parsed_artifact_fingerprint": parent_parsed_fingerprint,
        "parent_parsed_manifest_sha256": _sha256_file(parent_manifest_path),
        "chunker_fingerprint": chunker_fingerprint,
        "chunker_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "repository_revision": repo_revision,
        "python_version": python_version,
        "package_versions": package_versions,
        "stats": track_stats,
        "quality": {
            "empty_input_pages": total_empty_pages,
            "documents_with_zero_chunks": total_zero_chunks,
            "zero_length_chunks": total_zero_length,
            "total_chunks": total_chunks,
        },
        "total_failures": len(all_errors),
        "chunk_bytes": chunk_bytes,
        "files": files,
        "persistence": {
            "backend": "huggingface-storage-bucket",
            "private": True,
            "persisted": False,
            "artifact_uri": None,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if len(all_errors) > max_failures:
        raise RuntimeError(
            f"chunk failures {len(all_errors)} exceed allowed maximum {max_failures}"
        )
    if total_zero_length:
        raise RuntimeError(f"generated {total_zero_length} zero-length chunks")

    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(".benchmark/chunks"))
    parser.add_argument("--benchmark-id", default="composite-v1")
    parser.add_argument("--parent-parsed-fingerprint", required=True)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--max-failures", type=int, default=0)
    args = parser.parse_args()

    try:
        output = chunk_parsed(
            args.parsed_root,
            args.output_root,
            benchmark_id=args.benchmark_id,
            parent_parsed_fingerprint=args.parent_parsed_fingerprint,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_failures=args.max_failures,
            repo_revision=os.environ.get("GITHUB_SHA", "local"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Benchmark chunking failed: {exc}", file=sys.stderr)
        return 1

    print(f"Chunked benchmark corpus: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
