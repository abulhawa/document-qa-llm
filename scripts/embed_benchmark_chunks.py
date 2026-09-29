"""Build one deterministic shard of benchmark embeddings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

TRACKS = ("open_ragbench", "officeqa", "nfcorpus", "miracl_de", "miracl_ar")
SCHEMA_VERSION = 1
EMBEDDER_INPUTS = (
    Path("scripts/embed_benchmark_chunks.py"),
    Path("embedder_api_multilingual/input_format.py"),
    Path("requirements/embed.txt"),
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


def _embedder_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in EMBEDDER_INPUTS:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in (
        "huggingface-hub",
        "numpy",
        "sentence-transformers",
        "torch",
        "transformers",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def _artifact_fingerprint(
    parent_chunks_fingerprint: str,
    embedder_fingerprint: str,
    package_versions: Mapping[str, str],
    python_version: str,
    model_name: str,
    model_revision: str,
    input_format: str,
    normalize_embeddings: bool,
    output_dtype: str,
    execution_device: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "parent_chunks_artifact_fingerprint": parent_chunks_fingerprint,
        "embedder_fingerprint": embedder_fingerprint,
        "package_versions": dict(sorted(package_versions.items())),
        "python_version": python_version,
        "model_name": model_name,
        "model_revision": model_revision,
        "input_format": input_format,
        "document_input_type": "passage",
        "query_input_type": "query",
        "normalize_embeddings": normalize_embeddings,
        "output_dtype": output_dtype,
        "execution_device": execution_device,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_parent(
    chunks_root: Path,
    *,
    benchmark_id: str,
    parent_chunks_fingerprint: str,
) -> None:
    manifest_path = chunks_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing chunks manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "chunks":
        raise ValueError("parent artifact is not a chunks artifact")
    if manifest.get("benchmark_id") != benchmark_id:
        raise ValueError("chunks benchmark_id mismatch")
    if manifest.get("artifact_fingerprint") != parent_chunks_fingerprint:
        raise ValueError("chunks manifest fingerprint mismatch")
    if not manifest.get("persistence", {}).get("persisted"):
        raise ValueError("parent chunks artifact is not marked persisted")


def _selected_chunks(
    chunks_root: Path,
    *,
    shard_index: int,
    shard_count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    global_index = 0
    for track in TRACKS:
        path = chunks_root / track / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing chunk file: {path}")
        for row in _jsonl_rows(path):
            if global_index % shard_count == shard_index:
                item = dict(row)
                item["_global_index"] = global_index
                selected.append(item)
            global_index += 1
    return selected


def _identity(
    *,
    parent_chunks_fingerprint: str,
    model_name: str,
    model_revision: str,
    input_format: str,
    execution_device: str,
) -> tuple[str, dict[str, str], str, str]:
    package_versions = _package_versions()
    python_version = platform.python_version()
    embedder_fingerprint = _embedder_fingerprint()
    artifact_fingerprint = _artifact_fingerprint(
        parent_chunks_fingerprint,
        embedder_fingerprint,
        package_versions,
        python_version,
        model_name,
        model_revision,
        input_format,
        True,
        "float32",
        execution_device,
    )
    return artifact_fingerprint, package_versions, python_version, embedder_fingerprint


def _encode(
    texts: list[str],
    *,
    model: Any,
    input_type: str,
    input_format: str,
    batch_size: int,
):
    import numpy as np
    from embedder_api_multilingual.input_format import prepare_embedding_texts

    prepared = prepare_embedding_texts(
        texts,
        input_type=input_type,  # type: ignore[arg-type]
        input_format=input_format,
    )
    encoded = model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(encoded, dtype=np.float32)


def build_shard(
    chunks_root: Path,
    output_root: Path,
    *,
    benchmark_id: str,
    parent_chunks_fingerprint: str,
    model_name: str,
    model_revision: str,
    input_format: str,
    execution_device: str,
    batch_size: int,
    shard_index: int,
    shard_count: int,
    fingerprint_only: bool,
    repo_revision: str,
) -> Path:
    import numpy as np

    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard layout")
    _validate_parent(
        chunks_root,
        benchmark_id=benchmark_id,
        parent_chunks_fingerprint=parent_chunks_fingerprint,
    )
    fingerprint, packages, python_version, code_fingerprint = _identity(
        parent_chunks_fingerprint=parent_chunks_fingerprint,
        model_name=model_name,
        model_revision=model_revision,
        input_format=input_format,
        execution_device=execution_device,
    )
    output_dir = output_root / benchmark_id
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "build.json").write_text(
        json.dumps(
            {
                "artifact_fingerprint": fingerprint,
                "package_versions": packages,
                "python_version": python_version,
                "embedder_fingerprint": code_fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if fingerprint_only:
        return output_dir

    from sentence_transformers import SentenceTransformer

    rows = _selected_chunks(
        chunks_root,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    texts = [str(row.get("text") or "") for row in rows]
    if any(not text for text in texts):
        raise RuntimeError("chunk artifact contains an empty text row")

    model = SentenceTransformer(
        model_name,
        revision=model_revision,
        device=execution_device,
    )
    vectors = _encode(
        texts,
        model=model,
        input_type="passage",
        input_format=input_format,
        batch_size=batch_size,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(rows):
        raise RuntimeError("embedding row count mismatch")
    if not np.isfinite(vectors).all():
        raise RuntimeError("embedding matrix contains non-finite values")

    shard_dir = output_dir / "shards" / f"{shard_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.save(shard_dir / "embeddings.npy", vectors, allow_pickle=False)
    _jsonl_write(
        shard_dir / "records.jsonl",
        (
            {
                "row_index": i,
                "global_index": int(row["_global_index"]),
                "id": row["id"],
                "source_benchmark": row["source_benchmark"],
                "source_document_id": row["source_document_id"],
                "chunk_index": row["chunk_index"],
            }
            for i, row in enumerate(rows)
        ),
    )

    query_count = 0
    query_files: list[dict[str, Any]] = []
    if shard_index == 0:
        for track in TRACKS:
            query_path = chunks_root / track / "evaluation" / "queries.jsonl"
            if not query_path.exists():
                continue
            queries = list(_jsonl_rows(query_path))
            query_vectors = _encode(
                [str(row["text"]) for row in queries],
                model=model,
                input_type="query",
                input_format=input_format,
                batch_size=batch_size,
            )
            query_dir = output_dir / "queries" / track
            query_dir.mkdir(parents=True, exist_ok=True)
            np.save(query_dir / "embeddings.npy", query_vectors, allow_pickle=False)
            _jsonl_write(
                query_dir / "records.jsonl",
                (
                    {"row_index": i, "query_id": row["query_id"]}
                    for i, row in enumerate(queries)
                ),
            )
            query_count += len(queries)
            for path in (query_dir / "embeddings.npy", query_dir / "records.jsonl"):
                query_files.append(
                    {
                        "path": path.relative_to(output_dir).as_posix(),
                        "sha256": _sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )

    files: list[dict[str, Any]] = []
    for path in (shard_dir / "embeddings.npy", shard_dir / "records.jsonl"):
        files.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    files.extend(query_files)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "artifact_type": "embedding-shard",
        "artifact_fingerprint": fingerprint,
        "parent_chunks_artifact_fingerprint": parent_chunks_fingerprint,
        "repository_revision": repo_revision,
        "embedder_fingerprint": code_fingerprint,
        "python_version": python_version,
        "package_versions": packages,
        "model": {
            "name": model_name,
            "revision": model_revision,
            "input_format": input_format,
            "document_input_type": "passage",
            "query_input_type": "query",
            "normalize_embeddings": True,
            "execution_device": execution_device,
            "output_dtype": "float32",
        },
        "shard_index": shard_index,
        "shard_count": shard_count,
        "chunk_embeddings": len(rows),
        "query_embeddings": query_count,
        "vector_dimension": int(vectors.shape[1]),
        "files": files,
    }
    (shard_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(".benchmark/embeddings"))
    parser.add_argument("--benchmark-id", default="composite-v1")
    parser.add_argument("--parent-chunks-fingerprint", required=True)
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--input-format", choices=("e5", "raw"), default="e5")
    parser.add_argument("--execution-device", choices=("cpu",), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--fingerprint-only", action="store_true")
    args = parser.parse_args()

    try:
        output = build_shard(
            args.chunks_root,
            args.output_root,
            benchmark_id=args.benchmark_id,
            parent_chunks_fingerprint=args.parent_chunks_fingerprint,
            model_name=args.model_name,
            model_revision=args.model_revision,
            input_format=args.input_format,
            execution_device=args.execution_device,
            batch_size=args.batch_size,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            fingerprint_only=args.fingerprint_only,
            repo_revision=os.environ.get("GITHUB_SHA", "local"),
        )
    except Exception as exc:
        print(f"Benchmark embedding failed: {exc}", file=sys.stderr)
        return 1
    print(f"Embedding shard output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
