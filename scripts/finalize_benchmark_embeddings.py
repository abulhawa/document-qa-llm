"""Finalize a multi-shard benchmark embedding artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def finalize(
    markers_dir: Path,
    output_path: Path,
    *,
    benchmark_id: str,
    parent_chunks_fingerprint: str,
    artifact_uri: str,
    expected_shards: int,
) -> dict[str, Any]:
    paths = sorted(markers_dir.glob("shard-*.json"))
    if len(paths) != expected_shards:
        raise ValueError(f"expected {expected_shards} shard markers, found {len(paths)}")
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    indexes = sorted(int(row["shard_index"]) for row in shards)
    if indexes != list(range(expected_shards)):
        raise ValueError(f"incomplete shard indexes: {indexes}")

    first = shards[0]
    fingerprint = first["artifact_fingerprint"]
    dimension = int(first["vector_dimension"])
    for shard in shards:
        if shard["artifact_fingerprint"] != fingerprint:
            raise ValueError("shards disagree on artifact fingerprint")
        if shard["parent_chunks_artifact_fingerprint"] != parent_chunks_fingerprint:
            raise ValueError("shard parent fingerprint mismatch")
        if shard["benchmark_id"] != benchmark_id:
            raise ValueError("shard benchmark_id mismatch")
        if int(shard["vector_dimension"]) != dimension:
            raise ValueError("shards disagree on vector dimension")
        for key in ("model", "package_versions", "python_version", "embedder_fingerprint"):
            if shard[key] != first[key]:
                raise ValueError(f"shards disagree on {key}")

    files = {}
    for shard in shards:
        for row in shard.get("files", []):
            files[row["path"]] = row

    manifest = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "artifact_type": "embeddings",
        "artifact_fingerprint": fingerprint,
        "parent_chunks_artifact_fingerprint": parent_chunks_fingerprint,
        "embedder_fingerprint": first["embedder_fingerprint"],
        "python_version": first["python_version"],
        "package_versions": first["package_versions"],
        "model": first["model"],
        "vector_dimension": dimension,
        "shard_count": expected_shards,
        "chunk_embeddings": sum(int(row["chunk_embeddings"]) for row in shards),
        "query_embeddings": sum(int(row["query_embeddings"]) for row in shards),
        "files": [files[key] for key in sorted(files)],
        "persistence": {
            "backend": "huggingface-storage-bucket",
            "private": True,
            "persisted": True,
            "artifact_uri": artifact_uri,
            "reused_existing": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markers-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--parent-chunks-fingerprint", required=True)
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    args = parser.parse_args()
    try:
        finalize(
            args.markers_dir,
            args.output,
            benchmark_id=args.benchmark_id,
            parent_chunks_fingerprint=args.parent_chunks_fingerprint,
            artifact_uri=args.artifact_uri,
            expected_shards=args.expected_shards,
        )
    except Exception as exc:
        print(f"Embedding finalization failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
