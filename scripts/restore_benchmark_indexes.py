"""Restore and validate Benchmark 5 native index snapshots for Benchmark 6."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any, Mapping


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_engine_manifest(
    manifest: Mapping[str, Any],
    *,
    engine: str,
    engine_fingerprint: str,
    benchmark_id: str,
) -> None:
    if manifest.get("artifact_type") != "native-index-engine":
        raise ValueError(f"{engine} artifact is not a native index engine")
    if manifest.get("engine") != engine:
        raise ValueError(f"engine manifest mismatch: expected {engine}")
    if manifest.get("engine_fingerprint") != engine_fingerprint:
        raise ValueError(f"{engine} fingerprint mismatch")
    if manifest.get("benchmark_id") != benchmark_id:
        raise ValueError(f"{engine} benchmark_id mismatch")
    if not manifest.get("persistence", {}).get("persisted"):
        raise ValueError(f"{engine} artifact is not marked persisted")


def _validate_snapshot(path: Path, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    snapshot = manifest.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("engine manifest has no snapshot metadata")
    expected = str(snapshot.get("sha256") or "")
    actual = _sha256_file(path)
    if not expected or actual != expected:
        raise ValueError(f"snapshot checksum mismatch: expected={expected} actual={actual}")
    return snapshot


def _wait_http(url: str, *, timeout_seconds: int = 180) -> None:
    import requests

    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"service did not become ready: {url}; last_error={last_error}")


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    if destination.exists():
        for child in destination.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        destination.mkdir(parents=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination, filter="data")


def restore_opensearch(args: argparse.Namespace) -> None:
    from opensearchpy import OpenSearch

    manifest = _json(args.manifest)
    _validate_engine_manifest(
        manifest,
        engine="opensearch",
        engine_fingerprint=args.engine_fingerprint,
        benchmark_id=args.benchmark_id,
    )
    snapshot = _validate_snapshot(args.snapshot, manifest)
    backend = manifest.get("backend", {})
    if backend.get("name") != "opensearch":
        raise ValueError("invalid OpenSearch backend metadata")
    if str(backend.get("version")) != args.backend_version:
        raise ValueError("OpenSearch backend version mismatch")
    index_name = str(backend.get("index_name") or "")
    if not index_name:
        raise ValueError("OpenSearch manifest is missing index_name")

    _safe_extract_tar(args.snapshot, args.repo_dir)
    _wait_http(f"{args.url.rstrip('/')}/_cluster/health")
    client = OpenSearch(hosts=[args.url], timeout=120)
    client.snapshot.create_repository(
        repository=args.repository_name,
        body={
            "type": "fs",
            "settings": {"location": args.container_repo_dir, "compress": True},
        },
    )
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    response = client.snapshot.restore(
        repository=args.repository_name,
        snapshot=str(snapshot["name"]),
        body={"indices": index_name, "include_global_state": False},
        params={"wait_for_completion": "true"},
    )
    if response.get("snapshot", {}).get("shards", {}).get("failed", 0):
        raise RuntimeError(f"OpenSearch restore failed: {response}")
    client.indices.refresh(index=index_name)
    actual_count = int(client.count(index=index_name)["count"])
    expected_count = int(manifest.get("document_count") or 0)
    if actual_count != expected_count:
        raise RuntimeError(
            f"OpenSearch restored document count mismatch: {actual_count} != {expected_count}"
        )
    print(json.dumps({"engine": "opensearch", "index": index_name, "documents": actual_count}))


def restore_qdrant(args: argparse.Namespace) -> None:
    import requests
    from qdrant_client import QdrantClient

    manifest = _json(args.manifest)
    _validate_engine_manifest(
        manifest,
        engine="qdrant",
        engine_fingerprint=args.engine_fingerprint,
        benchmark_id=args.benchmark_id,
    )
    snapshot = _validate_snapshot(args.snapshot, manifest)
    backend = manifest.get("backend", {})
    if backend.get("name") != "qdrant":
        raise ValueError("invalid Qdrant backend metadata")
    if str(backend.get("version")) != args.backend_version:
        raise ValueError("Qdrant backend version mismatch")
    collection = str(backend.get("collection_name") or "")
    if not collection:
        raise ValueError("Qdrant manifest is missing collection_name")

    base = args.url.rstrip("/")
    _wait_http(f"{base}/readyz")
    params: dict[str, str] = {"priority": "snapshot", "wait": "true"}
    native_checksum = snapshot.get("native_checksum")
    if native_checksum:
        params["checksum"] = str(native_checksum)
    with args.snapshot.open("rb") as fh:
        response = requests.post(
            f"{base}/collections/{collection}/snapshots/upload",
            params=params,
            files={"snapshot": (args.snapshot.name, fh, "application/octet-stream")},
            timeout=600,
        )
    response.raise_for_status()
    if response.json().get("result") is not True:
        raise RuntimeError(f"Qdrant snapshot restore failed: {response.text}")

    client = QdrantClient(url=args.url, timeout=120)
    actual_count = int(client.count(collection_name=collection, exact=True).count)
    expected_count = int(manifest.get("point_count") or 0)
    if actual_count != expected_count:
        raise RuntimeError(
            f"Qdrant restored point count mismatch: {actual_count} != {expected_count}"
        )
    print(json.dumps({"engine": "qdrant", "collection": collection, "points": actual_count}))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", type=Path, required=True)
    common.add_argument("--snapshot", type=Path, required=True)
    common.add_argument("--benchmark-id", required=True)
    common.add_argument("--engine-fingerprint", required=True)
    common.add_argument("--backend-version", required=True)
    common.add_argument("--url", required=True)

    opensearch = subparsers.add_parser("opensearch", parents=[common])
    opensearch.add_argument("--repo-dir", type=Path, required=True)
    opensearch.add_argument("--repository-name", default="benchmark-repo")
    opensearch.add_argument("--container-repo-dir", default="/mnt/opensearch-backups")

    subparsers.add_parser("qdrant", parents=[common])
    args = parser.parse_args()

    if args.command == "opensearch":
        restore_opensearch(args)
    else:
        restore_qdrant(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
