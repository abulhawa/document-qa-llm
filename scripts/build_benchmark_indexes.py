"""Build, checkpoint, snapshot, and finalize Benchmark 5 native indexes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

TRACKS = ("open_ragbench", "officeqa", "nfcorpus", "miracl_de", "miracl_ar")
INDEX_ARTIFACT_SCHEMA_VERSION = 1
OPENSEARCH_CONTRACT_VERSION = 2
QDRANT_CONTRACT_VERSION = 2
CHECKPOINT_SCHEMA_VERSION = 1
OPENSEARCH_INDEX_NAME = "benchmark-documents"
OPENSEARCH_REPOSITORY = "benchmark-repo"
OPENSEARCH_FINAL_SNAPSHOT = "benchmark-final"
OPENSEARCH_CHECKPOINT_SNAPSHOT = "benchmark-checkpoint"
QDRANT_COLLECTION_NAME = "benchmark-document-chunks"
QDRANT_PAYLOAD_KEYS = ("checksum", "id", "path")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield dict(json.loads(line))


def _validate_chunks(
    chunks_root: Path,
    *,
    benchmark_id: str,
    chunks_fingerprint: str,
) -> dict[str, Any]:
    manifest = _json(chunks_root / "manifest.json")
    if manifest.get("artifact_type") != "chunks":
        raise ValueError("parent artifact is not chunks")
    if manifest.get("benchmark_id") != benchmark_id:
        raise ValueError("chunks benchmark_id mismatch")
    if manifest.get("artifact_fingerprint") != chunks_fingerprint:
        raise ValueError("chunks fingerprint mismatch")
    if not manifest.get("persistence", {}).get("persisted"):
        raise ValueError("chunks artifact is not marked persisted")
    return manifest


def _validate_embeddings(
    embeddings_root: Path,
    *,
    benchmark_id: str,
    chunks_fingerprint: str,
    embeddings_fingerprint: str,
) -> dict[str, Any]:
    manifest = _json(embeddings_root / "manifest.json")
    if manifest.get("artifact_type") != "embeddings":
        raise ValueError("parent artifact is not embeddings")
    if manifest.get("benchmark_id") != benchmark_id:
        raise ValueError("embeddings benchmark_id mismatch")
    if manifest.get("artifact_fingerprint") != embeddings_fingerprint:
        raise ValueError("embeddings fingerprint mismatch")
    if manifest.get("parent_chunks_artifact_fingerprint") != chunks_fingerprint:
        raise ValueError("embeddings parent chunks fingerprint mismatch")
    if not manifest.get("persistence", {}).get("persisted"):
        raise ValueError("embeddings artifact is not marked persisted")
    return manifest


def _production_opensearch_settings() -> dict[str, Any]:
    from utils.opensearch_utils import CHUNKS_INDEX_SETTINGS

    settings = copy.deepcopy(CHUNKS_INDEX_SETTINGS)
    index_settings = settings.setdefault("settings", {}).setdefault("index", {})
    index_settings["number_of_replicas"] = 0
    index_settings["refresh_interval"] = "1s"
    return settings


def opensearch_engine_fingerprint(
    chunks_fingerprint: str,
    opensearch_version: str,
    *,
    settings: Mapping[str, Any] | None = None,
) -> str:
    effective = dict(settings or _production_opensearch_settings())
    return _canonical_sha256(
        {
            "artifact_schema_version": INDEX_ARTIFACT_SCHEMA_VERSION,
            "engine": "opensearch",
            "contract_version": OPENSEARCH_CONTRACT_VERSION,
            "parent_chunks_artifact_fingerprint": chunks_fingerprint,
            "opensearch_version": opensearch_version,
            "index_name": OPENSEARCH_INDEX_NAME,
            "document_id_strategy": "uuid5(chunk_id,path)",
            "settings": effective,
        }
    )


def qdrant_engine_fingerprint(
    chunks_fingerprint: str,
    embeddings_fingerprint: str,
    qdrant_version: str,
    vector_dimension: int,
) -> str:
    return _canonical_sha256(
        {
            "artifact_schema_version": INDEX_ARTIFACT_SCHEMA_VERSION,
            "engine": "qdrant",
            "contract_version": QDRANT_CONTRACT_VERSION,
            "parent_chunks_artifact_fingerprint": chunks_fingerprint,
            "parent_embeddings_artifact_fingerprint": embeddings_fingerprint,
            "qdrant_version": qdrant_version,
            "collection_name": QDRANT_COLLECTION_NAME,
            "point_id_strategy": "uuid5(chunk_id,path)",
            "vectors": {"size": int(vector_dimension), "distance": "cosine"},
            "payload_keys": list(QDRANT_PAYLOAD_KEYS),
        }
    )


def checkpoint_signature(engine_fingerprint: str, builder_fingerprint: str) -> str:
    return _canonical_sha256(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "engine_fingerprint": engine_fingerprint,
            "builder_fingerprint": builder_fingerprint,
        }
    )


def combined_index_fingerprint(
    *,
    benchmark_id: str,
    chunks_fingerprint: str,
    embeddings_fingerprint: str,
    opensearch_fingerprint: str,
    qdrant_fingerprint: str,
) -> str:
    return _canonical_sha256(
        {
            "artifact_schema_version": INDEX_ARTIFACT_SCHEMA_VERSION,
            "benchmark_id": benchmark_id,
            "artifact_type": "native-indexes",
            "parent_chunks_artifact_fingerprint": chunks_fingerprint,
            "parent_embeddings_artifact_fingerprint": embeddings_fingerprint,
            "opensearch_engine_fingerprint": opensearch_fingerprint,
            "qdrant_engine_fingerprint": qdrant_fingerprint,
        }
    )


def _builder_fingerprint() -> str:
    return _sha256_file(Path(__file__))


def _backend_chunk_id(row: Mapping[str, Any]) -> str:
    """Return a benchmark-scoped backend key without changing the chunk artifact.

    Chunk artifact IDs are content-addressed and can legitimately repeat when
    identical source content appears under more than one benchmark document path.
    OpenSearch document IDs and Qdrant point IDs must still be unique per benchmark
    row so retrieval preserves track/document provenance.
    """

    chunk_id = str(row.get("id") or "")
    path = str(row.get("path") or "")
    if not chunk_id or not path:
        raise ValueError("chunk is missing id/path required for backend identity")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk_id}|{path}"))


def _chunk_source(row: Mapping[str, Any]) -> dict[str, Any]:
    checksum = str(row.get("source_sha256") or row.get("source_document_id") or "")
    if not checksum:
        raise ValueError("chunk is missing a stable document identity")
    source: dict[str, Any] = {
        "text": str(row.get("text") or ""),
        "path": str(row.get("path") or ""),
        "chunk_index": int(row.get("chunk_index", 0)),
        "checksum": checksum,
        "chunk_char_len": int(row.get("chunk_char_len", len(str(row.get("text") or "")))),
        "filetype": row.get("filetype"),
        "page": row.get("page"),
        "location_percent": row.get("location_percent"),
    }
    return {key: value for key, value in source.items() if value is not None}


def _qdrant_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    source = _chunk_source(row)
    return {
        "id": _backend_chunk_id(row),
        "checksum": source["checksum"],
        "path": source["path"],
    }


def _iter_chunks(chunks_root: Path, *, start: int = 0) -> Iterator[tuple[int, dict[str, Any]]]:
    global_index = 0
    for track in TRACKS:
        path = chunks_root / track / "chunks.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing chunk file: {path}")
        for row in _jsonl_rows(path):
            if global_index >= start:
                yield global_index, row
            global_index += 1


def _load_embedding_shards(embeddings_root: Path, manifest: Mapping[str, Any]):
    import numpy as np

    shard_count = int(manifest["shard_count"])
    vectors = []
    records = []
    for shard_index in range(shard_count):
        shard_dir = embeddings_root / "shards" / f"{shard_index:03d}"
        shard_vectors = np.load(shard_dir / "embeddings.npy", allow_pickle=False, mmap_mode="r")
        shard_records = list(_jsonl_rows(shard_dir / "records.jsonl"))
        if shard_vectors.ndim != 2 or shard_vectors.shape[0] != len(shard_records):
            raise ValueError(f"embedding shard {shard_index} row mismatch")
        vectors.append(shard_vectors)
        records.append(shard_records)
    return shard_count, vectors, records


def _iter_qdrant_rows(
    chunks_root: Path,
    embeddings_root: Path,
    embeddings_manifest: Mapping[str, Any],
    *,
    start: int = 0,
):
    shard_count, vectors, records = _load_embedding_shards(embeddings_root, embeddings_manifest)
    for global_index, row in _iter_chunks(chunks_root, start=start):
        shard_index = global_index % shard_count
        row_index = global_index // shard_count
        record = records[shard_index][row_index]
        if int(record["global_index"]) != global_index or str(record["id"]) != str(row["id"]):
            raise ValueError(f"embedding alignment mismatch at global row {global_index}")
        yield global_index, row, vectors[shard_index][row_index]


def _hf(args: list[str], *, quiet: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required")
    kwargs: dict[str, Any] = {"env": os.environ.copy(), "text": True, "check": check}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.run(["hf", "buckets", *args], **kwargs)


def _download_optional(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    result = _hf(["cp", remote, str(local)], quiet=True, check=False)
    return result.returncode == 0 and local.exists()


def _upload(local: Path, remote: str) -> None:
    _hf(["cp", str(local), remote])


def _checkpoint_pointer_uri(remote_root: str) -> str:
    return f"{remote_root.rstrip('/')}/work/current.json"


def _restore_checkpoint_metadata(
    *,
    remote_root: str,
    work_dir: Path,
    engine: str,
    engine_fingerprint: str,
    expected_signature: str,
) -> tuple[dict[str, Any], Path] | None:
    pointer_path = work_dir / "checkpoint-current.json"
    if not _download_optional(_checkpoint_pointer_uri(remote_root), pointer_path):
        return None
    pointer = _json(pointer_path)
    if pointer.get("engine") != engine or pointer.get("engine_fingerprint") != engine_fingerprint:
        return None
    if pointer.get("checkpoint_signature") != expected_signature:
        return None
    checkpoint_uri = str(pointer.get("artifact_uri") or "")
    if not checkpoint_uri:
        return None
    manifest_path = work_dir / "checkpoint-manifest.json"
    if not _download_optional(f"{checkpoint_uri}/manifest.json", manifest_path):
        return None
    manifest = _json(manifest_path)
    if manifest.get("engine_fingerprint") != engine_fingerprint:
        return None
    if manifest.get("checkpoint_signature") != expected_signature:
        return None
    return manifest, manifest_path


def _persist_checkpoint(
    *,
    remote_root: str,
    work_dir: Path,
    manifest: Mapping[str, Any],
    snapshot_path: Path,
) -> None:
    rows = int(manifest["rows_indexed"])
    checkpoint_uri = f"{remote_root.rstrip('/')}/work/checkpoints/{rows:09d}"
    remote_snapshot_name = str(manifest["snapshot"]["filename"])
    _upload(snapshot_path, f"{checkpoint_uri}/{remote_snapshot_name}")
    manifest_path = work_dir / f"checkpoint-{rows:09d}.json"
    _write_json(manifest_path, manifest)
    _upload(manifest_path, f"{checkpoint_uri}/manifest.json")
    pointer = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "engine": manifest["engine"],
        "engine_fingerprint": manifest["engine_fingerprint"],
        "checkpoint_signature": manifest["checkpoint_signature"],
        "rows_indexed": rows,
        "artifact_uri": checkpoint_uri,
    }
    pointer_path = work_dir / "checkpoint-current-upload.json"
    _write_json(pointer_path, pointer)
    _upload(pointer_path, _checkpoint_pointer_uri(remote_root))


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


def _tar_directory(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for child in sorted(source.iterdir()):
            archive.add(child, arcname=child.name)


def _extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination, filter="data")


def _engine_manifest_matches(
    manifest: Mapping[str, Any],
    *,
    engine: str,
    engine_fingerprint: str,
) -> bool:
    return (
        manifest.get("artifact_type") == "native-index-engine"
        and manifest.get("engine") == engine
        and manifest.get("engine_fingerprint") == engine_fingerprint
        and manifest.get("persistence", {}).get("persisted") is True
    )


def _opensearch_identity(args: argparse.Namespace) -> dict[str, Any]:
    _validate_chunks(
        args.chunks_root,
        benchmark_id=args.benchmark_id,
        chunks_fingerprint=args.chunks_fingerprint,
    )
    settings = _production_opensearch_settings()
    fingerprint = opensearch_engine_fingerprint(
        args.chunks_fingerprint,
        args.backend_version,
        settings=settings,
    )
    return {
        "engine": "opensearch",
        "engine_fingerprint": fingerprint,
        "checkpoint_signature": checkpoint_signature(fingerprint, _builder_fingerprint()),
        "settings_sha256": _canonical_sha256(settings),
    }


def _qdrant_identity(args: argparse.Namespace) -> dict[str, Any]:
    if args.embeddings_root is None or not args.embeddings_fingerprint:
        raise ValueError("Qdrant identity requires embeddings inputs")
    _validate_chunks(
        args.chunks_root,
        benchmark_id=args.benchmark_id,
        chunks_fingerprint=args.chunks_fingerprint,
    )
    embeddings = _validate_embeddings(
        args.embeddings_root,
        benchmark_id=args.benchmark_id,
        chunks_fingerprint=args.chunks_fingerprint,
        embeddings_fingerprint=args.embeddings_fingerprint,
    )
    dimension = int(embeddings["vector_dimension"])
    fingerprint = qdrant_engine_fingerprint(
        args.chunks_fingerprint,
        args.embeddings_fingerprint,
        args.backend_version,
        dimension,
    )
    return {
        "engine": "qdrant",
        "engine_fingerprint": fingerprint,
        "checkpoint_signature": checkpoint_signature(fingerprint, _builder_fingerprint()),
        "vector_dimension": dimension,
    }


def _create_opensearch_snapshot(client: Any, snapshot_name: str) -> None:
    try:
        client.snapshot.delete(repository=OPENSEARCH_REPOSITORY, snapshot=snapshot_name)
    except Exception:  # noqa: BLE001
        pass
    response = client.snapshot.create(
        repository=OPENSEARCH_REPOSITORY,
        snapshot=snapshot_name,
        body={"indices": OPENSEARCH_INDEX_NAME, "include_global_state": False, "partial": False},
        params={"wait_for_completion": "true"},
    )
    snapshot = response.get("snapshot", {})
    if snapshot.get("state") != "SUCCESS":
        raise RuntimeError(f"OpenSearch snapshot failed: {response}")


def _opensearch_checkpoint(
    *,
    client: Any,
    repo_dir: Path,
    remote_root: str,
    work_dir: Path,
    identity: Mapping[str, Any],
    benchmark_id: str,
    chunks_fingerprint: str,
    backend_version: str,
    rows_indexed: int,
    total_rows: int,
) -> None:
    client.indices.flush(index=OPENSEARCH_INDEX_NAME)
    _create_opensearch_snapshot(client, OPENSEARCH_CHECKPOINT_SNAPSHOT)
    snapshot_path = work_dir / f"opensearch-{rows_indexed:09d}.tar.gz"
    _tar_directory(repo_dir, snapshot_path)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": "native-index-checkpoint",
        "engine": "opensearch",
        "engine_fingerprint": identity["engine_fingerprint"],
        "checkpoint_signature": identity["checkpoint_signature"],
        "benchmark_id": benchmark_id,
        "parent_chunks_artifact_fingerprint": chunks_fingerprint,
        "backend_version": backend_version,
        "rows_indexed": rows_indexed,
        "total_rows": total_rows,
        "snapshot": {
            "format": "opensearch-fs-repository-tar",
            "name": OPENSEARCH_CHECKPOINT_SNAPSHOT,
            "filename": snapshot_path.name,
            "sha256": _sha256_file(snapshot_path),
            "bytes": snapshot_path.stat().st_size,
        },
    }
    _persist_checkpoint(remote_root=remote_root, work_dir=work_dir, manifest=manifest, snapshot_path=snapshot_path)
    client.snapshot.delete(repository=OPENSEARCH_REPOSITORY, snapshot=OPENSEARCH_CHECKPOINT_SNAPSHOT)


def build_opensearch(args: argparse.Namespace) -> dict[str, Any]:
    from opensearchpy import OpenSearch, helpers

    chunks_manifest = _validate_chunks(
        args.chunks_root,
        benchmark_id=args.benchmark_id,
        chunks_fingerprint=args.chunks_fingerprint,
    )
    identity = _opensearch_identity(args)
    total_rows = int(chunks_manifest.get("total_chunks") or 0)
    if total_rows <= 0:
        total_rows = sum(1 for _ in _iter_chunks(args.chunks_root))

    final_manifest_path = args.work_dir / "opensearch-final-manifest.json"
    remote_manifest = args.work_dir / "opensearch-remote-final.json"
    if _download_optional(f"{args.remote_root}/manifest.json", remote_manifest):
        existing = _json(remote_manifest)
        if not _engine_manifest_matches(existing, engine="opensearch", engine_fingerprint=identity["engine_fingerprint"]):
            raise RuntimeError("existing OpenSearch final manifest is incompatible")
        return existing

    _wait_http(f"{args.url}/_cluster/health")
    client = OpenSearch(hosts=[args.url], timeout=120)
    args.repo_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    restored_rows = 0
    checkpoint = _restore_checkpoint_metadata(
        remote_root=args.remote_root,
        work_dir=args.work_dir,
        engine="opensearch",
        engine_fingerprint=identity["engine_fingerprint"],
        expected_signature=identity["checkpoint_signature"],
    )
    if checkpoint:
        checkpoint_manifest, _ = checkpoint
        snapshot_meta = checkpoint_manifest["snapshot"]
        checkpoint_uri = f"{args.remote_root}/work/checkpoints/{int(checkpoint_manifest['rows_indexed']):09d}"
        archive_path = args.work_dir / str(snapshot_meta["filename"])
        if not _download_optional(f"{checkpoint_uri}/{snapshot_meta['filename']}", archive_path):
            raise RuntimeError("OpenSearch checkpoint snapshot is missing")
        if _sha256_file(archive_path) != snapshot_meta["sha256"]:
            raise RuntimeError("OpenSearch checkpoint snapshot checksum mismatch")
        if args.repo_dir.exists():
            for child in args.repo_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        _extract_tar(archive_path, args.repo_dir)

    client.snapshot.create_repository(
        repository=OPENSEARCH_REPOSITORY,
        body={"type": "fs", "settings": {"location": "/mnt/opensearch-backups", "compress": True}},
    )

    if checkpoint:
        checkpoint_manifest, _ = checkpoint
        response = client.snapshot.restore(
            repository=OPENSEARCH_REPOSITORY,
            snapshot=str(checkpoint_manifest["snapshot"]["name"]),
            body={"indices": OPENSEARCH_INDEX_NAME, "include_global_state": False},
            params={"wait_for_completion": "true"},
        )
        if response.get("snapshot", {}).get("shards", {}).get("failed", 0):
            raise RuntimeError(f"OpenSearch checkpoint restore failed: {response}")
        restored_rows = int(checkpoint_manifest["rows_indexed"])
        count = int(client.count(index=OPENSEARCH_INDEX_NAME)["count"])
        if count != restored_rows:
            raise RuntimeError(f"OpenSearch checkpoint row count mismatch: {count} != {restored_rows}")
        client.snapshot.delete(repository=OPENSEARCH_REPOSITORY, snapshot=OPENSEARCH_CHECKPOINT_SNAPSHOT)
    else:
        settings = _production_opensearch_settings()
        settings["settings"]["index"]["refresh_interval"] = "-1"
        client.indices.create(index=OPENSEARCH_INDEX_NAME, body=settings)

    client.indices.put_settings(index=OPENSEARCH_INDEX_NAME, body={"index": {"refresh_interval": "-1"}})

    actions: list[dict[str, Any]] = []
    processed = restored_rows
    next_checkpoint = ((processed // args.checkpoint_rows) + 1) * args.checkpoint_rows
    for global_index, row in _iter_chunks(args.chunks_root, start=restored_rows):
        actions.append(
            {
                "_op_type": "index",
                "_index": OPENSEARCH_INDEX_NAME,
                "_id": _backend_chunk_id(row),
                "_source": _chunk_source(row),
            }
        )
        processed = global_index + 1
        if len(actions) >= args.batch_size:
            success, errors = helpers.bulk(client, actions, raise_on_error=False, request_timeout=120)
            if errors or success != len(actions):
                raise RuntimeError(f"OpenSearch bulk indexing errors: success={success}, errors={errors[:3]}")
            actions.clear()
        if processed >= next_checkpoint and processed < total_rows:
            if actions:
                success, errors = helpers.bulk(client, actions, raise_on_error=False, request_timeout=120)
                if errors or success != len(actions):
                    raise RuntimeError("OpenSearch bulk indexing failed before checkpoint")
                actions.clear()
            _opensearch_checkpoint(
                client=client,
                repo_dir=args.repo_dir,
                remote_root=args.remote_root,
                work_dir=args.work_dir,
                identity=identity,
                benchmark_id=args.benchmark_id,
                chunks_fingerprint=args.chunks_fingerprint,
                backend_version=args.backend_version,
                rows_indexed=processed,
                total_rows=total_rows,
            )
            next_checkpoint = ((processed // args.checkpoint_rows) + 1) * args.checkpoint_rows
    if actions:
        success, errors = helpers.bulk(client, actions, raise_on_error=False, request_timeout=120)
        if errors or success != len(actions):
            raise RuntimeError("OpenSearch final bulk indexing failed")

    if processed != total_rows:
        raise RuntimeError(f"OpenSearch processed row mismatch: {processed} != {total_rows}")
    client.indices.put_settings(index=OPENSEARCH_INDEX_NAME, body={"index": {"refresh_interval": "1s"}})
    client.indices.refresh(index=OPENSEARCH_INDEX_NAME)
    client.indices.flush(index=OPENSEARCH_INDEX_NAME)
    count = int(client.count(index=OPENSEARCH_INDEX_NAME)["count"])
    if count != total_rows:
        raise RuntimeError(f"OpenSearch final count mismatch: {count} != {total_rows}")

    _create_opensearch_snapshot(client, OPENSEARCH_FINAL_SNAPSHOT)
    snapshot_path = args.work_dir / "opensearch-snapshot.tar.gz"
    _tar_directory(args.repo_dir, snapshot_path)
    snapshot_sha = _sha256_file(snapshot_path)
    _upload(snapshot_path, f"{args.remote_root}/snapshot.tar.gz")
    manifest = {
        "schema_version": INDEX_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "native-index-engine",
        "engine": "opensearch",
        "engine_fingerprint": identity["engine_fingerprint"],
        "checkpoint_signature": identity["checkpoint_signature"],
        "benchmark_id": args.benchmark_id,
        "parent_chunks_artifact_fingerprint": args.chunks_fingerprint,
        "backend": {
            "name": "opensearch",
            "version": args.backend_version,
            "index_name": OPENSEARCH_INDEX_NAME,
            "settings_sha256": identity["settings_sha256"],
        },
        "document_count": count,
        "snapshot": {
            "format": "opensearch-fs-repository-tar",
            "name": OPENSEARCH_FINAL_SNAPSHOT,
            "filename": "snapshot.tar.gz",
            "sha256": snapshot_sha,
            "bytes": snapshot_path.stat().st_size,
        },
        "repository_revision": os.environ.get("GITHUB_SHA", "local"),
        "persistence": {
            "backend": "huggingface-storage-bucket",
            "private": True,
            "persisted": True,
            "artifact_uri": args.remote_root,
        },
    }
    _write_json(final_manifest_path, manifest)
    _upload(final_manifest_path, f"{args.remote_root}/manifest.json")
    return manifest


def _qdrant_snapshot(url: str, collection: str, output: Path) -> tuple[str, str | None]:
    import requests

    response = requests.post(f"{url}/collections/{collection}/snapshots", timeout=300)
    response.raise_for_status()
    result = response.json().get("result", {})
    name = str(result["name"])
    checksum = result.get("checksum")
    with requests.get(f"{url}/collections/{collection}/snapshots/{name}", stream=True, timeout=300) as download:
        download.raise_for_status()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as fh:
            for block in download.iter_content(chunk_size=1024 * 1024):
                if block:
                    fh.write(block)
    return name, str(checksum) if checksum else None


def _delete_qdrant_snapshot(url: str, collection: str, name: str) -> None:
    import requests

    requests.delete(f"{url}/collections/{collection}/snapshots/{name}", timeout=60)


def _restore_qdrant_snapshot(url: str, collection: str, snapshot: Path, checksum: str | None) -> None:
    import requests

    params = {"priority": "snapshot", "wait": "true"}
    if checksum:
        params["checksum"] = checksum
    with snapshot.open("rb") as fh:
        response = requests.post(
            f"{url}/collections/{collection}/snapshots/upload",
            params=params,
            files={"snapshot": (snapshot.name, fh, "application/octet-stream")},
            timeout=600,
        )
    response.raise_for_status()
    if response.json().get("result") is not True:
        raise RuntimeError(f"Qdrant snapshot restore failed: {response.text}")


def _wait_qdrant_optimized(url: str, collection: str, *, timeout_seconds: int = 900) -> None:
    import requests

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = requests.get(f"{url}/collections/{collection}", timeout=30)
        response.raise_for_status()
        result = response.json().get("result", {})
        status = str(result.get("status", "")).lower()
        optimizer = result.get("optimizer_status")
        optimizer_ok = optimizer == "ok" or (isinstance(optimizer, dict) and not optimizer.get("error"))
        if status == "green" and optimizer_ok:
            return
        time.sleep(5)
    raise TimeoutError("Qdrant collection did not reach green/optimized state")


def _qdrant_checkpoint(
    *,
    url: str,
    remote_root: str,
    work_dir: Path,
    identity: Mapping[str, Any],
    benchmark_id: str,
    chunks_fingerprint: str,
    embeddings_fingerprint: str,
    backend_version: str,
    rows_indexed: int,
    total_rows: int,
) -> None:
    snapshot_path = work_dir / f"qdrant-{rows_indexed:09d}.snapshot"
    name, native_checksum = _qdrant_snapshot(url, QDRANT_COLLECTION_NAME, snapshot_path)
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_type": "native-index-checkpoint",
        "engine": "qdrant",
        "engine_fingerprint": identity["engine_fingerprint"],
        "checkpoint_signature": identity["checkpoint_signature"],
        "benchmark_id": benchmark_id,
        "parent_chunks_artifact_fingerprint": chunks_fingerprint,
        "parent_embeddings_artifact_fingerprint": embeddings_fingerprint,
        "backend_version": backend_version,
        "rows_indexed": rows_indexed,
        "total_rows": total_rows,
        "snapshot": {
            "format": "qdrant-collection-snapshot",
            "name": name,
            "filename": snapshot_path.name,
            "sha256": _sha256_file(snapshot_path),
            "native_checksum": native_checksum,
            "bytes": snapshot_path.stat().st_size,
        },
    }
    _persist_checkpoint(remote_root=remote_root, work_dir=work_dir, manifest=manifest, snapshot_path=snapshot_path)
    _delete_qdrant_snapshot(url, QDRANT_COLLECTION_NAME, name)


def build_qdrant(args: argparse.Namespace) -> dict[str, Any]:
    from qdrant_client import QdrantClient, models

    chunks_manifest = _validate_chunks(
        args.chunks_root,
        benchmark_id=args.benchmark_id,
        chunks_fingerprint=args.chunks_fingerprint,
    )
    embeddings_manifest = _validate_embeddings(
        args.embeddings_root,
        benchmark_id=args.benchmark_id,
        chunks_fingerprint=args.chunks_fingerprint,
        embeddings_fingerprint=args.embeddings_fingerprint,
    )
    identity = _qdrant_identity(args)
    total_rows = int(embeddings_manifest["chunk_embeddings"])
    if int(chunks_manifest.get("total_chunks") or total_rows) != total_rows:
        raise ValueError("chunk and embedding counts disagree")

    final_manifest_path = args.work_dir / "qdrant-final-manifest.json"
    remote_manifest = args.work_dir / "qdrant-remote-final.json"
    if _download_optional(f"{args.remote_root}/manifest.json", remote_manifest):
        existing = _json(remote_manifest)
        if not _engine_manifest_matches(existing, engine="qdrant", engine_fingerprint=identity["engine_fingerprint"]):
            raise RuntimeError("existing Qdrant final manifest is incompatible")
        return existing

    _wait_http(f"{args.url}/readyz")
    client = QdrantClient(url=args.url, timeout=120)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    restored_rows = 0
    checkpoint = _restore_checkpoint_metadata(
        remote_root=args.remote_root,
        work_dir=args.work_dir,
        engine="qdrant",
        engine_fingerprint=identity["engine_fingerprint"],
        expected_signature=identity["checkpoint_signature"],
    )
    if checkpoint:
        checkpoint_manifest, _ = checkpoint
        rows = int(checkpoint_manifest["rows_indexed"])
        checkpoint_uri = f"{args.remote_root}/work/checkpoints/{rows:09d}"
        snapshot_meta = checkpoint_manifest["snapshot"]
        snapshot_path = args.work_dir / str(snapshot_meta["filename"])
        if not _download_optional(f"{checkpoint_uri}/{snapshot_meta['filename']}", snapshot_path):
            raise RuntimeError("Qdrant checkpoint snapshot is missing")
        if _sha256_file(snapshot_path) != snapshot_meta["sha256"]:
            raise RuntimeError("Qdrant checkpoint snapshot checksum mismatch")
        _restore_qdrant_snapshot(args.url, QDRANT_COLLECTION_NAME, snapshot_path, snapshot_meta.get("native_checksum"))
        restored_rows = rows
        count = int(client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count)
        if count != restored_rows:
            raise RuntimeError(f"Qdrant checkpoint row count mismatch: {count} != {restored_rows}")
    else:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=int(embeddings_manifest["vector_dimension"]),
                distance=models.Distance.COSINE,
            ),
        )

    batch: list[Any] = []
    processed = restored_rows
    next_checkpoint = ((processed // args.checkpoint_rows) + 1) * args.checkpoint_rows
    for global_index, row, vector in _iter_qdrant_rows(
        args.chunks_root,
        args.embeddings_root,
        embeddings_manifest,
        start=restored_rows,
    ):
        batch.append(
            models.PointStruct(
                id=_backend_chunk_id(row),
                vector=vector.tolist(),
                payload=_qdrant_payload(row),
            )
        )
        processed = global_index + 1
        if len(batch) >= args.batch_size:
            client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch, wait=True)
            batch.clear()
        if processed >= next_checkpoint and processed < total_rows:
            if batch:
                client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch, wait=True)
                batch.clear()
            count = int(client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count)
            if count != processed:
                raise RuntimeError(f"Qdrant checkpoint count mismatch: {count} != {processed}")
            _qdrant_checkpoint(
                url=args.url,
                remote_root=args.remote_root,
                work_dir=args.work_dir,
                identity=identity,
                benchmark_id=args.benchmark_id,
                chunks_fingerprint=args.chunks_fingerprint,
                embeddings_fingerprint=args.embeddings_fingerprint,
                backend_version=args.backend_version,
                rows_indexed=processed,
                total_rows=total_rows,
            )
            next_checkpoint = ((processed // args.checkpoint_rows) + 1) * args.checkpoint_rows
    if batch:
        client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=batch, wait=True)

    if processed != total_rows:
        raise RuntimeError(f"Qdrant processed row mismatch: {processed} != {total_rows}")
    count = int(client.count(collection_name=QDRANT_COLLECTION_NAME, exact=True).count)
    if count != total_rows:
        raise RuntimeError(f"Qdrant final count mismatch: {count} != {total_rows}")
    _wait_qdrant_optimized(args.url, QDRANT_COLLECTION_NAME)

    snapshot_path = args.work_dir / "qdrant.snapshot"
    snapshot_name, native_checksum = _qdrant_snapshot(args.url, QDRANT_COLLECTION_NAME, snapshot_path)
    snapshot_sha = _sha256_file(snapshot_path)
    _upload(snapshot_path, f"{args.remote_root}/snapshot.snapshot")
    manifest = {
        "schema_version": INDEX_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "native-index-engine",
        "engine": "qdrant",
        "engine_fingerprint": identity["engine_fingerprint"],
        "checkpoint_signature": identity["checkpoint_signature"],
        "benchmark_id": args.benchmark_id,
        "parent_chunks_artifact_fingerprint": args.chunks_fingerprint,
        "parent_embeddings_artifact_fingerprint": args.embeddings_fingerprint,
        "backend": {
            "name": "qdrant",
            "version": args.backend_version,
            "collection_name": QDRANT_COLLECTION_NAME,
            "vector_dimension": int(embeddings_manifest["vector_dimension"]),
            "distance": "cosine",
            "payload_keys": list(QDRANT_PAYLOAD_KEYS),
        },
        "point_count": count,
        "snapshot": {
            "format": "qdrant-collection-snapshot",
            "name": snapshot_name,
            "filename": "snapshot.snapshot",
            "sha256": snapshot_sha,
            "native_checksum": native_checksum,
            "bytes": snapshot_path.stat().st_size,
        },
        "repository_revision": os.environ.get("GITHUB_SHA", "local"),
        "persistence": {
            "backend": "huggingface-storage-bucket",
            "private": True,
            "persisted": True,
            "artifact_uri": args.remote_root,
        },
    }
    _write_json(final_manifest_path, manifest)
    _upload(final_manifest_path, f"{args.remote_root}/manifest.json")
    _delete_qdrant_snapshot(args.url, QDRANT_COLLECTION_NAME, snapshot_name)
    return manifest


def finalize_indexes(args: argparse.Namespace) -> dict[str, Any]:
    opensearch = _json(args.opensearch_manifest)
    qdrant = _json(args.qdrant_manifest)
    if not _engine_manifest_matches(opensearch, engine="opensearch", engine_fingerprint=args.opensearch_fingerprint):
        raise ValueError("OpenSearch engine manifest mismatch")
    if not _engine_manifest_matches(qdrant, engine="qdrant", engine_fingerprint=args.qdrant_fingerprint):
        raise ValueError("Qdrant engine manifest mismatch")
    for manifest in (opensearch, qdrant):
        if manifest.get("benchmark_id") != args.benchmark_id:
            raise ValueError("engine benchmark_id mismatch")
        if manifest.get("parent_chunks_artifact_fingerprint") != args.chunks_fingerprint:
            raise ValueError("engine chunks fingerprint mismatch")
    if qdrant.get("parent_embeddings_artifact_fingerprint") != args.embeddings_fingerprint:
        raise ValueError("Qdrant embeddings fingerprint mismatch")
    fingerprint = combined_index_fingerprint(
        benchmark_id=args.benchmark_id,
        chunks_fingerprint=args.chunks_fingerprint,
        embeddings_fingerprint=args.embeddings_fingerprint,
        opensearch_fingerprint=args.opensearch_fingerprint,
        qdrant_fingerprint=args.qdrant_fingerprint,
    )
    artifact_uri = f"{args.remote_base.rstrip('/')}/{fingerprint}"
    manifest = {
        "schema_version": INDEX_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "native-indexes",
        "artifact_fingerprint": fingerprint,
        "benchmark_id": args.benchmark_id,
        "parent_chunks_artifact_fingerprint": args.chunks_fingerprint,
        "parent_embeddings_artifact_fingerprint": args.embeddings_fingerprint,
        "engines": {
            "opensearch": {
                "engine_fingerprint": args.opensearch_fingerprint,
                "artifact_uri": opensearch["persistence"]["artifact_uri"],
                "backend": opensearch["backend"],
                "document_count": opensearch["document_count"],
            },
            "qdrant": {
                "engine_fingerprint": args.qdrant_fingerprint,
                "artifact_uri": qdrant["persistence"]["artifact_uri"],
                "backend": qdrant["backend"],
                "point_count": qdrant["point_count"],
            },
        },
        "repository_revision": os.environ.get("GITHUB_SHA", "local"),
        "persistence": {
            "backend": "huggingface-storage-bucket",
            "private": True,
            "persisted": True,
            "artifact_uri": artifact_uri,
        },
    }
    _write_json(args.output, manifest)
    return manifest


def _base_parser(parser: argparse.ArgumentParser, *, embeddings: bool = False) -> None:
    parser.add_argument("--chunks-root", type=Path, required=True)
    if embeddings:
        parser.add_argument("--embeddings-root", type=Path, required=True)
        parser.add_argument("--embeddings-fingerprint", required=True)
    parser.add_argument("--benchmark-id", required=True)
    parser.add_argument("--chunks-fingerprint", required=True)
    parser.add_argument("--backend-version", required=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    identity = sub.add_parser("identity")
    identity.add_argument("--engine", choices=("opensearch", "qdrant"), required=True)
    _base_parser(identity, embeddings=False)
    identity.add_argument("--embeddings-root", type=Path)
    identity.add_argument("--embeddings-fingerprint")
    identity.add_argument("--output", type=Path, required=True)

    opensearch = sub.add_parser("opensearch")
    _base_parser(opensearch)
    opensearch.add_argument("--url", default="http://localhost:9200")
    opensearch.add_argument("--remote-root", required=True)
    opensearch.add_argument("--repo-dir", type=Path, required=True)
    opensearch.add_argument("--work-dir", type=Path, required=True)
    opensearch.add_argument("--batch-size", type=int, default=2000)
    opensearch.add_argument("--checkpoint-rows", type=int, default=50000)

    qdrant = sub.add_parser("qdrant")
    _base_parser(qdrant, embeddings=True)
    qdrant.add_argument("--url", default="http://localhost:6333")
    qdrant.add_argument("--remote-root", required=True)
    qdrant.add_argument("--work-dir", type=Path, required=True)
    qdrant.add_argument("--batch-size", type=int, default=256)
    qdrant.add_argument("--checkpoint-rows", type=int, default=50000)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--benchmark-id", required=True)
    finalize.add_argument("--chunks-fingerprint", required=True)
    finalize.add_argument("--embeddings-fingerprint", required=True)
    finalize.add_argument("--opensearch-fingerprint", required=True)
    finalize.add_argument("--qdrant-fingerprint", required=True)
    finalize.add_argument("--opensearch-manifest", type=Path, required=True)
    finalize.add_argument("--qdrant-manifest", type=Path, required=True)
    finalize.add_argument("--remote-base", required=True)
    finalize.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    try:
        if args.command == "identity":
            if args.engine == "opensearch":
                result = _opensearch_identity(args)
            else:
                result = _qdrant_identity(args)
            _write_json(args.output, result)
        elif args.command == "opensearch":
            result = build_opensearch(args)
        elif args.command == "qdrant":
            result = build_qdrant(args)
        else:
            result = finalize_indexes(args)
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        import traceback

        print(f"Benchmark index stage failed: {exc}", file=os.sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
