"""Apply benchmark corpus deltas to verified native index snapshots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterator

from scripts import build_benchmark_indexes as base
from scripts.seed_benchmark_cache import signature


def _predecessor(args: argparse.Namespace, engine: str, identity: dict[str, Any]) -> dict[str, Any]:
    args.work_dir.mkdir(parents=True, exist_ok=True)
    path = args.work_dir / "predecessor-manifest.json"
    if not base._download_optional(f"{args.base_uri}/manifest.json", path):
        raise FileNotFoundError("predecessor index manifest is missing")
    manifest = base._json(path)
    if manifest.get("artifact_type") != "native-index-engine" or manifest.get("engine") != engine:
        raise ValueError("predecessor index type mismatch")
    if manifest.get("benchmark_id") != args.base_benchmark_id:
        raise ValueError("predecessor benchmark mismatch")
    if manifest.get("backend", {}).get("version") != args.backend_version:
        raise ValueError("backend version changed; cannot restore predecessor snapshot")
    if manifest.get("persistence", {}).get("artifact_uri") != args.base_uri:
        raise ValueError("predecessor artifact URI mismatch")
    if not manifest.get("persistence", {}).get("persisted"):
        raise ValueError("predecessor index is not persisted")
    if engine == "opensearch":
        if manifest["backend"].get("settings_sha256") != identity["settings_sha256"]:
            raise ValueError("OpenSearch index settings changed")
        if manifest["backend"].get("index_name") != base.OPENSEARCH_INDEX_NAME:
            raise ValueError("OpenSearch index name changed")
        if manifest.get("snapshot", {}).get("format") != "opensearch-fs-repository-tar":
            raise ValueError("OpenSearch snapshot format changed")
    else:
        if manifest["backend"].get("vector_dimension") != identity["vector_dimension"]:
            raise ValueError("Qdrant vector dimension changed")
        if manifest["backend"].get("distance") != "cosine":
            raise ValueError("Qdrant distance contract changed")
        if manifest["backend"].get("collection_name") != base.QDRANT_COLLECTION_NAME:
            raise ValueError("Qdrant collection name changed")
        if manifest["backend"].get("payload_keys") != list(base.QDRANT_PAYLOAD_KEYS):
            raise ValueError("Qdrant payload contract changed")
        if manifest.get("snapshot", {}).get("format") != "qdrant-collection-snapshot":
            raise ValueError("Qdrant snapshot format changed")
    return manifest


def _snapshot(args: argparse.Namespace, manifest: dict[str, Any]) -> Path:
    meta = manifest["snapshot"]
    path = args.work_dir / f"predecessor-{meta['filename']}"
    if not base._download_optional(f"{args.base_uri}/{meta['filename']}", path):
        raise FileNotFoundError("predecessor native snapshot is missing")
    if path.stat().st_size != meta["bytes"] or base._sha256_file(path) != meta["sha256"]:
        raise ValueError("predecessor native snapshot checksum mismatch")
    return path


def _remote_done(args: argparse.Namespace, engine: str, identity: dict[str, Any]) -> dict[str, Any] | None:
    path = args.work_dir / "existing-final-manifest.json"
    if base._download_optional(f"{args.remote_root}/manifest.json", path):
        result = base._json(path)
        if not base._engine_manifest_matches(result, engine=engine, engine_fingerprint=identity["engine_fingerprint"]):
            raise ValueError("existing final index has incompatible identity")
        return result
    return None


def _batched(items: Iterator[Any], size: int) -> Iterator[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def opensearch(args: argparse.Namespace) -> dict[str, Any]:
    from opensearchpy import OpenSearch, helpers

    chunks = base._validate_chunks(args.chunks_root, benchmark_id=args.benchmark_id, chunks_fingerprint=args.chunks_fingerprint)
    identity = base._opensearch_identity(args)
    done = _remote_done(args, "opensearch", identity)
    if done:
        return done
    prior = _predecessor(args, "opensearch", identity)
    snapshot = _snapshot(args, prior)
    base._wait_http(f"{args.url}/_cluster/health")
    args.repo_dir.mkdir(parents=True, exist_ok=True)
    base._extract_tar(snapshot, args.repo_dir)
    client = OpenSearch(hosts=[args.url], timeout=120)
    client.snapshot.create_repository(
        repository=base.OPENSEARCH_REPOSITORY,
        body={"type": "fs", "settings": {"location": "/mnt/opensearch-backups", "compress": True}},
    )
    result = client.snapshot.restore(
        repository=base.OPENSEARCH_REPOSITORY,
        snapshot=prior["snapshot"]["name"],
        body={"indices": base.OPENSEARCH_INDEX_NAME, "include_global_state": False},
        params={"wait_for_completion": "true"},
    )
    if result.get("snapshot", {}).get("shards", {}).get("failed", 0):
        raise RuntimeError(f"OpenSearch predecessor restore failed: {result}")
    old_count = int(client.count(index=base.OPENSEARCH_INDEX_NAME)["count"])
    if old_count != int(prior["document_count"]):
        raise ValueError("restored OpenSearch count disagrees with predecessor manifest")
    old_ids = {
        hit["_id"]
        for hit in helpers.scan(client, index=base.OPENSEARCH_INDEX_NAME, query={"query": {"match_all": {}}}, _source=False)
    }
    if len(old_ids) != old_count:
        raise ValueError("duplicate or missing OpenSearch document identity")
    client.indices.put_settings(index=base.OPENSEARCH_INDEX_NAME, body={"index": {"refresh_interval": "-1"}})
    seen: set[str] = set()
    added = 0
    actions: list[dict[str, Any]] = []

    def flush() -> None:
        if actions:
            success, errors = helpers.bulk(client, actions, raise_on_error=False, request_timeout=120)
            if errors or success != len(actions):
                raise RuntimeError(f"OpenSearch delta bulk failed: {errors[:3]}")
            actions.clear()

    for _, row in base._iter_chunks(args.chunks_root):
        key = base._backend_chunk_id(row)
        if key in seen:
            raise ValueError("duplicate target OpenSearch document identity")
        seen.add(key)
        if key not in old_ids:
            actions.append({"_op_type": "index", "_index": base.OPENSEARCH_INDEX_NAME, "_id": key, "_source": base._chunk_source(row)})
            added += 1
            if len(actions) >= args.batch_size:
                flush()
    flush()
    removed_ids = old_ids - seen
    for group in _batched(iter(removed_ids), args.batch_size):
        actions.extend({"_op_type": "delete", "_index": base.OPENSEARCH_INDEX_NAME, "_id": key} for key in group)
        flush()
    expected = int(chunks.get("total_chunks") or len(seen))
    if len(seen) != expected:
        raise ValueError("target chunk manifest count mismatch")
    client.indices.put_settings(index=base.OPENSEARCH_INDEX_NAME, body={"index": {"refresh_interval": "1s"}})
    client.indices.refresh(index=base.OPENSEARCH_INDEX_NAME)
    client.indices.flush(index=base.OPENSEARCH_INDEX_NAME)
    count = int(client.count(index=base.OPENSEARCH_INDEX_NAME)["count"])
    if count != expected:
        raise ValueError(f"OpenSearch delta count mismatch: {count} != {expected}")
    base._create_opensearch_snapshot(client, base.OPENSEARCH_FINAL_SNAPSHOT)
    target_snapshot = args.work_dir / "opensearch-snapshot.tar.gz"
    base._tar_directory(args.repo_dir, target_snapshot)
    digest = base._sha256_file(target_snapshot)
    base._upload(target_snapshot, f"{args.remote_root}/snapshot.tar.gz")
    manifest = {
        "schema_version": base.INDEX_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "native-index-engine", "engine": "opensearch",
        "engine_fingerprint": identity["engine_fingerprint"],
        "checkpoint_signature": identity["checkpoint_signature"],
        "benchmark_id": args.benchmark_id,
        "parent_chunks_artifact_fingerprint": args.chunks_fingerprint,
        "backend": {"name": "opensearch", "version": args.backend_version,
                    "index_name": base.OPENSEARCH_INDEX_NAME,
                    "settings_sha256": identity["settings_sha256"]},
        "document_count": count,
        "snapshot": {"format": "opensearch-fs-repository-tar", "name": base.OPENSEARCH_FINAL_SNAPSHOT,
                     "filename": "snapshot.tar.gz", "sha256": digest, "bytes": target_snapshot.stat().st_size},
        "reuse": {"base_benchmark_id": args.base_benchmark_id,
                  "base_engine_fingerprint": prior["engine_fingerprint"],
                  "entries_reused": len(seen & old_ids), "entries_added": added,
                  "entries_removed": len(removed_ids)},
        "repository_revision": os.environ.get("GITHUB_SHA", "local"),
        "persistence": {"backend": "huggingface-storage-bucket", "private": True,
                        "persisted": True, "artifact_uri": args.remote_root},
    }
    path = args.work_dir / "opensearch-final-manifest.json"
    base._write_json(path, manifest)
    base._upload(path, f"{args.remote_root}/manifest.json")
    return manifest


def qdrant(args: argparse.Namespace) -> dict[str, Any]:
    from qdrant_client import QdrantClient, models

    chunks = base._validate_chunks(args.chunks_root, benchmark_id=args.benchmark_id, chunks_fingerprint=args.chunks_fingerprint)
    embeddings = base._validate_embeddings(args.embeddings_root, benchmark_id=args.benchmark_id,
                                           chunks_fingerprint=args.chunks_fingerprint,
                                           embeddings_fingerprint=args.embeddings_fingerprint)
    identity = base._qdrant_identity(args)
    done = _remote_done(args, "qdrant", identity)
    if done:
        return done
    prior = _predecessor(args, "qdrant", identity)
    base_embeddings_path = args.work_dir / "predecessor-embeddings.json"
    if not base._download_optional(args.base_embeddings_manifest, base_embeddings_path):
        raise FileNotFoundError("predecessor embedding manifest is missing")
    base_embeddings = base._json(base_embeddings_path)
    if base_embeddings.get("artifact_fingerprint") != prior["parent_embeddings_artifact_fingerprint"]:
        raise ValueError("predecessor index/embedding lineage mismatch")
    same_vectors = signature(base_embeddings, "embeddings") == signature(embeddings, "embeddings")
    snapshot = _snapshot(args, prior)
    base._wait_http(f"{args.url}/readyz")
    base._restore_qdrant_snapshot(args.url, base.QDRANT_COLLECTION_NAME, snapshot,
                                  prior["snapshot"].get("native_checksum"))
    client = QdrantClient(url=args.url, timeout=120)
    old_count = int(client.count(collection_name=base.QDRANT_COLLECTION_NAME, exact=True).count)
    if old_count != int(prior["point_count"]):
        raise ValueError("restored Qdrant count disagrees with predecessor manifest")
    old_ids: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(collection_name=base.QDRANT_COLLECTION_NAME,
                                       offset=offset, limit=1000, with_payload=False, with_vectors=False)
        old_ids.update(str(point.id) for point in points)
        if offset is None:
            break
    if len(old_ids) != old_count:
        raise ValueError("duplicate or missing Qdrant point identity")
    seen: set[str] = set()
    batch: list[Any] = []
    upserted = 0
    for _, row, vector in base._iter_qdrant_rows(args.chunks_root, args.embeddings_root, embeddings):
        key = base._backend_chunk_id(row)
        if key in seen:
            raise ValueError("duplicate target Qdrant point identity")
        seen.add(key)
        if key not in old_ids or not same_vectors:
            batch.append(models.PointStruct(id=key, vector=vector.tolist(), payload=base._qdrant_payload(row)))
            upserted += 1
            if len(batch) >= args.batch_size:
                client.upsert(collection_name=base.QDRANT_COLLECTION_NAME, points=batch, wait=True)
                batch.clear()
    if batch:
        client.upsert(collection_name=base.QDRANT_COLLECTION_NAME, points=batch, wait=True)
    removed_ids = old_ids - seen
    for group in _batched(iter(removed_ids), args.batch_size):
        client.delete(collection_name=base.QDRANT_COLLECTION_NAME,
                      points_selector=models.PointIdsList(points=list(group)), wait=True)
    expected = int(embeddings["chunk_embeddings"])
    if len(seen) != expected or int(chunks.get("total_chunks") or expected) != expected:
        raise ValueError("target chunk/embedding count mismatch")
    count = int(client.count(collection_name=base.QDRANT_COLLECTION_NAME, exact=True).count)
    if count != expected:
        raise ValueError(f"Qdrant delta count mismatch: {count} != {expected}")
    base._wait_qdrant_optimized(args.url, base.QDRANT_COLLECTION_NAME)
    target_snapshot = args.work_dir / "qdrant.snapshot"
    name, native_checksum = base._qdrant_snapshot(args.url, base.QDRANT_COLLECTION_NAME, target_snapshot)
    digest = base._sha256_file(target_snapshot)
    base._upload(target_snapshot, f"{args.remote_root}/snapshot.snapshot")
    manifest = {
        "schema_version": base.INDEX_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "native-index-engine", "engine": "qdrant",
        "engine_fingerprint": identity["engine_fingerprint"],
        "checkpoint_signature": identity["checkpoint_signature"],
        "benchmark_id": args.benchmark_id,
        "parent_chunks_artifact_fingerprint": args.chunks_fingerprint,
        "parent_embeddings_artifact_fingerprint": args.embeddings_fingerprint,
        "backend": {"name": "qdrant", "version": args.backend_version,
                    "collection_name": base.QDRANT_COLLECTION_NAME,
                    "vector_dimension": int(embeddings["vector_dimension"]),
                    "distance": "cosine", "payload_keys": list(base.QDRANT_PAYLOAD_KEYS)},
        "point_count": count,
        "snapshot": {"format": "qdrant-collection-snapshot", "name": name,
                     "filename": "snapshot.snapshot", "sha256": digest,
                     "native_checksum": native_checksum, "bytes": target_snapshot.stat().st_size},
        "reuse": {"base_benchmark_id": args.base_benchmark_id,
                  "base_engine_fingerprint": prior["engine_fingerprint"],
                  "entries_reused": len(seen & old_ids) if same_vectors else 0,
                  "entries_upserted": upserted, "entries_removed": len(removed_ids),
                  "embedding_signature_unchanged": same_vectors},
        "repository_revision": os.environ.get("GITHUB_SHA", "local"),
        "persistence": {"backend": "huggingface-storage-bucket", "private": True,
                        "persisted": True, "artifact_uri": args.remote_root},
    }
    path = args.work_dir / "qdrant-final-manifest.json"
    base._write_json(path, manifest)
    base._upload(path, f"{args.remote_root}/manifest.json")
    base._delete_qdrant_snapshot(args.url, base.QDRANT_COLLECTION_NAME, name)
    return manifest


def main() -> None:
    cli = argparse.ArgumentParser()
    sub = cli.add_subparsers(dest="engine", required=True)
    for engine in ("opensearch", "qdrant"):
        command = sub.add_parser(engine)
        base._base_parser(command, embeddings=engine == "qdrant")
        command.add_argument("--base-benchmark-id", required=True)
        command.add_argument("--base-uri", required=True)
        command.add_argument("--remote-root", required=True)
        command.add_argument("--url", required=True)
        command.add_argument("--work-dir", type=Path, required=True)
        command.add_argument("--batch-size", type=int, required=True)
        if engine == "opensearch":
            command.add_argument("--repo-dir", type=Path, required=True)
        else:
            command.add_argument("--base-embeddings-manifest", required=True)
    args = cli.parse_args()
    print(json.dumps({"opensearch": opensearch, "qdrant": qdrant}[args.engine](args), sort_keys=True))


if __name__ == "__main__":
    main()

