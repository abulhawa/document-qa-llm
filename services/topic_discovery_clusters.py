from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import hdbscan
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

from config import QDRANT_FILE_VECTORS_COLLECTION, QDRANT_URL, logger


SCROLL_BATCH_SIZE = 256
REPRESENTATIVE_COUNT = 10
CACHE_LAST_FILENAME = "topic_discovery_clusters_last.json"

client = QdrantClient(url=QDRANT_URL)


def load_all_file_vectors() -> tuple[list[str], list[list[float]], list[dict]]:
    if not _collection_exists(QDRANT_FILE_VECTORS_COLLECTION):
        return [], [], []
    checksums: list[str] = []
    vectors: list[list[float]] = []
    payloads: list[dict] = []
    for point in _scroll_collection(
        collection_name=QDRANT_FILE_VECTORS_COLLECTION,
        with_vectors=True,
        with_payload=True,
    ):
        vector = _extract_vector(point)
        if vector is None:
            continue
        payload = _extract_payload(point)
        checksum = payload.get("checksum") or _extract_id(point)
        if not checksum:
            continue
        checksums.append(str(checksum))
        vectors.append(vector)
        payloads.append(payload)
    return checksums, vectors, payloads


def run_hdbscan(
    vectors: list[list[float]],
    min_cluster_size: int,
    min_samples: int,
    metric: str = "cosine",
) -> tuple[list[int], list[float], list[dict]]:
    if not vectors:
        return [], [], []
    data = np.asarray(vectors, dtype=np.float32)
    normalized = _l2_normalize_matrix(data)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(normalized)
    probs = clusterer.probabilities_

    clusters: list[dict] = []
    for cluster_id in sorted({label for label in labels if label >= 0}):
        member_indices = np.where(labels == cluster_id)[0]
        if member_indices.size == 0:
            continue
        member_vectors = normalized[member_indices]
        centroid = member_vectors.mean(axis=0)
        centroid = _l2_normalize_vector(centroid)
        similarities = member_vectors @ centroid
        top_indices = np.argsort(-similarities)[:REPRESENTATIVE_COUNT]
        representative_indices = [int(member_indices[idx]) for idx in top_indices]
        cluster_probs = probs[member_indices]
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(member_indices.size),
                "avg_prob": float(np.mean(cluster_probs)) if cluster_probs.size else 0.0,
                "centroid": centroid.tolist(),
                "representative_indices": representative_indices,
            }
        )
    return labels.astype(int).tolist(), probs.astype(float).tolist(), clusters


def attach_representative_checksums(
    clusters: list[dict],
    checksums: list[str],
) -> list[dict]:
    enriched: list[dict] = []
    for cluster in clusters:
        rep_indices = cluster.get("representative_indices", [])
        representative_checksums = [
            checksums[idx] for idx in rep_indices if 0 <= idx < len(checksums)
        ]
        enriched_cluster = dict(cluster)
        enriched_cluster.pop("representative_indices", None)
        enriched_cluster["representative_checksums"] = representative_checksums
        enriched.append(enriched_cluster)
    return enriched


def build_cluster_cache_result(
    *,
    checksums: list[str],
    payloads: list[dict],
    labels: list[int],
    probs: list[float],
    clusters: list[dict],
    params: dict,
) -> dict:
    checksums_hash = _hash_checksums(checksums)
    return {
        "collection": QDRANT_FILE_VECTORS_COLLECTION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "vector_count": len(checksums),
        "checksums_hash": checksums_hash,
        "checksums": checksums,
        "payloads": payloads,
        "labels": labels,
        "probs": probs,
        "clusters": clusters,
    }


def save_cluster_cache(result: dict) -> Path:
    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _cache_filename(result)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle)
    last_path = cache_dir / CACHE_LAST_FILENAME
    with last_path.open("w", encoding="utf-8") as handle:
        json.dump({"path": str(cache_path)}, handle)
    return cache_path


def load_last_cluster_cache() -> dict | None:
    cache_dir = _get_cache_dir()
    last_path = cache_dir / CACHE_LAST_FILENAME
    if not last_path.exists():
        return None
    try:
        with last_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cache_path = Path(payload.get("path", ""))
        if not cache_path.exists():
            return None
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load cluster cache: %s", exc)
        return None


def cluster_cache_exists() -> bool:
    cache_dir = _get_cache_dir()
    return (cache_dir / CACHE_LAST_FILENAME).exists()


def _cache_filename(result: dict) -> str:
    params = result.get("params", {})
    key = _cache_key(params, result.get("vector_count", 0), result.get("checksums_hash", ""))
    collection = result.get("collection", QDRANT_FILE_VECTORS_COLLECTION)
    safe_collection = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in collection)
    return f"topic_discovery_{safe_collection}_{key}.json"


def _cache_key(params: dict, vector_count: int, checksums_hash: str) -> str:
    params_blob = json.dumps(params, sort_keys=True)
    digest = hashlib.sha256(
        f"{params_blob}:{vector_count}:{checksums_hash}".encode("utf-8")
    ).hexdigest()
    return digest[:12]


def _hash_checksums(checksums: list[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(checksums)).encode("utf-8")).hexdigest()
    return digest


def _get_cache_dir() -> Path:
    base_dir = os.getenv("APP_DATA_DIR", "data")
    return Path(base_dir) / "topic_discovery"


def _collection_exists(collection_name: str) -> bool:
    collections = client.get_collections().collections
    return collection_name in [c.name for c in collections]


def _scroll_collection(
    *,
    collection_name: str,
    scroll_filter: models.Filter | None = None,
    with_vectors: bool = False,
    with_payload: bool = True,
    batch_size: int = SCROLL_BATCH_SIZE,
) -> Iterable[models.Record]:
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=scroll_filter,
            limit=batch_size,
            with_vectors=with_vectors,
            with_payload=with_payload,
            offset=offset,
        )
        for point in points:
            yield point
        if offset is None:
            break


def _extract_payload(point: models.Record) -> dict:
    if isinstance(point, dict):
        return point.get("payload", {}) or {}
    payload = getattr(point, "payload", None)
    return payload or {}


def _extract_vector(point: models.Record) -> list[float] | None:
    vector = None
    if isinstance(point, dict):
        vector = point.get("vector")
    else:
        vector = getattr(point, "vector", None)
    if vector is None:
        return None
    if isinstance(vector, dict):
        if vector:
            return list(next(iter(vector.values())))
        return None
    return list(vector)


def _extract_id(point: models.Record) -> str | None:
    if isinstance(point, dict):
        return str(point.get("id")) if point.get("id") is not None else None
    point_id = getattr(point, "id", None)
    return str(point_id) if point_id is not None else None


def _l2_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm
