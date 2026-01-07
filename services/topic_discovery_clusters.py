from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Iterable

import hdbscan
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from umap import UMAP
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
    *,
    use_umap: bool = False,
    umap_config: dict | None = None,
) -> tuple[list[int], list[float], list[dict]]:
    vector_count = len(vectors)
    if not vectors:
        return [], [], []
    if vector_count < 2:
        logger.warning(
            "Not enough vectors for HDBSCAN (count=%s). Returning all outliers.",
            vector_count,
        )
        return [-1] * vector_count, [0.0] * vector_count, []
    effective_min_cluster_size = min(min_cluster_size, vector_count)
    effective_min_samples = min(min_samples, vector_count)
    if (
        effective_min_cluster_size != min_cluster_size
        or effective_min_samples != min_samples
    ):
        logger.warning(
            "Adjusting HDBSCAN params for vector count=%s: min_cluster_size=%s->%s, min_samples=%s->%s",
            vector_count,
            min_cluster_size,
            effective_min_cluster_size,
            min_samples,
            effective_min_samples,
        )
    data = np.asarray(vectors, dtype=np.float32)
    normalized = _l2_normalize_matrix(data)
    hdbscan_data = normalized
    effective_metric = metric
    if use_umap:
        hdbscan_data = _apply_umap(normalized, umap_config or {})
        effective_metric = "euclidean"
    elif metric.lower() == "cosine":
        # HDBSCAN's BallTree backend does not accept cosine; for unit vectors,
        # Euclidean distance is monotonic with cosine distance.
        effective_metric = "euclidean"
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=effective_min_cluster_size,
        min_samples=effective_min_samples,
        metric=effective_metric,
    )
    labels = clusterer.fit_predict(hdbscan_data)
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
    topic_parent_map: dict[int, int] | None = None,
    parent_summaries: list[dict] | None = None,
    file_assignments: dict[str, dict] | None = None,
    macro_metrics: dict | None = None,
) -> dict:
    checksums_hash = _hash_checksums(checksums)
    result = {
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
    if topic_parent_map is not None:
        result["topic_parent_map"] = {str(key): value for key, value in topic_parent_map.items()}
    if parent_summaries is not None:
        result["parent_summaries"] = parent_summaries
    if file_assignments is not None:
        result["file_assignments"] = file_assignments
    if macro_metrics is not None:
        result["macro_metrics"] = macro_metrics
    return result


def run_topic_discovery_clustering(
    *,
    min_cluster_size: int,
    min_samples: int,
    metric: str = "cosine",
    use_umap: bool = False,
    umap_config: dict | None = None,
    macro_k_range: tuple[int, int] = (5, 10),
    allow_cache: bool = True,
) -> tuple[dict | None, bool]:
    checksums, vectors, payloads = load_all_file_vectors()
    if not checksums:
        return None, False
    vector_count = len(checksums)
    checksums_hash = _hash_checksums(checksums)
    params = {
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "metric": metric,
        "use_umap": use_umap,
        "umap": umap_config if use_umap else None,
        "macro_grouping": {"min_k": int(macro_k_range[0]), "max_k": int(macro_k_range[1])},
    }
    if allow_cache:
        cached = load_last_cluster_cache()
        if _is_cache_valid(cached, params, vector_count, checksums_hash):
            upgraded = _ensure_macro_grouping(
                cached,
                macro_k_range=macro_k_range,
            )
            if upgraded is not cached:
                save_cluster_cache(upgraded)
            return upgraded, True
    labels, probs, clusters = run_hdbscan(
        vectors=vectors,
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric=metric,
        use_umap=use_umap,
        umap_config=umap_config,
    )
    clusters = attach_representative_checksums(clusters, checksums)
    topic_parent_map, parent_summaries, macro_metrics = _macro_group_topics(
        clusters=clusters,
        macro_k_range=macro_k_range,
    )
    enforced = enforce_max_parent_share(
        {
            "clusters": clusters,
            "topic_parent_map": topic_parent_map,
            "parent_summaries": parent_summaries,
            "macro_metrics": macro_metrics,
        }
    )
    if enforced:
        topic_parent_map = enforced["topic_parent_map"]
        parent_summaries = enforced["parent_summaries"]
        macro_metrics = enforced["macro_metrics"]
    file_assignments = _build_file_assignments(
        checksums=checksums,
        labels=labels,
        probs=probs,
        topic_parent_map=topic_parent_map,
    )
    result = build_cluster_cache_result(
        checksums=checksums,
        payloads=payloads,
        labels=labels,
        probs=probs,
        clusters=clusters,
        params=params,
        topic_parent_map=topic_parent_map,
        parent_summaries=parent_summaries,
        file_assignments=file_assignments,
        macro_metrics=macro_metrics,
    )
    save_cluster_cache(result)
    return result, False


def _is_cache_valid(
    cached: dict | None,
    params: dict,
    vector_count: int,
    checksums_hash: str,
) -> bool:
    if not cached:
        return False
    return (
        cached.get("collection") == QDRANT_FILE_VECTORS_COLLECTION
        and cached.get("params") == params
        and cached.get("vector_count") == vector_count
        and cached.get("checksums_hash") == checksums_hash
    )


def _ensure_macro_grouping(
    result: dict,
    *,
    macro_k_range: tuple[int, int],
) -> dict:
    clusters = result.get("clusters", [])
    topic_parent_map = {
        int(topic_id): int(parent_id)
        for topic_id, parent_id in (result.get("topic_parent_map") or {}).items()
    }
    parent_summaries = result.get("parent_summaries")
    macro_metrics = result.get("macro_metrics")
    if not topic_parent_map or parent_summaries is None or macro_metrics is None:
        topic_parent_map, parent_summaries, macro_metrics = _macro_group_topics(
            clusters=clusters,
            macro_k_range=macro_k_range,
        )
    enforced = enforce_max_parent_share(
        {
            "clusters": clusters,
            "topic_parent_map": topic_parent_map,
            "parent_summaries": parent_summaries,
            "macro_metrics": macro_metrics,
        }
    )
    if enforced:
        topic_parent_map = enforced["topic_parent_map"]
        parent_summaries = enforced["parent_summaries"]
        macro_metrics = enforced["macro_metrics"]
    file_assignments = _build_file_assignments(
        checksums=result.get("checksums", []),
        labels=result.get("labels", []),
        probs=result.get("probs", []),
        topic_parent_map=topic_parent_map,
    )
    updated = dict(result)
    updated.setdefault("params", {})
    updated["params"].setdefault(
        "macro_grouping",
        {"min_k": int(macro_k_range[0]), "max_k": int(macro_k_range[1])},
    )
    updated["topic_parent_map"] = {
        str(key): value for key, value in topic_parent_map.items()
    }
    updated["parent_summaries"] = parent_summaries
    updated["file_assignments"] = file_assignments
    updated["macro_metrics"] = macro_metrics
    return updated


def ensure_macro_grouping(
    result: dict,
    *,
    macro_k_range: tuple[int, int],
) -> dict:
    return _ensure_macro_grouping(result, macro_k_range=macro_k_range)


def _macro_group_topics(
    *,
    clusters: list[dict],
    macro_k_range: tuple[int, int],
) -> tuple[dict[int, int], list[dict], dict]:
    topic_ids = [int(cluster["cluster_id"]) for cluster in clusters if cluster.get("cluster_id") is not None]
    if not topic_ids:
        return {}, [], {"selected_k": 0, "silhouette": None, "largest_parent_share": 0.0, "candidates": []}
    centroids = np.array([cluster["centroid"] for cluster in clusters], dtype=np.float32)
    topic_sizes = np.array([cluster.get("size", 0) for cluster in clusters], dtype=np.float32)
    total_files = float(topic_sizes.sum()) if topic_sizes.size else 0.0
    min_k = int(macro_k_range[0])
    max_k = int(macro_k_range[1])
    if min_k > max_k:
        min_k, max_k = max_k, min_k
    max_k = min(max_k, len(topic_ids))
    candidates = [
        k
        for k in range(max(min_k, 2), max_k + 1)
        if k < len(topic_ids)
    ]
    if not candidates:
        parent_map = {topic_id: 0 for topic_id in topic_ids}
        parent_summaries = _build_parent_summaries(
            clusters=clusters,
            topic_parent_map=parent_map,
        )
        largest_parent_share = (
            parent_summaries[0]["total_files"] / total_files if parent_summaries and total_files else 0.0
        )
        macro_metrics = {
            "selected_k": 1,
            "silhouette": None,
            "largest_parent_share": largest_parent_share,
            "candidates": [],
        }
        return parent_map, parent_summaries, macro_metrics

    candidate_metrics: list[dict] = []
    best_choice: dict | None = None
    for k in candidates:
        model = _build_agglomerative_model(k)
        labels = model.fit_predict(centroids)
        silhouette = float(silhouette_score(centroids, labels, metric="cosine"))
        parent_sizes = np.zeros(k, dtype=np.float32)
        for label, size in zip(labels, topic_sizes, strict=False):
            parent_sizes[int(label)] += float(size)
        largest_parent_share = (
            float(parent_sizes.max()) / total_files if total_files else 0.0
        )
        metrics = {
            "k": int(k),
            "silhouette": silhouette,
            "largest_parent_share": largest_parent_share,
        }
        candidate_metrics.append(metrics)
        if best_choice is None:
            best_choice = metrics
            best_labels = labels
        else:
            if silhouette > best_choice["silhouette"] or (
                silhouette == best_choice["silhouette"]
                and largest_parent_share < best_choice["largest_parent_share"]
            ):
                best_choice = metrics
                best_labels = labels
    parent_map = {
        topic_id: int(parent_id)
        for topic_id, parent_id in zip(topic_ids, best_labels, strict=False)
    }
    parent_summaries = _build_parent_summaries(
        clusters=clusters,
        topic_parent_map=parent_map,
    )
    macro_metrics = {
        "selected_k": int(best_choice["k"]) if best_choice else 0,
        "silhouette": best_choice["silhouette"] if best_choice else None,
        "largest_parent_share": best_choice["largest_parent_share"] if best_choice else 0.0,
        "candidates": sorted(candidate_metrics, key=lambda item: item["k"]),
    }
    return parent_map, parent_summaries, macro_metrics


def _build_agglomerative_model(n_clusters: int) -> AgglomerativeClustering:
    kwargs = {"n_clusters": n_clusters, "linkage": "average"}
    if "metric" in inspect.signature(AgglomerativeClustering).parameters:
        kwargs["metric"] = "cosine"
    else:
        kwargs["affinity"] = "cosine"
    return AgglomerativeClustering(**kwargs)


def enforce_max_parent_share(
    macro_result: dict,
    *,
    max_share: float = 0.35,
) -> dict:
    macro_metrics = macro_result.get("macro_metrics", {})
    if macro_metrics.get("auto_split", {}).get("applied"):
        return macro_result
    topic_parent_map = {
        int(topic_id): int(parent_id)
        for topic_id, parent_id in macro_result.get("topic_parent_map", {}).items()
    }
    clusters = macro_result.get("clusters", [])
    if not topic_parent_map or not clusters:
        return macro_result
    parent_totals: dict[int, int] = {}
    for cluster in clusters:
        topic_id = int(cluster.get("cluster_id", -1))
        if topic_id not in topic_parent_map:
            continue
        parent_id = int(topic_parent_map[topic_id])
        parent_totals[parent_id] = parent_totals.get(parent_id, 0) + int(cluster.get("size", 0))
    total_files = sum(parent_totals.values())
    if total_files <= 0 or not parent_totals:
        return macro_result
    largest_parent_id = max(parent_totals, key=parent_totals.get)
    largest_share = parent_totals[largest_parent_id] / total_files if total_files else 0.0
    if largest_share <= max_share:
        return macro_result
    parent_topics = [
        cluster
        for cluster in clusters
        if topic_parent_map.get(int(cluster.get("cluster_id", -1))) == largest_parent_id
    ]
    if len(parent_topics) < 2:
        return macro_result
    centroids = np.asarray([cluster["centroid"] for cluster in parent_topics], dtype=np.float32)
    if "metric" in inspect.signature(AgglomerativeClustering).parameters:
        model = AgglomerativeClustering(n_clusters=2, linkage="average", metric="cosine")
        labels = model.fit_predict(centroids)
    else:
        distances = cosine_distances(centroids)
        model = AgglomerativeClustering(
            n_clusters=2,
            linkage="average",
            affinity="precomputed",
        )
        labels = model.fit_predict(distances)
    new_parent_base = max(topic_parent_map.values(), default=-1) + 1
    new_parent_ids = (new_parent_base, new_parent_base + 1)
    updated_parent_map = dict(topic_parent_map)
    for cluster, label in zip(parent_topics, labels, strict=False):
        topic_id = int(cluster.get("cluster_id", -1))
        if topic_id < 0:
            continue
        updated_parent_map[topic_id] = new_parent_ids[int(label)]
    parent_summaries = _build_parent_summaries(
        clusters=clusters,
        topic_parent_map=updated_parent_map,
    )
    largest_parent_share = (
        max((summary["total_files"] for summary in parent_summaries), default=0) / total_files
        if total_files
        else 0.0
    )
    new_share_lookup = {summary["parent_id"]: summary["total_files"] / total_files for summary in parent_summaries}
    macro_metrics = dict(macro_metrics)
    macro_metrics["selected_k"] = len(parent_summaries)
    macro_metrics["largest_parent_share"] = largest_parent_share
    macro_metrics["auto_split"] = {
        "applied": True,
        "original_parent_id": largest_parent_id,
        "original_share": largest_share,
        "new_parent_ids": new_parent_ids,
        "new_shares": [new_share_lookup.get(new_parent_ids[0], 0.0), new_share_lookup.get(new_parent_ids[1], 0.0)],
    }
    return {
        "topic_parent_map": updated_parent_map,
        "parent_summaries": parent_summaries,
        "macro_metrics": macro_metrics,
    }


def _build_parent_summaries(
    *,
    clusters: list[dict],
    topic_parent_map: dict[int, int],
) -> list[dict]:
    summaries: dict[int, dict] = {}
    for cluster in clusters:
        topic_id = int(cluster.get("cluster_id", -1))
        if topic_id not in topic_parent_map:
            continue
        parent_id = int(topic_parent_map[topic_id])
        size = int(cluster.get("size", 0))
        avg_prob = float(cluster.get("avg_prob", 0.0))
        summary = summaries.setdefault(
            parent_id,
            {"parent_id": parent_id, "total_files": 0, "n_topics": 0, "prob_sum": 0.0},
        )
        summary["total_files"] += size
        summary["n_topics"] += 1
        summary["prob_sum"] += avg_prob * size
    parent_summaries: list[dict] = []
    for parent_id, summary in summaries.items():
        total_files = summary["total_files"]
        avg_child_prob = summary["prob_sum"] / total_files if total_files else 0.0
        parent_summaries.append(
            {
                "parent_id": parent_id,
                "total_files": total_files,
                "n_topics": summary["n_topics"],
                "avg_child_prob": avg_child_prob,
            }
        )
    return sorted(parent_summaries, key=lambda item: item["total_files"], reverse=True)


def _build_file_assignments(
    *,
    checksums: list[str],
    labels: list[int],
    probs: list[float],
    topic_parent_map: dict[int, int],
) -> dict[str, dict]:
    assignments: dict[str, dict] = {}
    for checksum, label, prob in zip(checksums, labels, probs, strict=False):
        topic_id = int(label)
        parent_id = topic_parent_map.get(topic_id, -1) if topic_id >= 0 else -1
        assignments[str(checksum)] = {
            "topic_id": topic_id,
            "parent_id": int(parent_id),
            "prob": float(prob),
        }
    return assignments


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


def _apply_umap(matrix: np.ndarray, config: dict) -> np.ndarray:
    sample_count = matrix.shape[0]
    if sample_count < 3:
        logger.warning(
            "Skipping UMAP reduction for vector count=%s; need at least 3 samples.",
            sample_count,
        )
        return matrix
    defaults = {
        "n_components": 10,
        "n_neighbors": 30,
        "min_dist": 0.1,
        "metric": "cosine",
        "random_state": 42,
    }
    params = {**defaults, **config}
    max_neighbors = max(2, sample_count - 1)
    n_neighbors = max(2, min(int(params["n_neighbors"]), max_neighbors))
    n_components = min(int(params["n_components"]), max(2, sample_count))
    if n_neighbors != params["n_neighbors"] or n_components != params["n_components"]:
        logger.warning(
            "Adjusting UMAP params for vector count=%s: n_neighbors=%s->%s, n_components=%s->%s",
            sample_count,
            params["n_neighbors"],
            n_neighbors,
            params["n_components"],
            n_components,
        )
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=float(params["min_dist"]),
        metric=str(params["metric"]),
        random_state=params["random_state"],
    )
    return reducer.fit_transform(matrix)
