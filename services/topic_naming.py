from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from config import CHUNKS_INDEX, FULLTEXT_INDEX, QDRANT_COLLECTION, QDRANT_URL, logger
from core.llm import ask_llm_with_status, check_llm_status
from core.opensearch_client import get_client
from services.topic_discovery_clusters import load_last_cluster_cache
from utils.opensearch.fulltext import get_fulltext_by_checksum
from utils.timing import timed_block
from qdrant_client import QdrantClient
from qdrant_client.http import models


CACHE_DIR = Path(".cache") / "topic_naming"
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_KEYWORDS = 20
DEFAULT_MAX_PATH_DEPTH = 4
DEFAULT_ROOT_PATH = ""
DEFAULT_TOP_EXTENSION_COUNT = 5
MIXEDNESS_COMPONENT_WEIGHTS = {
    "keyword_entropy": 0.45,
    "extension_entropy": 0.2,
    "embedding_spread": 0.35,
}
MIXEDNESS_RANGE_SCALE = 0.85
LLM_UNAVAILABLE_WARNING = (
    "LLM inactive (model not loaded). Using baseline names instead of LLM suggestions."
)
CACHE_BYPASS_WARNING = "Cache bypassed (regenerated)"
LLM_UNAVAILABLE_CACHE_BYPASS_WARNING = (
    "LLM unavailable; regenerated baseline (cache bypassed)"
)
FALLBACK_REASON_MESSAGES = {
    "cache_hit_baseline": "Loaded cached baseline (LLM not called)",
    "llm_model_not_loaded": LLM_UNAVAILABLE_WARNING,
    "llm_timeout": "LLM request timed out (used baseline)",
    "llm_unreachable": "LLM server unreachable (used baseline)",
    "llm_invalid_json": "LLM returned invalid JSON (used baseline)",
    "skipped_due_to_mixedness": "Skipped LLM due to high mixedness (used baseline)",
    "skipped_due_to_prompt_too_long": "Skipped LLM because prompt was too long (used baseline)",
    "other_exception": "LLM request failed (used baseline)",
}

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b")
_LONG_NUMBER_RE = re.compile(r"\b\d{6,}\b")
_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_-]{24,}\b")
_NON_WORD_RE = re.compile(r"[^A-Za-z0-9]+")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}
_GERMAN_STOPWORDS = {
    "als",
    "aber",
    "am",
    "an",
    "auch",
    "auf",
    "aus",
    "bei",
    "bin",
    "bis",
    "da",
    "das",
    "dass",
    "dem",
    "den",
    "der",
    "des",
    "die",
    "doch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "für",
    "hat",
    "im",
    "in",
    "ins",
    "ist",
    "mit",
    "nach",
    "nicht",
    "noch",
    "nur",
    "oder",
    "sich",
    "sein",
    "sind",
    "und",
    "von",
    "vom",
    "war",
    "wie",
    "zu",
    "zum",
    "zur",
}


def _timing_verbose() -> bool:
    return os.getenv("TIMING_VERBOSE") == "1"

_NAME_MAX_WORDS = 6
_NAME_MAX_CHARS = 60


@dataclass(frozen=True)
class NameSuggestion:
    name: str
    confidence: float | None = None
    source: str = "heuristic"
    cache_key: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ClusterProfile:
    cluster_id: int
    size: int
    avg_prob: float
    centroid: list[float]
    keyword_entropy: float = 0.0
    extension_entropy: float = 0.0
    embedding_spread: float = 0.0
    mixedness: float = 0.0
    representative_checksums: list[str] = field(default_factory=list)
    representative_files: list[dict[str, Any]] = field(default_factory=list)
    representative_paths: list[str] = field(default_factory=list)
    representative_snippets: list[str] = field(default_factory=list)
    representative_snippet_metadata: dict[str, Any] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    top_extensions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ParentProfile:
    parent_id: int
    cluster_ids: list[int]
    size: int
    avg_prob: float
    centroid: list[float]
    keyword_entropy: float = 0.0
    extension_entropy: float = 0.0
    embedding_spread: float = 0.0
    mixedness: float = 0.0
    representative_checksums: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    top_extensions: list[dict[str, Any]] = field(default_factory=list)


def tokenize_filename(filename: str) -> list[str]:
    base = os.path.splitext(os.path.basename(filename))[0]
    tokens = [tok for tok in _NON_WORD_RE.split(base.lower()) if tok]
    return [tok for tok in tokens if tok not in _STOPWORDS]


def extract_path_segments(
    path: str,
    *,
    max_depth: int | None = None,
    root_path: str | None = None,
) -> list[str]:
    cleaned_path = path
    if root_path:
        try:
            root_value = str(Path(root_path))
            common = os.path.commonpath([root_value, str(path)])
            if common == root_value:
                cleaned_path = os.path.relpath(str(path), root_value)
        except ValueError:
            cleaned_path = path
    parts = [part for part in Path(cleaned_path).parts if part not in ("/", "")]
    if max_depth and max_depth > 0:
        parts = parts[-max_depth:]
    segments: list[str] = []
    for part in parts:
        segments.extend(tokenize_filename(part))
    return segments


def build_cluster_profile(
    cluster: dict[str, Any],
    checksum_payloads: dict[str, dict[str, Any]],
    *,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    max_path_depth: int | None = DEFAULT_MAX_PATH_DEPTH,
    root_path: str | None = DEFAULT_ROOT_PATH,
    top_extension_count: int = DEFAULT_TOP_EXTENSION_COUNT,
) -> ClusterProfile:
    representative_checksums = [
        str(checksum) for checksum in cluster.get("representative_checksums", [])
    ]
    representative_files = select_representative_files(
        cluster,
        checksum_payloads,
    )
    representative_paths = _extract_representative_paths(representative_files)
    snippets, snippet_metadata = select_representative_chunks_for_files(representative_files)
    keyword_counts = _keyword_counts_from_os(
        representative_checksums,
        snippets,
        max_keywords=max_keywords,
        max_path_depth=max_path_depth,
        root_path=root_path,
    )
    keywords = _top_keywords_from_counts(keyword_counts, max_keywords=max_keywords)
    keyword_entropy = _keyword_mixedness(keyword_counts, max_keywords=max_keywords)
    extension_counts = _extension_counts(representative_files)
    extension_entropy = _normalized_entropy(extension_counts)
    top_extensions = _format_top_extensions(
        extension_counts,
        limit=top_extension_count,
    )
    embedding_spread = _embedding_spread(
        representative_checksums,
        avg_prob=float(cluster.get("avg_prob", 0.0)),
    )
    mixedness = _combined_mixedness(
        keyword_entropy,
        extension_entropy,
        embedding_spread,
    )
    centroid = [float(val) for val in cluster.get("centroid", [])]
    return ClusterProfile(
        cluster_id=int(cluster.get("cluster_id", -1)),
        size=int(cluster.get("size", 0)),
        avg_prob=float(cluster.get("avg_prob", 0.0)),
        centroid=centroid,
        keyword_entropy=keyword_entropy,
        extension_entropy=extension_entropy,
        embedding_spread=embedding_spread,
        mixedness=mixedness,
        representative_checksums=representative_checksums,
        representative_files=representative_files,
        representative_paths=representative_paths,
        representative_snippets=snippets,
        representative_snippet_metadata=snippet_metadata,
        keywords=keywords,
        top_extensions=top_extensions,
    )


def build_parent_profile(
    parent_id: int,
    child_profiles: Sequence[ClusterProfile],
    *,
    top_extension_count: int = DEFAULT_TOP_EXTENSION_COUNT,
) -> ParentProfile:
    cluster_ids = [profile.cluster_id for profile in child_profiles]
    size = sum(profile.size for profile in child_profiles)
    avg_prob = _weighted_avg(
        [profile.avg_prob for profile in child_profiles],
        [profile.size for profile in child_profiles],
    )
    centroid = compute_centroid([profile.centroid for profile in child_profiles])
    keyword_entropy = _weighted_avg(
        [profile.keyword_entropy for profile in child_profiles],
        [profile.size for profile in child_profiles],
    )
    extension_entropy = _weighted_avg(
        [profile.extension_entropy for profile in child_profiles],
        [profile.size for profile in child_profiles],
    )
    embedding_spread = _weighted_avg(
        [profile.embedding_spread for profile in child_profiles],
        [profile.size for profile in child_profiles],
    )
    representative_checksums: list[str] = []
    keywords: list[str] = []
    extension_counts: dict[str, int] = {}
    for profile in child_profiles:
        representative_checksums.extend(profile.representative_checksums)
        keywords.extend(profile.keywords)
        for entry in profile.top_extensions:
            ext = str(entry.get("extension") or "")
            if not ext:
                continue
            extension_counts[ext] = extension_counts.get(ext, 0) + int(entry.get("count", 0))
    top_extensions = _format_top_extensions(extension_counts, limit=top_extension_count)
    if extension_counts:
        extension_entropy = _normalized_entropy(extension_counts)
    mixedness = _combined_mixedness(
        keyword_entropy,
        extension_entropy,
        embedding_spread,
    )
    return ParentProfile(
        parent_id=parent_id,
        cluster_ids=cluster_ids,
        size=size,
        avg_prob=avg_prob,
        centroid=centroid,
        keyword_entropy=keyword_entropy,
        extension_entropy=extension_entropy,
        embedding_spread=embedding_spread,
        mixedness=mixedness,
        representative_checksums=_dedupe_keep_order(representative_checksums),
        keywords=_dedupe_keep_order(keywords),
        top_extensions=top_extensions,
    )


def compute_centroid(vectors: Sequence[Sequence[float]]) -> list[float]:
    if not vectors:
        return []
    length = len(vectors[0])
    centroid = [0.0] * length
    count = 0
    for vec in vectors:
        if len(vec) != length:
            continue
        for idx, val in enumerate(vec):
            centroid[idx] += float(val)
        count += 1
    if count == 0:
        return []
    centroid = [val / count for val in centroid]
    norm = sum(val * val for val in centroid) ** 0.5
    if norm == 0:
        return centroid
    return [val / norm for val in centroid]


def select_representative_files(
    cluster: dict[str, Any],
    checksum_payloads: dict[str, dict[str, Any]],
    *,
    max_files: int = 10,
) -> list[dict[str, Any]]:
    checksums = [
        str(checksum) for checksum in cluster.get("representative_checksums", [])
    ]
    if not checksums:
        checksums = list(checksum_payloads.keys())
    selected_checksums = _select_representative_checksums(
        checksums,
        cluster=cluster,
        max_files=max_files,
    )
    representative_files: list[dict[str, Any]] = []
    for checksum in selected_checksums:
        payload = dict(checksum_payloads.get(checksum, {}))
        payload.setdefault("checksum", checksum)
        fulltext = _safe_fetch_fulltext(checksum)
        if fulltext:
            payload.setdefault("path", fulltext.get("path"))
            payload.setdefault("filename", fulltext.get("filename"))
            payload.setdefault("filetype", fulltext.get("filetype"))
            payload.setdefault("text_full", fulltext.get("text_full"))
        representative_files.append(payload)
    return representative_files


def _select_representative_checksums(
    checksums: Sequence[str],
    *,
    cluster: dict[str, Any],
    max_files: int,
) -> list[str]:
    if not checksums:
        return []
    embedding_payloads = _load_chunk_embeddings(checksums)
    file_vectors = _compute_file_vectors(embedding_payloads)
    if not file_vectors:
        return list(checksums)[:max_files]
    centroid = [float(val) for val in cluster.get("centroid", [])]
    if not centroid:
        centroid = compute_centroid(list(file_vectors.values()))
    if not centroid:
        return list(checksums)[:max_files]
    centroid = _l2_normalize(centroid)
    return _select_diversified_near_centroid(
        file_vectors,
        centroid,
        max_files=max_files,
    )


def _compute_file_vectors(
    embedding_payloads: dict[str, list[dict[str, Any]]],
) -> dict[str, list[float]]:
    file_vectors: dict[str, list[float]] = {}
    for checksum, payloads in embedding_payloads.items():
        vectors = [payload["vector"] for payload in payloads if payload.get("vector")]
        if not vectors:
            continue
        length = len(vectors[0])
        summed = [0.0] * length
        count = 0
        for vec in vectors:
            if len(vec) != length:
                continue
            for idx, val in enumerate(vec):
                summed[idx] += float(val)
            count += 1
        if count == 0:
            continue
        mean_vec = [val / count for val in summed]
        file_vectors[checksum] = _l2_normalize(mean_vec)
    return file_vectors


def _select_diversified_near_centroid(
    file_vectors: dict[str, list[float]],
    centroid: Sequence[float],
    *,
    max_files: int,
    diversity_threshold: float = 0.95,
) -> list[str]:
    if not file_vectors:
        return []
    scored = [
        (checksum, _cosine_similarity(centroid, vector))
        for checksum, vector in file_vectors.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    medoid = scored[0][0]
    selected = [medoid]
    remaining = [checksum for checksum, _score in scored[1:]]
    for checksum in remaining:
        if len(selected) >= max_files:
            break
        candidate_vec = file_vectors[checksum]
        max_similarity = max(
            _cosine_similarity(candidate_vec, file_vectors[chosen])
            for chosen in selected
        )
        if max_similarity < diversity_threshold:
            selected.append(checksum)
    if len(selected) >= max_files:
        return selected[:max_files]
    for checksum in remaining:
        if checksum in selected:
            continue
        selected.append(checksum)
        if len(selected) >= max_files:
            break
    return selected


def select_representative_chunks_for_files(
    files: Sequence[dict[str, Any]],
    *,
    max_chunks_per_file: int = 2,
    max_chars: int = 200,
) -> tuple[list[str], dict[str, Any]]:
    snippet_metadata: dict[str, Any] = {}
    checksums = [str(entry.get("checksum")) for entry in files if entry.get("checksum")]
    embedding_payloads = _load_chunk_embeddings(checksums)
    if not embedding_payloads:
        snippet_metadata = {
            "method": "fallback_text",
            "reason": "embeddings_unavailable",
        }
        return (
            _fallback_snippets_from_text(files, max_chunks_per_file=max_chunks_per_file, max_chars=max_chars),
            snippet_metadata,
        )

    file_vectors = _compute_file_vectors(embedding_payloads)
    centroid = compute_centroid(list(file_vectors.values()))
    if not centroid:
        snippet_metadata = {
            "method": "fallback_text",
            "reason": "empty_centroid",
        }
        return (
            _fallback_snippets_from_text(files, max_chunks_per_file=max_chunks_per_file, max_chars=max_chars),
            snippet_metadata,
        )

    centroid = _l2_normalize(centroid)
    chunk_ids = {
        payload["chunk_id"]
        for chunks in embedding_payloads.values()
        for payload in chunks
        if payload.get("chunk_id")
    }
    chunk_texts = _fetch_chunk_texts(chunk_ids)

    snippets: list[str] = []
    fallback_files: list[str] = []
    for file_entry in files:
        checksum = str(file_entry.get("checksum") or "")
        if not checksum:
            continue
        candidates = embedding_payloads.get(checksum, [])
        if not candidates:
            fallback_files.append(checksum)
            snippets.extend(
                _fallback_snippets_from_text(
                    [file_entry],
                    max_chunks_per_file=1,
                    max_chars=max_chars,
                )
            )
            continue
        best = max(
            candidates,
            key=lambda payload: _cosine_similarity(centroid, payload["vector"]),
        )
        chunk_id = best.get("chunk_id")
        text = chunk_texts.get(chunk_id) if chunk_id else None
        if not text:
            fallback_files.append(checksum)
            snippets.extend(
                _fallback_snippets_from_text(
                    [file_entry],
                    max_chunks_per_file=1,
                    max_chars=max_chars,
                )
            )
            continue
        sanitized = _scrub_text(str(text)).strip()
        if sanitized:
            snippets.append(sanitized[:max_chars])

    snippet_metadata = {
        "method": "centroid_medoid",
        "centroid_source": "file_vectors",
        "source": "qdrant",
        "files_total": len(checksums),
        "files_with_embeddings": len(embedding_payloads),
        "chunks_considered": sum(len(chunks) for chunks in embedding_payloads.values()),
        "fallback_files": fallback_files,
    }
    return snippets, snippet_metadata


def _fallback_snippets_from_text(
    files: Sequence[dict[str, Any]],
    *,
    max_chunks_per_file: int,
    max_chars: int,
) -> list[str]:
    snippets: list[str] = []
    for file_entry in files:
        text_full = file_entry.get("text_full")
        if not text_full:
            continue
        chunks_added = 0
        for chunk in _split_snippets(str(text_full)):
            sanitized = _scrub_text(chunk)
            sanitized = sanitized.strip()
            if not sanitized:
                continue
            snippets.append(sanitized[:max_chars])
            chunks_added += 1
            if chunks_added >= max_chunks_per_file:
                break
    return snippets


def _load_chunk_embeddings(
    checksums: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    if not checksums:
        return {}
    client = QdrantClient(url=QDRANT_URL)
    try:
        collections = client.get_collections().collections
        if QDRANT_COLLECTION not in [collection.name for collection in collections]:
            return {}
    except Exception:  # noqa: BLE001
        return {}
    embedding_payloads: dict[str, list[dict[str, Any]]] = {}
    for checksum in checksums:
        scroll_filter = models.Filter(
            must=[models.FieldCondition(key="checksum", match=models.MatchValue(value=checksum))]
        )
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=QDRANT_COLLECTION,
                scroll_filter=scroll_filter,
                limit=256,
                with_vectors=True,
                with_payload=True,
                offset=offset,
            )
            for point in points:
                vector = _extract_vector(point)
                if not vector:
                    continue
                payload = _extract_payload(point)
                chunk_id = payload.get("id")
                entry = {
                    "vector": _l2_normalize(vector),
                    "chunk_id": str(chunk_id) if chunk_id else None,
                }
                embedding_payloads.setdefault(checksum, []).append(entry)
            if offset is None:
                break
    return embedding_payloads


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


def _fetch_chunk_texts(chunk_ids: set[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    try:
        os_client = get_client()
        response = os_client.mget(
            index=CHUNKS_INDEX,
            body={"ids": list(chunk_ids), "_source": ["text"]},
        )
    except Exception:  # noqa: BLE001
        return {}
    texts: dict[str, str] = {}
    for doc in response.get("docs", []):
        if not doc.get("found"):
            continue
        chunk_id = doc.get("_id")
        if chunk_id:
            texts[str(chunk_id)] = doc.get("_source", {}).get("text", "")
    return texts


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(val) * float(val) for val in vec))
    if norm == 0.0:
        return [0.0 for _ in vec]
    return [float(val) / norm for val in vec]


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    if len(vec_a) != len(vec_b):
        return float("-inf")
    return sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))


def get_significant_keywords_from_os(
    checksums: Sequence[str],
    snippets: Sequence[str] | None = None,
    *,
    max_keywords: int = DEFAULT_MAX_KEYWORDS,
    max_path_depth: int | None = DEFAULT_MAX_PATH_DEPTH,
    root_path: str | None = DEFAULT_ROOT_PATH,
) -> list[str]:
    counts = _keyword_counts_from_os(
        checksums,
        snippets,
        max_keywords=max_keywords,
        max_path_depth=max_path_depth,
        root_path=root_path,
    )
    return _top_keywords_from_counts(counts, max_keywords=max_keywords)


def suggest_child_name_with_llm(
    profile: ClusterProfile,
    *,
    model_id: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    language: str = DEFAULT_LANGUAGE,
    llm_callable: Any | None = None,
    allow_cache: bool = True,
    ignore_cache: bool = False,
) -> NameSuggestion:
    timing_context = (
        timed_block(
            "step.topic_naming.cluster",
            extra={"cluster_id": profile.cluster_id, "model_id": model_id},
            logger=logger,
        )
        if _timing_verbose()
        else nullcontext()
    )
    with timing_context:
        cache_key = hash_profile(profile, prompt_version, model_id, language=language)
        if allow_cache and not ignore_cache:
            cached = _load_cached_suggestion(cache_key)
            if cached:
                logger.debug(
                    "Topic naming cache hit for cluster %s (llm_used=%s)",
                    profile.cluster_id,
                    (cached.metadata or {}).get("llm_cache", {}).get("llm_used"),
                )
                _log_naming_result(
                    profile.cluster_id,
                    "cluster",
                    cached,
                    cache_hit=True,
                    llm_state=None,
                )
                return cached
        llm_state = None
        if not llm_callable:
            llm_state, llm_status = _llm_readiness_state()
            if llm_state == "not_loaded":
                suggestion = _baseline_suggestion(
                    profile,
                    cache_key=cache_key,
                    fallback_reason="llm_model_not_loaded",
                    error_summary=llm_status.get("status_message"),
                )
                if ignore_cache and allow_cache:
                    suggestion = _apply_cache_bypass_warning(
                        suggestion,
                        llm_state=llm_state,
                    )
                _log_naming_result(
                    profile.cluster_id,
                    "cluster",
                    suggestion,
                    cache_hit=False,
                    llm_state=llm_state,
                )
                return _cache_suggestion(cache_key, suggestion)
        try:
            suggestion = _suggest_name_with_llm(
                profile,
                model_id=model_id,
                prompt_version=prompt_version,
                language=language,
                llm_callable=llm_callable,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM naming failed for cluster %s: %s", profile.cluster_id, exc)
            suggestion = _baseline_suggestion(
                profile,
                cache_key=cache_key,
                fallback_reason="other_exception",
                error_summary=str(exc),
            )
        if ignore_cache and allow_cache:
            suggestion = _apply_cache_bypass_warning(suggestion, llm_state=llm_state)
        _log_naming_result(
            profile.cluster_id,
            "cluster",
            suggestion,
            cache_hit=False,
            llm_state=llm_state,
        )
        return _cache_suggestion(cache_key, suggestion)


def suggest_parent_name_with_llm(
    profile: ParentProfile,
    *,
    model_id: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    language: str = DEFAULT_LANGUAGE,
    llm_callable: Any | None = None,
    allow_cache: bool = True,
    ignore_cache: bool = False,
) -> NameSuggestion:
    timing_context = (
        timed_block(
            "step.topic_naming.parent",
            extra={"parent_id": profile.parent_id, "model_id": model_id},
            logger=logger,
        )
        if _timing_verbose()
        else nullcontext()
    )
    with timing_context:
        cache_key = hash_profile(profile, prompt_version, model_id, language=language)
        if allow_cache and not ignore_cache:
            cached = _load_cached_suggestion(cache_key)
            if cached:
                logger.debug(
                    "Topic naming cache hit for parent %s (llm_used=%s)",
                    profile.parent_id,
                    (cached.metadata or {}).get("llm_cache", {}).get("llm_used"),
                )
                _log_naming_result(
                    profile.parent_id,
                    "parent",
                    cached,
                    cache_hit=True,
                    llm_state=None,
                )
                return cached
        llm_state = None
        if not llm_callable:
            llm_state, llm_status = _llm_readiness_state()
            if llm_state == "not_loaded":
                suggestion = _baseline_suggestion(
                    profile,
                    cache_key=cache_key,
                    fallback_reason="llm_model_not_loaded",
                    error_summary=llm_status.get("status_message"),
                )
                if ignore_cache and allow_cache:
                    suggestion = _apply_cache_bypass_warning(
                        suggestion,
                        llm_state=llm_state,
                    )
                _log_naming_result(
                    profile.parent_id,
                    "parent",
                    suggestion,
                    cache_hit=False,
                    llm_state=llm_state,
                )
                return _cache_suggestion(cache_key, suggestion)
        try:
            suggestion = _suggest_name_with_llm(
                profile,
                model_id=model_id,
                prompt_version=prompt_version,
                language=language,
                llm_callable=llm_callable,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM naming failed for parent %s: %s", profile.parent_id, exc)
            suggestion = _baseline_suggestion(
                profile,
                cache_key=cache_key,
                fallback_reason="other_exception",
                error_summary=str(exc),
            )
        if ignore_cache and allow_cache:
            suggestion = _apply_cache_bypass_warning(suggestion, llm_state=llm_state)
        _log_naming_result(
            profile.parent_id,
            "parent",
            suggestion,
            cache_hit=False,
            llm_state=llm_state,
        )
        return _cache_suggestion(cache_key, suggestion)


def postprocess_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip()
    if not cleaned:
        return "Untitled"
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return "Untitled"
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", cleaned).strip().lower()
    if normalized in {"untitled", "misc", "miscellaneous"}:
        return "Untitled"
    words = cleaned.split()
    truncated = " ".join(words[:_NAME_MAX_WORDS])
    titled = truncated.title()
    if len(titled) > _NAME_MAX_CHARS:
        titled = titled[:_NAME_MAX_CHARS].rstrip()
    return titled


def english_only_check(name: str) -> bool:
    if not re.fullmatch(r"[A-Za-z0-9 .,&()\-']+", name):
        return False
    tokens = [tok for tok in _NON_WORD_RE.split(name.lower()) if tok]
    return not any(token in _GERMAN_STOPWORDS for token in tokens)


def _format_differentiator(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    if cleaned.startswith(".") and len(cleaned) > 1:
        cleaned = cleaned[1:]
    formatted = postprocess_name(cleaned)
    if formatted == "Untitled":
        return None
    return formatted


def disambiguate_duplicate_names(
    names: Sequence[str],
    *,
    differentiators: Sequence[str | None] | None = None,
) -> list[str]:
    disambiguators: list[str | None]
    if differentiators is None:
        disambiguators = [None] * len(names)
    else:
        disambiguators = list(differentiators)
        if len(disambiguators) < len(names):
            disambiguators.extend([None] * (len(names) - len(disambiguators)))
        elif len(disambiguators) > len(names):
            disambiguators = disambiguators[: len(names)]

    indices_by_name: dict[str, list[int]] = {}
    for idx, name in enumerate(names):
        indices_by_name.setdefault(name, []).append(idx)

    unique = list(names)
    for base, indices in indices_by_name.items():
        if len(indices) == 1:
            continue
        used_names: set[str] = set()
        used_suffixes: set[str] = set()
        per_base_candidates = [
            _format_differentiator(disambiguators[idx]) for idx in indices
        ]
        candidate_iter = iter(per_base_candidates)
        first_idx = indices[0]
        unique[first_idx] = base
        used_names.add(base)
        for idx in indices[1:]:
            candidate = next(candidate_iter, None)
            if candidate and candidate not in used_suffixes:
                new_name = f"{base} ({candidate})"
                used_suffixes.add(candidate)
            else:
                counter = 2
                new_name = f"{base} ({counter})"
                while new_name in used_names:
                    counter += 1
                    new_name = f"{base} ({counter})"
            unique[idx] = new_name
            used_names.add(new_name)
    return unique


def hash_profile(
    profile: ClusterProfile | ParentProfile | dict[str, Any],
    prompt_version: str,
    model_id: str,
    *,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    if isinstance(profile, dict):
        profile_payload = profile
    elif is_dataclass(profile):
        profile_payload = asdict(profile)
    else:
        profile_payload = dict(profile)
    hash_payload = {
        "profile": profile_payload,
        "prompt_version": prompt_version,
        "model_id": model_id,
        "language": language,
    }
    blob = json.dumps(hash_payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_topic_discovery_payloads() -> dict[str, dict[str, Any]]:
    cache = load_last_cluster_cache()
    if not cache:
        return {}
    checksums = [str(checksum) for checksum in cache.get("checksums", [])]
    payloads = cache.get("payloads", []) or []
    mapping: dict[str, dict[str, Any]] = {}
    for checksum, payload in zip(checksums, payloads, strict=False):
        entry = dict(payload or {})
        entry.setdefault("checksum", checksum)
        mapping[checksum] = entry
    return mapping


def _fallback_name(keywords: Sequence[str]) -> str:
    if not keywords:
        return "General"
    return " ".join(_dedupe_keep_order(keywords)[:3]).title()


def _scrub_text(text: str) -> str:
    text = _EMAIL_RE.sub(" ", text)
    text = _LONG_NUMBER_RE.sub(" ", text)
    text = _TOKEN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_snippets(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _tokenize_text(text: str) -> list[str]:
    tokens = [tok for tok in _NON_WORD_RE.split(text.lower()) if tok]
    return [tok for tok in tokens if tok not in _STOPWORDS]


def _extract_representative_paths(files: Sequence[dict[str, Any]]) -> list[str]:
    paths = [
        str(entry.get("path"))
        for entry in files
        if entry.get("path")
    ]
    return _dedupe_keep_order(paths)


def _extract_extension(entry: dict[str, Any]) -> str | None:
    path_value = entry.get("path") or entry.get("filename") or ""
    suffix = Path(str(path_value)).suffix.lower()
    if suffix:
        return suffix
    filetype = entry.get("filetype")
    if not filetype:
        return None
    filetype_str = str(filetype).lower().strip()
    if not filetype_str:
        return None
    if not filetype_str.startswith("."):
        return f".{filetype_str}"
    return filetype_str


def _format_top_extensions(
    counts: dict[str, int],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"extension": ext, "count": count}
        for ext, count in ranked[:limit]
    ]


def _extension_counts(
    files: Sequence[dict[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in files:
        ext = _extract_extension(entry)
        if not ext:
            continue
        counts[ext] = counts.get(ext, 0) + 1
    return counts


def _safe_fetch_fulltext(checksum: str) -> dict[str, Any] | None:
    try:
        return get_fulltext_by_checksum(checksum)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OpenSearch lookup failed for checksum=%s: %s", checksum, exc)
        return None


def _weighted_avg(values: Sequence[float], weights: Sequence[int]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        return float(sum(values) / len(values)) if values else 0.0
    return float(sum(val * weight for val, weight in zip(values, weights, strict=False)) / total_weight)


def _dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _format_fallback_warning(fallback_reason: str | None) -> str:
    if not fallback_reason:
        return ""
    if fallback_reason.startswith("llm_error_http_"):
        code = fallback_reason.split("_")[-1]
        return f"LLM request failed (HTTP {code}) (used baseline)"
    return FALLBACK_REASON_MESSAGES.get(
        fallback_reason, FALLBACK_REASON_MESSAGES["other_exception"]
    )


def _apply_cache_bypass_warning(
    suggestion: NameSuggestion,
    *,
    llm_state: str | None,
) -> NameSuggestion:
    if llm_state == "not_loaded" and suggestion.source == "baseline":
        return _with_warning(
            suggestion,
            LLM_UNAVAILABLE_CACHE_BYPASS_WARNING,
            replace=True,
        )
    return _with_warning(suggestion, CACHE_BYPASS_WARNING)


def _with_warning(
    suggestion: NameSuggestion,
    warning: str,
    *,
    replace: bool = False,
) -> NameSuggestion:
    if not warning:
        return suggestion
    metadata = dict(suggestion.metadata or {})
    existing = metadata.get("warning")
    if replace or not existing:
        metadata["warning"] = warning
    else:
        metadata["warning"] = f"{warning}; {existing}"
    return NameSuggestion(
        name=suggestion.name,
        confidence=suggestion.confidence,
        source=suggestion.source,
        cache_key=suggestion.cache_key,
        metadata=metadata,
    )


def _map_llm_error_to_reason(error_type: str, status_code: int | None) -> str:
    if error_type == "timeout":
        return "llm_timeout"
    if error_type == "http_error":
        return f"llm_error_http_{status_code or 'unknown'}"
    if error_type == "invalid_json":
        return "llm_invalid_json"
    if error_type == "request_exception":
        return "llm_unreachable"
    if error_type == "empty_response":
        return "other_exception"
    return "other_exception"


def _llm_readiness_state() -> tuple[str, dict[str, Any]]:
    status = check_llm_status()
    if status.get("server_online") and not status.get("model_loaded"):
        return "not_loaded", status
    if status.get("active"):
        return "ready", status
    return "unknown", status


def _load_cached_suggestion(cache_key: str) -> NameSuggestion | None:
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        suggestion = NameSuggestion(**payload)
        return _mark_cache_hit(suggestion)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load topic naming cache %s: %s", cache_path, exc)
        return None


def _cache_suggestion(cache_key: str, suggestion: NameSuggestion) -> NameSuggestion:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.json"
    payload = asdict(suggestion)
    payload["cache_key"] = cache_key
    try:
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write topic naming cache %s: %s", cache_path, exc)
    return NameSuggestion(**payload)


def _mark_cache_hit(suggestion: NameSuggestion) -> NameSuggestion:
    metadata = dict(suggestion.metadata or {})
    llm_cache = dict(metadata.get("llm_cache", {}))
    llm_used = bool(llm_cache.get("llm_used", suggestion.source == "llm"))
    llm_cache["llm_used"] = llm_used
    llm_cache["cache_hit"] = True
    if not llm_used:
        llm_cache["fallback_reason"] = "cache_hit_baseline"
        metadata["warning"] = _format_fallback_warning("cache_hit_baseline")
    else:
        llm_cache["fallback_reason"] = None
        metadata.pop("warning", None)
    metadata["llm_cache"] = llm_cache
    return NameSuggestion(
        name=suggestion.name,
        confidence=suggestion.confidence,
        source=suggestion.source,
        cache_key=suggestion.cache_key,
        metadata=metadata,
    )


def _log_naming_result(
    identifier: int,
    level: str,
    suggestion: NameSuggestion,
    *,
    cache_hit: bool,
    llm_state: str | None,
) -> None:
    llm_cache = (suggestion.metadata or {}).get("llm_cache", {})
    logger.debug(
        "Topic naming result for %s %s: cache_hit=%s llm_state=%s llm_used=%s fallback_reason=%s error=%s",
        level,
        identifier,
        cache_hit,
        llm_state,
        llm_cache.get("llm_used"),
        llm_cache.get("fallback_reason"),
        llm_cache.get("error_summary"),
    )


def _load_cached_profiles(cache_key: str) -> dict[str, Any] | None:
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load topic naming cache %s: %s", cache_path, exc)
        return None


def _cache_profiles(cache_key: str, payload: dict[str, Any]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.json"
    try:
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write topic naming cache %s: %s", cache_path, exc)


def _suggest_name_with_llm(
    profile: ClusterProfile | ParentProfile,
    *,
    model_id: str,
    prompt_version: str,
    language: str,
    llm_callable: Any | None,
) -> NameSuggestion:
    if llm_callable:
        result = llm_callable(profile)
        if isinstance(result, NameSuggestion):
            return _ensure_llm_cache(result, profile=profile, used=True)
        if isinstance(result, str) and result.strip():
            return _suggestion_from_text(
                profile,
                result,
                cache_key=None,
                used=True,
                prompt_chars=None,
            )
    prompt = _build_llm_prompt(profile, prompt_version=prompt_version, language=language)
    response = ask_llm_with_status(
        prompt,
        mode="completion",
        model=model_id,
        max_tokens=24,
        temperature=0.2,
    )
    if response["error"]:
        fallback_reason = _map_llm_error_to_reason(
            response["error"]["type"], response["error"]["status_code"]
        )
        return _baseline_suggestion(
            profile,
            cache_key=None,
            fallback_reason=fallback_reason,
            error_summary=response["error"]["summary"],
            prompt_chars=response["prompt_length"],
        )
    if not response["content"]:
        return _baseline_suggestion(
            profile,
            cache_key=None,
            fallback_reason="other_exception",
            error_summary="Empty LLM response",
            prompt_chars=response["prompt_length"],
        )
    cleaned = postprocess_name(response["content"])
    if language == "en" and not english_only_check(cleaned):
        retry_prompt = (
            f"{prompt}\nEnglish only; no German words."
        )
        retry_response = ask_llm_with_status(
            retry_prompt,
            mode="completion",
            model=model_id,
            max_tokens=24,
            temperature=0.2,
        )
        if retry_response["error"]:
            fallback_reason = _map_llm_error_to_reason(
                retry_response["error"]["type"], retry_response["error"]["status_code"]
            )
            fallback = _baseline_suggestion(
                profile,
                cache_key=None,
                fallback_reason=fallback_reason,
                error_summary=retry_response["error"]["summary"],
                prompt_chars=retry_response["prompt_length"],
            )
            return _with_llm_cache(
                fallback,
                {
                    "retry_attempted": True,
                    "retry_success": False,
                    "retry_reason": "llm_error",
                },
            )
        if not retry_response["content"]:
            fallback = _baseline_suggestion(
                profile,
                cache_key=None,
                fallback_reason="other_exception",
                error_summary="Empty LLM response",
                prompt_chars=retry_response["prompt_length"],
            )
            return _with_llm_cache(
                fallback,
                {
                    "retry_attempted": True,
                    "retry_success": False,
                    "retry_reason": "empty_response",
                },
            )
        cleaned_retry = postprocess_name(retry_response["content"])
        if not english_only_check(cleaned_retry):
            fallback = _baseline_suggestion(
                profile,
                cache_key=None,
                fallback_reason="other_exception",
                error_summary="Non-English response",
                prompt_chars=retry_response["prompt_length"],
            )
            return _with_llm_cache(
                fallback,
                {
                    "retry_attempted": True,
                    "retry_success": False,
                    "retry_reason": "non_english",
                },
            )
        suggestion = _suggestion_from_text(
            profile,
            cleaned_retry,
            cache_key=None,
            used=True,
            prompt_chars=retry_response["prompt_length"],
        )
        return _with_llm_cache(
            suggestion,
            {
                "retry_attempted": True,
                "retry_success": True,
            },
        )
    return _suggestion_from_text(
        profile,
        cleaned,
        cache_key=None,
        used=True,
        prompt_chars=response["prompt_length"],
    )


def _build_llm_prompt(
    profile: ClusterProfile | ParentProfile,
    *,
    prompt_version: str,
    language: str,
) -> str:
    keywords = ", ".join(profile.keywords[:12])
    if isinstance(profile, ClusterProfile):
        file_names = ", ".join(
            _dedupe_keep_order(
                [
                    str(entry.get("filename") or entry.get("path") or "")
                    for entry in profile.representative_files
                ]
            )
        )
        representative_paths = ", ".join(_dedupe_keep_order(profile.representative_paths)[:8])
        top_extensions = ", ".join(
            f"{entry.get('extension')} ({entry.get('count')})"
            for entry in profile.top_extensions
            if entry.get("extension")
        )
        snippets = "\n".join(profile.representative_snippets[:6])
        return (
            f"Prompt version: {prompt_version}\n"
            f"Language: {language}\n"
            "You label document clusters with concise English names (2-6 words).\n"
            f"Cluster size: {profile.size}\n"
            f"Keywords: {keywords}\n"
            f"Representative files: {file_names}\n"
            f"Representative paths: {representative_paths}\n"
            f"Top extensions: {top_extensions}\n"
            f"Snippets:\n{snippets}\n"
            "Return only the name."
        )
    top_extensions = ", ".join(
        f"{entry.get('extension')} ({entry.get('count')})"
        for entry in profile.top_extensions
        if entry.get("extension")
    )
    return (
        f"Prompt version: {prompt_version}\n"
        f"Language: {language}\n"
        "You label parent topic groups with concise English names (2-6 words).\n"
        f"Child clusters: {', '.join(str(cid) for cid in profile.cluster_ids)}\n"
        f"Keywords: {keywords}\n"
        f"Top extensions: {top_extensions}\n"
        "Return only the name."
    )


def _suggestion_from_text(
    profile: ClusterProfile | ParentProfile,
    name: str,
    *,
    cache_key: str | None,
    used: bool,
    prompt_chars: int | None,
) -> NameSuggestion:
    metadata = _profile_metadata(profile)
    metadata["llm_cache"] = {
        "llm_used": used,
        "fallback_reason": None,
        "error_summary": None,
        "cache_hit": False,
        "prompt_chars": prompt_chars,
    }
    return NameSuggestion(
        name=postprocess_name(name),
        confidence=None,
        source="llm" if used else "baseline",
        cache_key=cache_key,
        metadata=metadata,
    )


def _with_llm_cache(suggestion: NameSuggestion, updates: dict[str, Any]) -> NameSuggestion:
    metadata = dict(suggestion.metadata or {})
    llm_cache = dict(metadata.get("llm_cache", {}))
    llm_cache.update(updates)
    metadata["llm_cache"] = llm_cache
    return NameSuggestion(
        name=suggestion.name,
        confidence=suggestion.confidence,
        source=suggestion.source,
        cache_key=suggestion.cache_key,
        metadata=metadata,
    )


def _baseline_suggestion(
    profile: ClusterProfile | ParentProfile,
    *,
    cache_key: str | None,
    fallback_reason: str,
    error_summary: str | None,
    prompt_chars: int | None = None,
) -> NameSuggestion:
    metadata = _profile_metadata(profile)
    metadata["warning"] = _format_fallback_warning(fallback_reason)
    metadata["llm_cache"] = {
        "llm_used": False,
        "fallback_reason": fallback_reason,
        "error_summary": error_summary,
        "cache_hit": False,
        "prompt_chars": prompt_chars,
    }
    name = _baseline_name(profile)
    return NameSuggestion(
        name=postprocess_name(name),
        confidence=None,
        source="baseline",
        cache_key=cache_key,
        metadata=metadata,
    )


def _profile_metadata(profile: ClusterProfile | ParentProfile) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "keywords": list(profile.keywords),
        "mixedness": profile.mixedness,
        "mixedness_components": {
            "keyword_entropy": profile.keyword_entropy,
            "extension_entropy": profile.extension_entropy,
            "embedding_spread": profile.embedding_spread,
        },
    }
    if isinstance(profile, ClusterProfile):
        metadata["cluster_id"] = profile.cluster_id
        metadata["size"] = profile.size
        metadata["representative_paths"] = list(profile.representative_paths)
        metadata["snippet_selection"] = dict(profile.representative_snippet_metadata)
        metadata["top_extensions"] = list(profile.top_extensions)
    else:
        metadata["parent_id"] = profile.parent_id
        metadata["size"] = profile.size
        metadata["cluster_ids"] = list(profile.cluster_ids)
        metadata["top_extensions"] = list(profile.top_extensions)
    return metadata


def _significant_terms_from_os(
    checksums: Sequence[str],
    *,
    max_keywords: int,
) -> dict[str, float]:
    if not checksums:
        return {}
    size = max(15, min(max_keywords, 30))
    body = {
        "size": 0,
        "query": {"bool": {"filter": [{"terms": {"checksum": list(checksums)}}]}},
        "aggs": {
            "significant_terms": {
                "significant_terms": {
                    "field": "text_full",
                    "size": size,
                    "min_doc_count": 1,
                }
            }
        },
    }
    try:
        client = get_client()
        with timed_block(
            "step.opensearch.significant_terms",
            extra={"index": FULLTEXT_INDEX, "operation": "significant_terms"},
            logger=logger,
        ):
            response = client.search(index=FULLTEXT_INDEX, body=body)
    except Exception:  # noqa: BLE001
        return {}
    buckets = (
        response.get("aggregations", {})
        .get("significant_terms", {})
        .get("buckets", [])
    )
    results: dict[str, float] = {}
    for bucket in buckets:
        key = bucket.get("key")
        if not key:
            continue
        token = str(key).lower().strip()
        if not token or token in _STOPWORDS or len(token) < 3:
            continue
        score = bucket.get("score")
        if score is None:
            score = bucket.get("doc_count", 0)
        results[token] = float(score or 0)
    return results


def _keyword_counts_from_os(
    checksums: Sequence[str],
    snippets: Sequence[str] | None,
    *,
    max_keywords: int,
    max_path_depth: int | None,
    root_path: str | None,
) -> dict[str, float]:
    significant_terms = _significant_terms_from_os(
        checksums,
        max_keywords=max_keywords,
    )
    if significant_terms:
        return significant_terms
    tokens: list[str] = []
    for checksum in checksums:
        fulltext = _safe_fetch_fulltext(checksum)
        if not fulltext:
            continue
        path = fulltext.get("path") or ""
        filename = fulltext.get("filename") or ""
        tokens.extend(
            extract_path_segments(
                path,
                max_depth=max_path_depth,
                root_path=root_path,
            )
        )
        tokens.extend(tokenize_filename(filename))
    if snippets:
        for snippet in snippets:
            tokens.extend(_tokenize_text(snippet))
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for token in tokens:
        if token in _STOPWORDS or len(token) < 3:
            continue
        counts[token] = counts.get(token, 0) + 1
    if not counts:
        return {}
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    limited = ranked[:max_keywords]
    return {token: float(count) for token, count in limited}


def _top_keywords_from_counts(counts: dict[str, float], *, max_keywords: int) -> list[str]:
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:max_keywords]]


def _keyword_mixedness(counts: dict[str, float], *, max_keywords: int) -> float:
    if not counts:
        return 0.0
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top = ranked[:max_keywords]
    total = sum(count for _, count in top)
    if total <= 0 or len(top) <= 1:
        return 0.0
    entropy = 0.0
    for _, count in top:
        prob = count / total
        if prob <= 0:
            continue
        entropy -= prob * math.log(prob)
    normalizer = math.log(len(top))
    if normalizer <= 0:
        return 0.0
    return float(entropy / normalizer)


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized_entropy(counts: dict[str, float] | dict[str, int]) -> float:
    if not counts:
        return 0.0
    total = sum(float(value) for value in counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        prob = float(value) / total
        if prob <= 0:
            continue
        entropy -= prob * math.log(prob)
    normalizer = math.log(len(counts))
    if normalizer <= 0:
        return 0.0
    return _clamp_unit(entropy / normalizer)


def _combined_mixedness(
    keyword_entropy: float,
    extension_entropy: float,
    embedding_spread: float,
) -> float:
    weighted_sum = (
        keyword_entropy * MIXEDNESS_COMPONENT_WEIGHTS["keyword_entropy"]
        + extension_entropy * MIXEDNESS_COMPONENT_WEIGHTS["extension_entropy"]
        + embedding_spread * MIXEDNESS_COMPONENT_WEIGHTS["embedding_spread"]
    )
    weight_total = sum(MIXEDNESS_COMPONENT_WEIGHTS.values()) or 1.0
    combined = weighted_sum / weight_total
    return _clamp_unit(combined * MIXEDNESS_RANGE_SCALE)


def _embedding_spread(
    representative_checksums: Sequence[str],
    *,
    avg_prob: float,
) -> float:
    embedding_payloads = _load_chunk_embeddings(representative_checksums)
    if not embedding_payloads:
        return _clamp_unit(1.0 - avg_prob)
    file_vectors = _compute_file_vectors(embedding_payloads)
    if not file_vectors:
        return _clamp_unit(1.0 - avg_prob)
    centroid = compute_centroid(list(file_vectors.values()))
    if not centroid:
        return _clamp_unit(1.0 - avg_prob)
    centroid = _l2_normalize(centroid)
    similarities = [
        _cosine_similarity(centroid, vector) for vector in file_vectors.values()
    ]
    if not similarities:
        return _clamp_unit(1.0 - avg_prob)
    avg_similarity = sum(similarities) / len(similarities)
    spread = (1.0 - avg_similarity) / 2.0
    return _clamp_unit(spread)


def _baseline_name(profile: ClusterProfile | ParentProfile) -> str:
    name = _fallback_name(profile.keywords)
    if name != "General":
        return name
    if isinstance(profile, ClusterProfile):
        return f"Cluster {profile.cluster_id}"
    return f"Group {profile.parent_id}"


def _ensure_llm_cache(
    suggestion: NameSuggestion,
    *,
    profile: ClusterProfile | ParentProfile,
    used: bool,
) -> NameSuggestion:
    metadata = dict(suggestion.metadata or {})
    metadata = {**_profile_metadata(profile), **metadata}
    llm_cache = dict(metadata.get("llm_cache", {}))
    llm_cache.setdefault("llm_used", used)
    llm_cache.setdefault("fallback_reason", None)
    llm_cache.setdefault("error_summary", None)
    llm_cache.setdefault("cache_hit", False)
    llm_cache.setdefault("prompt_chars", None)
    if used:
        llm_cache["llm_used"] = True
        metadata.pop("warning", None)
    metadata["llm_cache"] = llm_cache
    return NameSuggestion(
        name=suggestion.name,
        confidence=suggestion.confidence,
        source=suggestion.source,
        cache_key=suggestion.cache_key,
        metadata=metadata,
    )


__all__ = [
    "ClusterProfile",
    "ParentProfile",
    "NameSuggestion",
    "tokenize_filename",
    "extract_path_segments",
    "build_cluster_profile",
    "build_parent_profile",
    "compute_centroid",
    "select_representative_files",
    "select_representative_chunks_for_files",
    "get_significant_keywords_from_os",
    "suggest_child_name_with_llm",
    "suggest_parent_name_with_llm",
    "postprocess_name",
    "english_only_check",
    "disambiguate_duplicate_names",
    "hash_profile",
    "load_topic_discovery_payloads",
]
