from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from config import logger
from services.topic_discovery_clusters import load_last_cluster_cache
from utils.opensearch.fulltext import get_fulltext_by_checksum


CACHE_DIR = Path(".cache") / "topic_naming"
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_LANGUAGE = "en"

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
    representative_checksums: list[str] = field(default_factory=list)
    representative_files: list[dict[str, Any]] = field(default_factory=list)
    representative_snippets: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParentProfile:
    parent_id: int
    cluster_ids: list[int]
    size: int
    avg_prob: float
    centroid: list[float]
    representative_checksums: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


def tokenize_filename(filename: str) -> list[str]:
    base = os.path.splitext(os.path.basename(filename))[0]
    tokens = [tok for tok in _NON_WORD_RE.split(base.lower()) if tok]
    return [tok for tok in tokens if tok not in _STOPWORDS]


def extract_path_segments(path: str) -> list[str]:
    parts = [part for part in Path(path).parts if part not in ("/", "")]
    segments: list[str] = []
    for part in parts:
        segments.extend(tokenize_filename(part))
    return segments


def build_cluster_profile(
    cluster: dict[str, Any],
    checksum_payloads: dict[str, dict[str, Any]],
) -> ClusterProfile:
    representative_checksums = [
        str(checksum) for checksum in cluster.get("representative_checksums", [])
    ]
    representative_files = select_representative_files(
        cluster,
        checksum_payloads,
    )
    snippets = select_representative_chunks_for_files(representative_files)
    keywords = get_significant_keywords_from_os(representative_checksums, snippets)
    centroid = [float(val) for val in cluster.get("centroid", [])]
    return ClusterProfile(
        cluster_id=int(cluster.get("cluster_id", -1)),
        size=int(cluster.get("size", 0)),
        avg_prob=float(cluster.get("avg_prob", 0.0)),
        centroid=centroid,
        representative_checksums=representative_checksums,
        representative_files=representative_files,
        representative_snippets=snippets,
        keywords=keywords,
    )


def build_parent_profile(
    parent_id: int,
    child_profiles: Sequence[ClusterProfile],
) -> ParentProfile:
    cluster_ids = [profile.cluster_id for profile in child_profiles]
    size = sum(profile.size for profile in child_profiles)
    avg_prob = _weighted_avg(
        [profile.avg_prob for profile in child_profiles],
        [profile.size for profile in child_profiles],
    )
    centroid = compute_centroid([profile.centroid for profile in child_profiles])
    representative_checksums: list[str] = []
    keywords: list[str] = []
    for profile in child_profiles:
        representative_checksums.extend(profile.representative_checksums)
        keywords.extend(profile.keywords)
    return ParentProfile(
        parent_id=parent_id,
        cluster_ids=cluster_ids,
        size=size,
        avg_prob=avg_prob,
        centroid=centroid,
        representative_checksums=_dedupe_keep_order(representative_checksums),
        keywords=_dedupe_keep_order(keywords),
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
    representative_files: list[dict[str, Any]] = []
    for checksum in checksums[:max_files]:
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


def select_representative_chunks_for_files(
    files: Sequence[dict[str, Any]],
    *,
    max_chunks_per_file: int = 2,
    max_chars: int = 200,
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


def get_significant_keywords_from_os(
    checksums: Sequence[str],
    snippets: Sequence[str] | None = None,
    *,
    max_keywords: int = 8,
) -> list[str]:
    tokens: list[str] = []
    for checksum in checksums:
        fulltext = _safe_fetch_fulltext(checksum)
        if not fulltext:
            continue
        path = fulltext.get("path") or ""
        filename = fulltext.get("filename") or ""
        tokens.extend(extract_path_segments(path))
        tokens.extend(tokenize_filename(filename))
    if snippets:
        for snippet in snippets:
            tokens.extend(_tokenize_text(snippet))
    if not tokens:
        return []
    counts: dict[str, int] = {}
    for token in tokens:
        if token in _STOPWORDS or len(token) < 3:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:max_keywords]]


def suggest_child_name_with_llm(
    profile: ClusterProfile,
    *,
    model_id: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    language: str = DEFAULT_LANGUAGE,
    llm_callable: Any | None = None,
    allow_cache: bool = True,
) -> NameSuggestion:
    cache_key = hash_profile(profile, prompt_version, model_id, language=language)
    if allow_cache:
        cached = _load_cached_suggestion(cache_key)
        if cached:
            return cached
    if llm_callable:
        suggestion = llm_callable(profile)
        if isinstance(suggestion, NameSuggestion):
            return _cache_suggestion(cache_key, suggestion)
    name = _fallback_name(profile.keywords)
    suggestion = NameSuggestion(
        name=postprocess_name(name),
        confidence=None,
        source="heuristic",
        cache_key=cache_key,
        metadata={"cluster_id": profile.cluster_id},
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
) -> NameSuggestion:
    cache_key = hash_profile(profile, prompt_version, model_id, language=language)
    if allow_cache:
        cached = _load_cached_suggestion(cache_key)
        if cached:
            return cached
    if llm_callable:
        suggestion = llm_callable(profile)
        if isinstance(suggestion, NameSuggestion):
            return _cache_suggestion(cache_key, suggestion)
    name = _fallback_name(profile.keywords)
    suggestion = NameSuggestion(
        name=postprocess_name(name),
        confidence=None,
        source="heuristic",
        cache_key=cache_key,
        metadata={"parent_id": profile.parent_id},
    )
    return _cache_suggestion(cache_key, suggestion)


def postprocess_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip()
    if not cleaned:
        return "Untitled"
    return cleaned


def english_only_check(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9 .,&()\-']+", name))


def disambiguate_duplicate_names(names: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        base = name
        count = seen.get(base, 0) + 1
        seen[base] = count
        if count == 1:
            unique.append(base)
        else:
            unique.append(f"{base} ({count})")
    return unique


def hash_profile(
    profile: ClusterProfile | ParentProfile | dict[str, Any],
    prompt_version: str,
    model_id: str,
    *,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    if hasattr(profile, "__dataclass_fields__"):
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


def _load_cached_suggestion(cache_key: str) -> NameSuggestion | None:
    cache_path = CACHE_DIR / f"{cache_key}.json"
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return NameSuggestion(**payload)
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
