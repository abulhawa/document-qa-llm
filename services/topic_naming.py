from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from config import logger
from core.llm import ask_llm, check_llm_status
from services.topic_discovery_clusters import load_last_cluster_cache
from utils.opensearch.fulltext import get_fulltext_by_checksum


CACHE_DIR = Path(".cache") / "topic_naming"
DEFAULT_PROMPT_VERSION = "v1"
DEFAULT_LANGUAGE = "en"
DEFAULT_MAX_KEYWORDS = 20
DEFAULT_MAX_PATH_DEPTH = 4
DEFAULT_ROOT_PATH = ""
DEFAULT_TOP_EXTENSION_COUNT = 5
LLM_UNAVAILABLE_WARNING = (
    "LLM unavailable (model not loaded). Naming skipped or using baseline names."
)

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
    mixedness: float = 0.0
    representative_checksums: list[str] = field(default_factory=list)
    representative_files: list[dict[str, Any]] = field(default_factory=list)
    representative_paths: list[str] = field(default_factory=list)
    representative_snippets: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    top_extensions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ParentProfile:
    parent_id: int
    cluster_ids: list[int]
    size: int
    avg_prob: float
    centroid: list[float]
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
    snippets = select_representative_chunks_for_files(representative_files)
    keyword_counts = _keyword_counts_from_os(
        representative_checksums,
        snippets,
        max_keywords=max_keywords,
        max_path_depth=max_path_depth,
        root_path=root_path,
    )
    keywords = _top_keywords_from_counts(keyword_counts, max_keywords=max_keywords)
    mixedness = _keyword_mixedness(keyword_counts, max_keywords=max_keywords)
    top_extensions = _top_file_extensions(
        representative_files,
        limit=top_extension_count,
    )
    centroid = [float(val) for val in cluster.get("centroid", [])]
    return ClusterProfile(
        cluster_id=int(cluster.get("cluster_id", -1)),
        size=int(cluster.get("size", 0)),
        avg_prob=float(cluster.get("avg_prob", 0.0)),
        centroid=centroid,
        mixedness=mixedness,
        representative_checksums=representative_checksums,
        representative_files=representative_files,
        representative_paths=representative_paths,
        representative_snippets=snippets,
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
    mixedness = _weighted_avg(
        [profile.mixedness for profile in child_profiles],
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
    return ParentProfile(
        parent_id=parent_id,
        cluster_ids=cluster_ids,
        size=size,
        avg_prob=avg_prob,
        centroid=centroid,
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
) -> NameSuggestion:
    cache_key = hash_profile(profile, prompt_version, model_id, language=language)
    if allow_cache:
        cached = _load_cached_suggestion(cache_key)
        if cached:
            return cached
    if not llm_callable and not _is_llm_ready():
        suggestion = _baseline_suggestion(
            profile,
            cache_key=cache_key,
            warning=LLM_UNAVAILABLE_WARNING,
            reason="llm_unavailable",
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
            warning=LLM_UNAVAILABLE_WARNING,
            reason="llm_exception",
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
    if not llm_callable and not _is_llm_ready():
        suggestion = _baseline_suggestion(
            profile,
            cache_key=cache_key,
            warning=LLM_UNAVAILABLE_WARNING,
            reason="llm_unavailable",
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
            warning=LLM_UNAVAILABLE_WARNING,
            reason="llm_exception",
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


def _top_file_extensions(
    files: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for entry in files:
        ext = _extract_extension(entry)
        if not ext:
            continue
        counts[ext] = counts.get(ext, 0) + 1
    return _format_top_extensions(counts, limit=limit)


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


def _is_llm_ready() -> bool:
    status = check_llm_status()
    if not status.get("active"):
        logger.warning(LLM_UNAVAILABLE_WARNING)
        return False
    return True


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
            return _suggestion_from_text(profile, result, cache_key=None, used=True)
    prompt = _build_llm_prompt(profile, prompt_version=prompt_version, language=language)
    response = ask_llm(
        prompt,
        mode="completion",
        model=model_id,
        max_tokens=24,
        temperature=0.2,
    )
    if not response or response.strip() == "[LLM Error]":
        return _baseline_suggestion(
            profile,
            cache_key=None,
            warning=LLM_UNAVAILABLE_WARNING,
            reason="llm_error",
        )
    cleaned = postprocess_name(response)
    if language == "en" and not english_only_check(cleaned):
        retry_prompt = (
            f"{prompt}\nEnglish only; no German words."
        )
        retry_response = ask_llm(
            retry_prompt,
            mode="completion",
            model=model_id,
            max_tokens=24,
            temperature=0.2,
        )
        if not retry_response or retry_response.strip() == "[LLM Error]":
            fallback = _baseline_suggestion(
                profile,
                cache_key=None,
                warning=LLM_UNAVAILABLE_WARNING,
                reason="llm_error",
            )
            return _with_llm_cache(
                fallback,
                {
                    "retry_attempted": True,
                    "retry_success": False,
                    "retry_reason": "llm_error",
                },
            )
        cleaned_retry = postprocess_name(retry_response)
        if not english_only_check(cleaned_retry):
            fallback = _baseline_suggestion(
                profile,
                cache_key=None,
                warning=LLM_UNAVAILABLE_WARNING,
                reason="llm_non_english",
            )
            return _with_llm_cache(
                fallback,
                {
                    "retry_attempted": True,
                    "retry_success": False,
                    "retry_reason": "non_english",
                },
            )
        suggestion = _suggestion_from_text(profile, cleaned_retry, cache_key=None, used=True)
        return _with_llm_cache(
            suggestion,
            {
                "retry_attempted": True,
                "retry_success": True,
            },
        )
    return _suggestion_from_text(profile, cleaned, cache_key=None, used=True)


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
) -> NameSuggestion:
    metadata = _profile_metadata(profile)
    metadata["llm_cache"] = {"llm_used": used}
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
    warning: str,
    reason: str,
) -> NameSuggestion:
    metadata = _profile_metadata(profile)
    metadata["warning"] = warning
    metadata["llm_cache"] = {"llm_used": False, "reason": reason}
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
    }
    if isinstance(profile, ClusterProfile):
        metadata["cluster_id"] = profile.cluster_id
        metadata["size"] = profile.size
        metadata["representative_paths"] = list(profile.representative_paths)
        metadata["top_extensions"] = list(profile.top_extensions)
    else:
        metadata["parent_id"] = profile.parent_id
        metadata["size"] = profile.size
        metadata["cluster_ids"] = list(profile.cluster_ids)
        metadata["top_extensions"] = list(profile.top_extensions)
    return metadata


def _keyword_counts_from_os(
    checksums: Sequence[str],
    snippets: Sequence[str] | None,
    *,
    max_keywords: int,
    max_path_depth: int | None,
    root_path: str | None,
) -> dict[str, int]:
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
    return {token: count for token, count in limited}


def _top_keywords_from_counts(counts: dict[str, int], *, max_keywords: int) -> list[str]:
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [token for token, _ in ranked[:max_keywords]]


def _keyword_mixedness(counts: dict[str, int], *, max_keywords: int) -> float:
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
    metadata.setdefault("llm_cache", {"llm_used": used})
    if used:
        metadata["llm_cache"]["llm_used"] = True
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
