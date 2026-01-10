from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from opensearchpy import exceptions

from config import (
    LLM_CACHE_BACKEND,
    LLM_CACHE_ENABLED,
    LLM_CACHE_INDEX,
    LLM_CACHE_STORE_PROMPT_TEXT,
    LLM_CACHE_TTL_DAYS,
    logger,
)
from core.opensearch_client import get_client

LLM_CACHE_INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "cache_key": {"type": "keyword"},
            "created_at": {"type": "date"},
            "last_access_at": {"type": "date"},
            "hit_count": {"type": "integer"},
            "model_id": {"type": "keyword"},
            "endpoint_id": {"type": "keyword"},
            "request_fingerprint": {"type": "keyword"},
            "request_params": {"type": "object", "enabled": False},
            "prompt_hash": {"type": "keyword"},
            "prompt_text": {"type": "text"},
            "response_text": {"type": "text"},
            "response_hash": {"type": "keyword"},
            "status": {"type": "keyword"},
            "expires_at": {"type": "date"},
        }
    }
}

_cache_unavailable = False
_warned_unavailable = False
_index_ready = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _warn_unavailable(exc: Exception) -> None:
    global _cache_unavailable, _warned_unavailable
    _cache_unavailable = True
    if not _warned_unavailable:
        logger.warning("LLM cache unavailable; proceeding without cache: %s", exc)
        _warned_unavailable = True


def is_cache_enabled(use_cache: bool) -> bool:
    return (
        use_cache
        and LLM_CACHE_ENABLED
        and LLM_CACHE_BACKEND == "opensearch"
        and not _cache_unavailable
    )


def ensure_cache_index() -> bool:
    global _index_ready
    if _cache_unavailable:
        return False
    if _index_ready:
        return True
    try:
        client = get_client()
        if not client.indices.exists(index=LLM_CACHE_INDEX):
            logger.info("Creating OpenSearch index: %s", LLM_CACHE_INDEX)
            client.indices.create(
                index=LLM_CACHE_INDEX,
                body=LLM_CACHE_INDEX_SETTINGS,
                params={"wait_for_active_shards": "1"},
            )
        _index_ready = True
        return True
    except exceptions.OpenSearchException as exc:
        _warn_unavailable(exc)
    except Exception as exc:  # noqa: BLE001
        _warn_unavailable(exc)
    return False


def build_cache_key(
    *,
    prompt: str | List[Dict[str, str]],
    mode: str,
    model_id: str,
    endpoint_id: str,
    decoding_params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any], str, str, str | None]:
    system_prompt = None
    if isinstance(prompt, list):
        system_prompt = "\n".join(
            [msg.get("content", "") for msg in prompt if msg.get("role") == "system"]
        ).strip() or None
        user_prompt = "\n".join(
            [msg.get("content", "") for msg in prompt if msg.get("role") == "user"]
        )
    else:
        user_prompt = prompt

    canonical: Dict[str, Any] = {
        "model_id": model_id,
        "endpoint_id": endpoint_id,
        "decoding_params": decoding_params,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "mode": mode,
    }
    if isinstance(prompt, list):
        canonical["chat_messages"] = prompt
    else:
        canonical["prompt"] = prompt

    serialized = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    cache_key = _hash_text(serialized)
    prompt_hash = _hash_text(user_prompt)
    return cache_key, canonical, user_prompt, prompt_hash, system_prompt


def build_request_params(canonical: Dict[str, Any], prompt_hash: str) -> Dict[str, Any]:
    if LLM_CACHE_STORE_PROMPT_TEXT:
        return canonical
    sanitized = {
        k: v
        for k, v in canonical.items()
        if k not in {"prompt", "chat_messages", "user_prompt", "system_prompt"}
    }
    sanitized["prompt_hash"] = prompt_hash
    return sanitized


def get_cached_response(cache_key: str) -> str | None:
    if not ensure_cache_index():
        return None
    try:
        client = get_client()
        response = client.get(index=LLM_CACHE_INDEX, id=cache_key)
        source = response.get("_source", {})
    except exceptions.NotFoundError:
        return None
    except exceptions.OpenSearchException as exc:
        _warn_unavailable(exc)
        return None
    except Exception as exc:  # noqa: BLE001
        _warn_unavailable(exc)
        return None

    expires_at = _parse_iso(source.get("expires_at"))
    if expires_at and datetime.now(timezone.utc) > expires_at:
        return None

    try:
        client.update(
            index=LLM_CACHE_INDEX,
            id=cache_key,
            body={
                "script": {
                    "source": (
                        "ctx._source.hit_count = (ctx._source.hit_count != null ? "
                        "ctx._source.hit_count : 0) + params.count; "
                        "ctx._source.last_access_at = params.ts;"
                    ),
                    "lang": "painless",
                    "params": {"count": 1, "ts": _now_iso()},
                }
            },
        )
    except exceptions.OpenSearchException:
        pass
    except Exception:  # noqa: BLE001
        pass

    return source.get("response_text")


def store_cache_entry(
    *,
    cache_key: str,
    canonical: Dict[str, Any],
    prompt_text: str,
    prompt_hash: str,
    response_text: str,
    model_id: str,
    endpoint_id: str,
    status: str,
) -> None:
    if not ensure_cache_index():
        return

    now_iso = _now_iso()
    expires_at = None
    if LLM_CACHE_TTL_DAYS and LLM_CACHE_TTL_DAYS > 0:
        expires_at = (datetime.now(timezone.utc) + timedelta(days=LLM_CACHE_TTL_DAYS)).isoformat()

    doc = {
        "cache_key": cache_key,
        "created_at": now_iso,
        "last_access_at": now_iso,
        "hit_count": 0,
        "model_id": model_id,
        "endpoint_id": endpoint_id,
        "request_fingerprint": cache_key,
        "request_params": build_request_params(canonical, prompt_hash),
        "prompt_hash": prompt_hash,
        "prompt_text": prompt_text if LLM_CACHE_STORE_PROMPT_TEXT else "",
        "response_text": response_text,
        "response_hash": _hash_text(response_text),
        "status": status,
    }
    if expires_at:
        doc["expires_at"] = expires_at

    try:
        client = get_client()
        client.index(index=LLM_CACHE_INDEX, id=cache_key, body=doc)
    except exceptions.OpenSearchException as exc:
        _warn_unavailable(exc)
    except Exception as exc:  # noqa: BLE001
        _warn_unavailable(exc)
