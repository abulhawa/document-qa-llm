"""Run-scoped cache for background actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RunCacheEntry:
    run_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    ttl_seconds: Optional[int] = None

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        if self.ttl_seconds is None:
            return False
        reference = now or _utcnow()
        return reference - self.updated_at > timedelta(seconds=self.ttl_seconds)


class RunCache:
    """In-memory cache keyed by run id.

    Intended for storing run-local state (progress, summaries) in a UI-agnostic way.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, RunCacheEntry] = {}

    def get(self, run_id: str) -> Optional[RunCacheEntry]:
        entry = self._entries.get(run_id)
        if not entry:
            return None
        if entry.is_expired():
            self._entries.pop(run_id, None)
            return None
        return entry

    def set(self, entry: RunCacheEntry) -> None:
        entry.updated_at = _utcnow()
        self._entries[entry.run_id] = entry

    def update(self, run_id: str, payload: Dict[str, Any]) -> RunCacheEntry:
        existing = self._entries.get(run_id)
        if existing is None:
            existing = RunCacheEntry(run_id=run_id, payload={})
        existing.payload.update(payload)
        existing.updated_at = _utcnow()
        self._entries[run_id] = existing
        return existing

    def clear(self, run_id: str) -> None:
        self._entries.pop(run_id, None)

    def clear_expired(self) -> int:
        now = _utcnow()
        expired = [key for key, entry in self._entries.items() if entry.is_expired(now)]
        for key in expired:
            self._entries.pop(key, None)
        return len(expired)
