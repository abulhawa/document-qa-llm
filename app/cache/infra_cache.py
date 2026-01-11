"""Infrastructure status cache for UI-agnostic usage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class InfraStatus:
    service: str
    ok: bool
    checked_at: datetime = field(default_factory=_utcnow)
    details: Dict[str, Any] = field(default_factory=dict)
    ttl_seconds: int = 30

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        reference = now or _utcnow()
        return reference - self.checked_at > timedelta(seconds=self.ttl_seconds)


class InfraCache:
    """In-memory cache for infrastructure status (LLM, OpenSearch, Qdrant)."""

    def __init__(self) -> None:
        self._entries: Dict[str, InfraStatus] = {}

    def get(self, service: str) -> Optional[InfraStatus]:
        status = self._entries.get(service)
        if not status:
            return None
        if status.is_expired():
            self._entries.pop(service, None)
            return None
        return status

    def set(self, status: InfraStatus) -> None:
        self._entries[status.service] = status

    def update(self, service: str, ok: bool, details: Optional[Dict[str, Any]] = None) -> InfraStatus:
        payload = details or {}
        status = InfraStatus(service=service, ok=ok, details=payload)
        self._entries[service] = status
        return status

    def clear(self, service: str) -> None:
        self._entries.pop(service, None)

    def clear_expired(self) -> int:
        now = _utcnow()
        expired = [key for key, entry in self._entries.items() if entry.is_expired(now)]
        for key in expired:
            self._entries.pop(key, None)
        return len(expired)
