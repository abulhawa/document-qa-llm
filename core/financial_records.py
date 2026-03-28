from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def get_financial_records_for_checksums(
    *,
    checksums: Sequence[str],
    year: Optional[int] = None,
    size: int = 200,
) -> List[Dict[str, Any]]:
    from ingestion.financial_records_store import fetch_financial_records

    return fetch_financial_records(checksums=checksums, year=year, size=size)
