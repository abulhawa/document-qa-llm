from __future__ import annotations
from typing import Any, Dict, List, TypedDict, Optional

class DocHit(TypedDict, total=False):
    id: str              # semantic id
    _id: str             # bm25 id
    path: str
    text: str
    score: float
    chunk_index: int
    modified_at: str
    checksum: str
    # hybrid fields
    score_vector: float
    score_bm25: float
    hybrid_score: float
    source: str          # "semantic" | "keyword" | "semantic/keyword"
