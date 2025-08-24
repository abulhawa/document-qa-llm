from dataclasses import dataclass
from typing import List
from core.chunking import split_documents


@dataclass
class Chunk:
    text: str
    metadata: dict


def split_with_langchain(docs: List, chunk_size: int, overlap: int) -> List[Chunk]:
    parts = split_documents(docs, chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for p in parts:
        meta = {
            "path": p.get("path"),
            "page": p.get("page"),
            "chunk_index": p.get("chunk_index"),
            "location_percent": p.get("location_percent"),
        }
        chunks.append(Chunk(text=p.get("text", ""), metadata=meta))
    return chunks
