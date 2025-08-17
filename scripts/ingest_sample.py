"""Ingest a small folder of documents for smoke testing."""

import os
import sys

from core.ingestion import ingest_one


def ingest_folder(folder: str) -> None:
    for root, _dirs, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            ingest_one(path)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "assets/sample_docs"
    ingest_folder(folder)

