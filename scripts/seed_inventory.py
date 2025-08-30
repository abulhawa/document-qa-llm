#!/usr/bin/env python
from __future__ import annotations

import argparse
from typing import Optional

from utils.inventory import (
    ensure_watch_inventory_index_exists,
    seed_watch_inventory_from_fulltext,
    seed_inventory_indexed_chunked_count,
    count_watch_inventory_remaining,
)


def main(prefix: str, do_fulltext: bool, do_chunks: bool) -> None:
    ensure_watch_inventory_index_exists()
    if do_fulltext:
        n = seed_watch_inventory_from_fulltext(prefix)
        print(f"Seeded from full-text: {n} entry(ies)")
    if do_chunks:
        n = seed_inventory_indexed_chunked_count(prefix)
        print(f"Seeded chunk counts: {n} entry(ies)")
    remaining = count_watch_inventory_remaining(prefix)
    print(f"Remaining (exists_now & missing last_indexed) under prefix: {remaining}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Seed watch inventory from existing indices")
    ap.add_argument("prefix", help="Path prefix/folder to target (e.g., C:/Docs)")
    ap.add_argument("--fulltext", action="store_true", help="Seed from full-text index")
    ap.add_argument("--chunks", action="store_true", help="Seed chunked counts from documents index")
    args = ap.parse_args()
    # If neither flag is set, do both
    do_fulltext = args.fulltext or (not args.fulltext and not args.chunks)
    do_chunks = args.chunks or (not args.fulltext and not args.chunks)
    main(args.prefix, do_fulltext, do_chunks)

