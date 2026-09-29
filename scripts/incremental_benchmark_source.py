"""Build a new frozen corpus while copying unchanged source documents from v1."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path
from typing import Any

from scripts import build_benchmark_source as source


def _load_lock(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding="utf-8"))


def build(args: argparse.Namespace) -> Path:
    base_manifest = json.loads((args.base_source_root / "manifest.json").read_text(encoding="utf-8"))
    if base_manifest.get("artifact_type") != "source" or not base_manifest.get("persistence", {}).get("persisted"):
        raise ValueError("base source artifact is not persisted")
    old_lock = _load_lock(args.base_source_root / "source.lock.json")
    new_lock = _load_lock(args.lock)
    files = {item["path"]: item for item in base_manifest["files"]}
    if old_lock["sources"]["open_ragbench"] != new_lock["sources"]["open_ragbench"]:
        raise ValueError("Open RAGBench source revision changed; refuse old PDF reuse")

    old_docs = {
        row["source_url"]: row
        for row in (
            json.loads(line)
            for line in (args.base_source_root / "open_ragbench" / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    original_download = source._download_http_file
    counters = {"open_ragbench_pdfs_reused": 0, "open_ragbench_pdfs_downloaded": 0, "tracks_reused": 0}

    def cached_download(url: str, cache_path: Path, *, session: Any) -> Path:
        row = old_docs.get(url)
        if row:
            relative = f"open_ragbench/{row['path']}"
            previous = args.base_source_root / relative
            listed = files.get(relative)
            if (
                listed and previous.is_file() and previous.stat().st_size == listed["bytes"]
                and source._sha256_file(previous) == listed["sha256"] == row["sha256"]
            ):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(previous, cache_path)
                counters["open_ragbench_pdfs_reused"] += 1
                return cache_path
        counters["open_ragbench_pdfs_downloaded"] += 1
        return original_download(url, cache_path, session=session)

    original_tracks = {
        "officeqa": source._materialize_officeqa,
        "nfcorpus": source._materialize_nfcorpus,
        "miracl_de": source._materialize_miracl,
        "miracl_ar": source._materialize_miracl,
    }

    def reuse_track(track: str, output_dir: Path) -> dict[str, Any]:
        previous = args.base_source_root / track
        if not previous.is_dir():
            raise ValueError(f"missing base track {track}")
        for name, meta in files.items():
            if name.startswith(f"{track}/"):
                path = args.base_source_root / name
                if not path.is_file() or path.stat().st_size != meta["bytes"] or source._sha256_file(path) != meta["sha256"]:
                    raise ValueError(f"base track file checksum mismatch: {name}")
        shutil.copytree(previous, output_dir / track)
        counters["tracks_reused"] += 1
        return dict(base_manifest["stats"][track])

    def officeqa(out_dir: Path, lock: Any, *, token: str) -> dict[str, Any]:
        if old_lock["sources"]["officeqa"] == new_lock["sources"]["officeqa"] and old_lock["selection"]["officeqa"] == new_lock["selection"]["officeqa"]:
            return reuse_track("officeqa", out_dir)
        return original_tracks["officeqa"](out_dir, lock, token=token)

    def nfcorpus(out_dir: Path, cache_dir: Path, lock: Any) -> dict[str, Any]:
        if old_lock["sources"]["nfcorpus"] == new_lock["sources"]["nfcorpus"] and old_lock["selection"]["nfcorpus"] == new_lock["selection"]["nfcorpus"]:
            return reuse_track("nfcorpus", out_dir)
        return original_tracks["nfcorpus"](out_dir, cache_dir, lock)

    def miracl(track: str, out_dir: Path, lock: Any, *, token: str) -> dict[str, Any]:
        if old_lock["sources"][track] == new_lock["sources"][track] and old_lock["selection"][track] == new_lock["selection"][track]:
            return reuse_track(track, out_dir)
        return original_tracks[track](track, out_dir, lock, token=token)

    source._download_http_file = cached_download
    source._materialize_officeqa = officeqa
    source._materialize_nfcorpus = nfcorpus
    source._materialize_miracl = miracl
    try:
        result = source.build_source(args.spec, args.lock, args.output_root, args.cache_dir, token=args.token)
    finally:
        source._download_http_file = original_download
        source._materialize_officeqa = original_tracks["officeqa"]
        source._materialize_nfcorpus = original_tracks["nfcorpus"]
        source._materialize_miracl = original_tracks["miracl_de"]
    path = result / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["reuse"] = {**counters, "base_artifact_fingerprint": base_manifest["artifact_fingerprint"]}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(counters))
    return result


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--spec", type=Path, required=True)
    cli.add_argument("--lock", type=Path, required=True)
    cli.add_argument("--base-source-root", type=Path, required=True)
    cli.add_argument("--output-root", type=Path, default=Path(".benchmark/source"))
    cli.add_argument("--cache-dir", type=Path, default=Path(".benchmark/cache/source"))
    args = cli.parse_args()
    import os
    args.token = os.environ["HF_TOKEN"]
    build(args)


if __name__ == "__main__":
    main()

