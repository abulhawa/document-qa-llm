"""Build the reproducible source corpus for a mixed retrieval benchmark.

The source lock is the canonical definition. Raw source files are cached for speed,
but they can always be reconstructed from the lock and immutable upstream revisions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests
import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

from evaluation.benchmarks.selection import (
    DEFAULT_SEED,
    SELECTION_ALGORITHM,
    select_miracl,
    select_nfcorpus,
    select_officeqa,
    select_open_ragbench,
)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_dump(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "__", value).strip("._")
    return clean or hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _split_source_files(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).split(";")
    return [str(item).strip() for item in items if str(item).strip()]


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_text_document(track_dir: Path, doc_id: str, title: str, text: str) -> dict[str, Any]:
    rel = Path("documents") / f"{_safe_name(doc_id)}.txt"
    path = track_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"{title.strip()}\n\n{text.strip()}".strip() + "\n" if title.strip() else text.strip() + "\n"
    path.write_text(content, encoding="utf-8")
    return {
        "source_document_id": doc_id,
        "path": rel.as_posix(),
        "media_type": "text/plain",
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "title": title,
    }


def _download_http_pdf(url: str, cache_path: Path, *, session: requests.Session) -> Path:
    if cache_path.exists() and cache_path.stat().st_size > 4:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".part")
    for attempt in range(3):
        try:
            with session.get(url, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            fh.write(chunk)
            if tmp.stat().st_size < 5:
                raise RuntimeError(f"downloaded file is unexpectedly small: {url}")
            tmp.replace(cache_path)
            return cache_path
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def _resolve_revision(api: HfApi, repo_id: str, token: str | None) -> str:
    info = api.dataset_info(repo_id=repo_id, revision="main", token=token)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return an immutable revision for {repo_id}")
    return info.sha


def _hf_json(repo_id: str, filename: str, revision: str, token: str | None) -> Any:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        revision=revision,
        token=token,
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_csv(repo_id: str, filename: str, revision: str, token: str | None) -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        revision=revision,
        token=token,
    )
    with Path(path).open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _spec_sha(spec_path: Path) -> str:
    return _sha256_file(spec_path)


def _pending_lock(path: Path) -> bool:
    if not path.exists():
        return True
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("status") != "locked"


def _resolve_and_select(
    spec: Mapping[str, Any],
    spec_path: Path,
    *,
    token: str,
) -> dict[str, Any]:
    api = HfApi()
    seed = int(spec.get("selection_seed", DEFAULT_SEED))
    sources = spec["sources"]

    repos = {
        "open_ragbench": sources["open_ragbench"]["upstream"],
        "officeqa": sources["officeqa"]["upstream"],
        "nfcorpus": sources["nfcorpus"]["upstream"],
        "nfcorpus_qrels": sources["nfcorpus"]["qrels_upstream"],
        "miracl_de": sources["miracl_de"]["upstream"],
        "miracl_ar": sources["miracl_ar"]["upstream"],
    }
    revisions = {name: _resolve_revision(api, repo_id, token) for name, repo_id in repos.items()}

    open_repo = repos["open_ragbench"]
    open_rev = revisions["open_ragbench"]
    open_queries = _hf_json(open_repo, "pdf/arxiv/queries.json", open_rev, token)
    open_qrels = _hf_json(open_repo, "pdf/arxiv/qrels.json", open_rev, token)
    open_urls = _hf_json(open_repo, "pdf/arxiv/pdf_urls.json", open_rev, token)
    open_cfg = sources["open_ragbench"]["selection"]
    open_selected = select_open_ragbench(
        open_queries,
        open_qrels,
        open_urls.keys(),
        positive_documents=int(open_cfg["positive_documents"]),
        hard_negative_documents=int(open_cfg["hard_negative_documents"]),
        questions_per_document=int(open_cfg["questions_per_positive_target"]),
        seed=seed,
    )

    office_repo = repos["officeqa"]
    office_rev = revisions["officeqa"]
    office_rows = _load_csv(office_repo, "officeqa_full.csv", office_rev, token)
    for row in office_rows:
        row["source_files"] = _split_source_files(row.get("source_files"))
    repo_files = api.list_repo_files(repo_id=office_repo, repo_type="dataset", revision=office_rev, token=token)
    office_corpus_names = [
        Path(name).name.removesuffix(".pdf") + ".txt"
        for name in repo_files
        if name.startswith("treasury_bulletin_pdfs/") and name.endswith(".pdf")
    ]
    office_cfg = sources["officeqa"]["selection"]
    office_selected = select_officeqa(
        office_rows,
        office_corpus_names,
        easy_questions=int(office_cfg["easy_questions"]),
        hard_questions=int(office_cfg["hard_questions"]),
        target_total_documents=int(office_cfg["distractor_documents_target_total_pdfs"]),
        seed=seed,
    )

    nf_repo = repos["nfcorpus"]
    nf_rev = revisions["nfcorpus"]
    nf_queries_ds = load_dataset(nf_repo, "queries", split="queries", revision=nf_rev, token=token)
    nf_qrels_ds = load_dataset(
        repos["nfcorpus_qrels"],
        split="test",
        revision=revisions["nfcorpus_qrels"],
        token=token,
    )
    positive_nf_queries = {
        str(row["query-id"])
        for row in nf_qrels_ds
        if int(row["score"]) > 0
    }
    nf_query_ids = [str(row["_id"]) for row in nf_queries_ds if str(row["_id"]) in positive_nf_queries]
    nf_selected = {
        "query_ids": select_nfcorpus(
            nf_query_ids,
            queries=int(sources["nfcorpus"]["selection"]["test_queries"]),
            seed=seed,
        ),
        "corpus": "full",
    }

    miracl_selected: dict[str, Any] = {}
    for track in ("miracl_de", "miracl_ar"):
        repo_id = repos[track]
        revision = revisions[track]
        language = sources[track]["language"]
        qrels_ds = load_dataset(repo_id, split="test", revision=revision, token=token)
        queries_ds = load_dataset(repo_id, "queries", split="test", revision=revision, token=token)
        qrels_rows = [dict(row) for row in qrels_ds]
        positive_query_ids = {
            str(row["query-id"]) for row in qrels_rows if int(row["score"]) > 0
        }
        query_ids = [str(row["_id"]) for row in queries_ds if str(row["_id"]) in positive_query_ids]
        cfg = sources[track]["selection"]
        selected = select_miracl(
            query_ids,
            qrels_rows,
            language=language,
            queries=int(cfg["queries"]),
            hard_negatives_per_query=int(cfg["hard_negatives_per_query_max"]),
            seed=seed,
        )
        miracl_selected[track] = {
            "query_ids": selected["query_ids"],
            "document_ids": selected["document_ids"],
        }

    return {
        "schema_version": 1,
        "benchmark_id": spec["benchmark_id"],
        "status": "locked",
        "selection_algorithm": spec.get("selection_algorithm", SELECTION_ALGORITHM),
        "selection_seed": seed,
        "spec_sha256": _spec_sha(spec_path),
        "sources": {
            name: {
                "repo_id": repo_id,
                "revision": revisions[name],
            }
            for name, repo_id in repos.items()
        },
        "selection": {
            "open_ragbench": open_selected,
            "officeqa": office_selected,
            "nfcorpus": nf_selected,
            **miracl_selected,
        },
    }


def _load_lock_or_resolve(
    spec: Mapping[str, Any],
    spec_path: Path,
    lock_path: Path,
    *,
    token: str,
) -> dict[str, Any]:
    if not _pending_lock(lock_path):
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("benchmark_id") != spec.get("benchmark_id"):
            raise ValueError("lock benchmark_id does not match composition")
        if lock.get("spec_sha256") != _spec_sha(spec_path):
            raise ValueError("locked source was created for a different composition file")
        return lock
    return _resolve_and_select(spec, spec_path, token=token)


def _materialize_open_ragbench(
    out_dir: Path,
    cache_dir: Path,
    lock: Mapping[str, Any],
    *,
    token: str,
) -> dict[str, Any]:
    track = "open_ragbench"
    source = lock["sources"][track]
    selected = lock["selection"][track]
    repo_id, revision = source["repo_id"], source["revision"]
    queries = _hf_json(repo_id, "pdf/arxiv/queries.json", revision, token)
    qrels = _hf_json(repo_id, "pdf/arxiv/qrels.json", revision, token)
    answers = _hf_json(repo_id, "pdf/arxiv/answers.json", revision, token)
    urls = _hf_json(repo_id, "pdf/arxiv/pdf_urls.json", revision, token)

    track_dir = out_dir / track
    all_docs = selected["positive_document_ids"] + selected["hard_negative_document_ids"]
    session = requests.Session()
    docs: list[dict[str, Any]] = []
    for doc_id in all_docs:
        url = str(urls[doc_id])
        cache_path = cache_dir / track / f"{_safe_name(doc_id)}.pdf"
        downloaded = _download_http_pdf(url, cache_path, session=session)
        rel = Path("documents") / f"{_safe_name(doc_id)}.pdf"
        destination = track_dir / rel
        _copy_file(downloaded, destination)
        docs.append({
            "source_document_id": doc_id,
            "path": rel.as_posix(),
            "media_type": "application/pdf",
            "sha256": _sha256_file(destination),
            "bytes": destination.stat().st_size,
            "source_url": url,
            "role": "positive" if doc_id in set(selected["positive_document_ids"]) else "hard_negative",
        })

    query_rows = []
    qrel_rows = []
    answer_rows = []
    for query_id in selected["query_ids"]:
        query = queries[query_id]
        rel = qrels[query_id]
        query_rows.append({
            "query_id": query_id,
            "text": query["query"],
            "type": query.get("type"),
            "modality": query.get("source"),
        })
        qrel_rows.append({
            "query_id": query_id,
            "document_id": str(rel["doc_id"]),
            "section_id": rel.get("section_id"),
            "score": 1,
        })
        if query_id in answers:
            answer_rows.append({"query_id": query_id, "answer": answers[query_id]})

    _jsonl_dump(track_dir / "documents.jsonl", docs)
    _jsonl_dump(track_dir / "queries.jsonl", query_rows)
    _jsonl_dump(track_dir / "qrels.jsonl", qrel_rows)
    _jsonl_dump(track_dir / "answers.jsonl", answer_rows)
    return {"documents": len(docs), "queries": len(query_rows), "qrels": len(qrel_rows)}


def _materialize_officeqa(
    out_dir: Path,
    lock: Mapping[str, Any],
    *,
    token: str,
) -> dict[str, Any]:
    track = "officeqa"
    source = lock["sources"][track]
    selected = lock["selection"][track]
    repo_id, revision = source["repo_id"], source["revision"]
    rows = _load_csv(repo_id, "officeqa_full.csv", revision, token)
    by_id = {str(row["uid"]): row for row in rows}
    selected_rows = [by_id[query_id] for query_id in selected["query_ids"]]

    track_dir = out_dir / track
    document_ids = selected["positive_document_ids"] + selected["distractor_document_ids"]
    positive_set = set(selected["positive_document_ids"])
    docs: list[dict[str, Any]] = []
    for doc_id in document_ids:
        stem = Path(doc_id).stem
        hf_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"treasury_bulletin_pdfs/{stem}.pdf",
            revision=revision,
            token=token,
        )
        rel = Path("documents") / f"{stem}.pdf"
        destination = track_dir / rel
        _copy_file(Path(hf_path), destination)
        docs.append({
            "source_document_id": doc_id,
            "path": rel.as_posix(),
            "media_type": "application/pdf",
            "sha256": _sha256_file(destination),
            "bytes": destination.stat().st_size,
            "role": "positive" if doc_id in positive_set else "distractor",
        })

    query_rows = []
    qrel_rows = []
    answer_rows = []
    for row in selected_rows:
        query_id = str(row["uid"])
        source_files = _split_source_files(row.get("source_files"))
        query_rows.append({
            "query_id": query_id,
            "text": row["question"],
            "difficulty": row.get("difficulty"),
        })
        for source_file in source_files:
            qrel_rows.append({"query_id": query_id, "document_id": source_file, "score": 1})
        answer_rows.append({"query_id": query_id, "answer": row.get("answer")})

    _jsonl_dump(track_dir / "documents.jsonl", docs)
    _jsonl_dump(track_dir / "queries.jsonl", query_rows)
    _jsonl_dump(track_dir / "qrels.jsonl", qrel_rows)
    _jsonl_dump(track_dir / "answers.jsonl", answer_rows)
    return {"documents": len(docs), "queries": len(query_rows), "qrels": len(qrel_rows)}


def _materialize_nfcorpus(
    out_dir: Path,
    lock: Mapping[str, Any],
    *,
    token: str,
) -> dict[str, Any]:
    track = "nfcorpus"
    source = lock["sources"][track]
    qrels_source = lock["sources"]["nfcorpus_qrels"]
    selected_ids = set(lock["selection"][track]["query_ids"])

    corpus_ds = load_dataset(
        source["repo_id"], "corpus", split="corpus", revision=source["revision"], token=token
    )
    queries_ds = load_dataset(
        source["repo_id"], "queries", split="queries", revision=source["revision"], token=token
    )
    qrels_ds = load_dataset(
        qrels_source["repo_id"], split="test", revision=qrels_source["revision"], token=token
    )

    track_dir = out_dir / track
    docs = [
        _write_text_document(
            track_dir,
            str(row["_id"]),
            str(row.get("title") or ""),
            str(row["text"]),
        )
        for row in corpus_ds
    ]
    query_rows = [
        {"query_id": str(row["_id"]), "text": str(row["text"])}
        for row in queries_ds
        if str(row["_id"]) in selected_ids
    ]
    qrel_rows = [
        {
            "query_id": str(row["query-id"]),
            "document_id": str(row["corpus-id"]),
            "score": int(row["score"]),
        }
        for row in qrels_ds
        if str(row["query-id"]) in selected_ids
    ]
    _jsonl_dump(track_dir / "documents.jsonl", docs)
    _jsonl_dump(track_dir / "queries.jsonl", query_rows)
    _jsonl_dump(track_dir / "qrels.jsonl", qrel_rows)
    return {"documents": len(docs), "queries": len(query_rows), "qrels": len(qrel_rows)}


def _materialize_miracl(
    track: str,
    out_dir: Path,
    lock: Mapping[str, Any],
    *,
    token: str,
) -> dict[str, Any]:
    source = lock["sources"][track]
    selected = lock["selection"][track]
    selected_queries = set(selected["query_ids"])
    selected_docs = set(selected["document_ids"])
    repo_id, revision = source["repo_id"], source["revision"]

    queries_ds = load_dataset(repo_id, "queries", split="test", revision=revision, token=token)
    corpus_ds = load_dataset(repo_id, "corpus", split="test", revision=revision, token=token)
    qrels_ds = load_dataset(repo_id, split="test", revision=revision, token=token)

    track_dir = out_dir / track
    docs = [
        _write_text_document(
            track_dir,
            str(row["_id"]),
            str(row.get("title") or ""),
            str(row["text"]),
        )
        for row in corpus_ds
        if str(row["_id"]) in selected_docs
    ]
    query_rows = [
        {"query_id": str(row["_id"]), "text": str(row["text"])}
        for row in queries_ds
        if str(row["_id"]) in selected_queries
    ]
    qrel_rows = [
        {
            "query_id": str(row["query-id"]),
            "document_id": str(row["corpus-id"]),
            "score": int(row["score"]),
        }
        for row in qrels_ds
        if str(row["query-id"]) in selected_queries and str(row["corpus-id"]) in selected_docs
    ]
    _jsonl_dump(track_dir / "documents.jsonl", docs)
    _jsonl_dump(track_dir / "queries.jsonl", query_rows)
    _jsonl_dump(track_dir / "qrels.jsonl", qrel_rows)
    return {"documents": len(docs), "queries": len(query_rows), "qrels": len(qrel_rows)}


def build_source(
    spec_path: Path,
    lock_path: Path,
    output_root: Path,
    cache_dir: Path,
    *,
    token: str,
) -> Path:
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    benchmark_id = str(spec["benchmark_id"])
    output_dir = output_root / benchmark_id

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    lock = _load_lock_or_resolve(spec, spec_path, lock_path, token=token)
    _json_dump(output_dir / "source.lock.json", lock)

    stats = {
        "open_ragbench": _materialize_open_ragbench(output_dir, cache_dir, lock, token=token),
        "officeqa": _materialize_officeqa(output_dir, lock, token=token),
        "nfcorpus": _materialize_nfcorpus(output_dir, lock, token=token),
        "miracl_de": _materialize_miracl("miracl_de", output_dir, lock, token=token),
        "miracl_ar": _materialize_miracl("miracl_ar", output_dir, lock, token=token),
    }

    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {"manifest.json"}:
            files.append({
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            })

    manifest = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "artifact_type": "source",
        "lock_sha256": _sha256_file(output_dir / "source.lock.json"),
        "spec_sha256": _spec_sha(spec_path),
        "stats": stats,
        "files": files,
    }
    _json_dump(output_dir / "manifest.json", manifest)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(".benchmark/source"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".benchmark/cache/source"))
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is required (OfficeQA is gated).", file=sys.stderr)
        return 2

    output_dir = build_source(
        args.spec,
        args.lock,
        args.output_root,
        args.cache_dir,
        token=token,
    )
    lock_path = output_dir / "source.lock.json"
    print(f"Built source corpus: {output_dir}")
    print(f"Source lock SHA256: {_sha256_file(lock_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
