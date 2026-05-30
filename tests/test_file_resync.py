import sys
import types

from core.sync import file_resync
from core.sync.file_resync import (
    Action,
    ApplyOptions,
    FileHit,
    IndexedDoc,
    PlanItem,
    ReconciliationPlan,
    ScanFilterOptions,
    ScanResult,
    apply_plan,
    build_reconciliation_plan,
    scan_files,
)


def _patch_snapshot(monkeypatch, docs):
    monkeypatch.setattr(
        file_resync,
        "_load_index_snapshot",
        lambda: ({doc.checksum: doc for doc in docs}, {}, {}),
    )


def test_partial_root_scan_does_not_mark_content_orphaned(monkeypatch):
    _patch_snapshot(
        monkeypatch,
        [
            IndexedDoc(
                content_id="c1",
                checksum="checksum-1",
                canonical_path="/sync/missing.pdf",
                aliases=["/outside/still-possible.pdf"],
            )
        ],
    )

    plan = build_reconciliation_plan(
        ScanResult(
            hits=[],
            scanned_roots_successful=["/sync"],
            scanned_roots_failed=[],
        )
    )

    assert plan.items == []


def test_missing_canonical_with_multiple_disk_paths_requires_review(monkeypatch):
    _patch_snapshot(
        monkeypatch,
        [
            IndexedDoc(
                content_id="c1",
                checksum="checksum-1",
                canonical_path="/sync/old.pdf",
                aliases=[],
            )
        ],
    )
    hits = [
        FileHit("/sync/a.pdf", size=2000, mtime=10.0, checksum="checksum-1", ext="pdf"),
        FileHit("/sync/b.pdf", size=2000, mtime=20.0, checksum="checksum-1", ext="pdf"),
    ]

    plan = build_reconciliation_plan(
        ScanResult(
            hits=hits,
            scanned_roots_successful=["/sync"],
            scanned_roots_failed=[],
        )
    )

    assert len(plan.items) == 1
    item = plan.items[0]
    assert item.bucket == "REVIEW"
    assert "CANONICAL_AMBIGUOUS" in item.reason
    assert any(action.type == "SET_CANONICAL" for action in item.actions)


def test_safe_apply_can_ingest_info_items_when_enabled(monkeypatch):
    ingested = []
    orchestrator_stub = types.ModuleType("ingestion.orchestrator")
    orchestrator_stub.ingest_one = lambda path, **kwargs: ingested.append((path, kwargs))
    monkeypatch.setitem(sys.modules, "ingestion.orchestrator", orchestrator_stub)
    plan = ReconciliationPlan(
        items=[
            PlanItem(
                bucket="INFO",
                reason="NOT_INDEXED",
                checksum="checksum-1",
                content_id=None,
                disk_paths=["/sync/new.pdf"],
                indexed_paths=[],
                actions=[Action("INGEST_NEW", {"path": "/sync/new.pdf"})],
            )
        ],
        counts={"INFO": 1},
        scanned_roots=["/sync"],
        scanned_roots_failed=[],
        generated_at=file_resync.datetime.now(file_resync.timezone.utc),
    )

    result = apply_plan(
        plan,
        ApplyOptions(ingest_missing=True, apply_safe_only=True),
    )

    assert result.ingested == 1
    assert ingested == [
        ("/sync/new.pdf", {"force": True, "replace": True, "op": "resync", "source": "resync"})
    ]


def test_destructive_delete_actions_require_matching_apply_option(monkeypatch):
    deleted = []
    qdrant_stub = types.ModuleType("utils.qdrant_utils")
    qdrant_stub.delete_vectors_by_checksum = lambda checksum: deleted.append(("q", checksum))
    monkeypatch.setitem(sys.modules, "utils.qdrant_utils", qdrant_stub)
    monkeypatch.setattr(file_resync, "delete_chunks_by_checksum", lambda checksum: deleted.append(("c", checksum)))
    monkeypatch.setattr(file_resync, "delete_fulltext_by_checksum", lambda checksum: deleted.append(("f", checksum)))

    plan = ReconciliationPlan(
        items=[
            PlanItem(
                bucket="REVIEW",
                reason="ORPHANED_INDEX_CONTENT",
                checksum="orphan",
                content_id="orphan",
                disk_paths=[],
                indexed_paths=["/sync/missing.pdf"],
                actions=[Action("DELETE_CONTENT", {"checksum": "orphan", "conditional": True})],
            ),
            PlanItem(
                bucket="REVIEW",
                reason="PATH_REPLACED",
                checksum="retired",
                content_id="retired",
                disk_paths=[],
                indexed_paths=["/sync/replaced.pdf"],
                actions=[
                    Action(
                        "DELETE_CONTENT",
                        {
                            "checksum": "retired",
                            "conditional": True,
                            "reason": "retire_replaced_content",
                        },
                    )
                ],
            ),
        ],
        counts={"REVIEW": 2},
        scanned_roots=["/sync"],
        scanned_roots_failed=[],
        generated_at=file_resync.datetime.now(file_resync.timezone.utc),
    )

    apply_plan(plan, ApplyOptions(apply_safe_only=False, delete_orphaned=True))

    assert [checksum for _, checksum in deleted] == ["orphan", "orphan", "orphan"]


def test_scan_filters_are_customizable(tmp_path):
    tiny_temp = tmp_path / "~$tiny.txt"
    tiny_temp.write_text("x")

    default_scan = scan_files([str(tmp_path)], [".txt"])

    custom_scan = scan_files(
        [str(tmp_path)],
        [".txt"],
        ScanFilterOptions(
            min_ingest_bytes=0,
            temp_prefixes=(),
            temp_suffixes=(),
            ignore_dir_names=(),
        ),
    )

    assert default_scan.hits == []
    assert default_scan.ignored_files == 1
    assert [hit.path for hit in custom_scan.hits] == [file_resync.normalize_path(str(tiny_temp))]
