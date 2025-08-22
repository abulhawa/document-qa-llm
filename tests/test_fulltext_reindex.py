from langchain_core.documents import Document
from utils import opensearch_utils


def test_list_files_missing_fulltext(monkeypatch):
    monkeypatch.setattr(
        opensearch_utils,
        "list_files_from_opensearch",
        lambda size=1000: [{"path": "a"}, {"path": "b"}],
    )
    monkeypatch.setattr(
        opensearch_utils, "list_fulltext_paths", lambda size=1000: ["a"]
    )
    missing = opensearch_utils.list_files_missing_fulltext()
    assert [m["path"] for m in missing] == ["b"]


def test_reindex_fulltext_from_chunks(tmp_path, monkeypatch):
    f = tmp_path / "f.txt"
    f.write_text("raw")

    monkeypatch.setattr(
        opensearch_utils,
        "load_documents",
        lambda p: [Document(page_content="one"), Document(page_content="two")],
    )
    monkeypatch.setattr(
        opensearch_utils,
        "preprocess_to_documents",
        lambda docs_like, source_path, cfg, doc_type: [
            Document(page_content="A"),
            Document(page_content="B"),
        ],
    )
    monkeypatch.setattr(opensearch_utils, "compute_checksum", lambda p: "x")
    monkeypatch.setattr(opensearch_utils, "get_file_size", lambda p: 1)
    monkeypatch.setattr(
        opensearch_utils,
        "get_file_timestamps",
        lambda p: {"created": "c", "modified": "m"},
    )
    monkeypatch.setattr(opensearch_utils, "hash_path", lambda p: "id")

    captured = []

    def fake_index(doc):
        captured.append(doc)

    monkeypatch.setattr(opensearch_utils, "index_fulltext_document", fake_index)
    count = opensearch_utils.reindex_fulltext_from_chunks([str(f)])
    assert count == 1
    assert captured and captured[0]["text_full"] == "A\n\nB"
    assert captured[0]["id"] == "id"
