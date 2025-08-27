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
