import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest


def test_status_filter_and_pagination(monkeypatch):
    """Search ingest logs UI shows filtered results and different pages."""
    logs = [
        {"path": "/a.txt", "status": "Success"},
        {"path": "/b.txt", "status": "Failed"},
        {"path": "/c.txt", "status": "Success"},
        {"path": "/d.txt", "status": "Failed"},
    ]
    page = {"value": 0}
    page_size = 2

    def fake_search_ingest_logs(*, status=None, **kwargs):
        start = page["value"] * page_size
        subset = logs[start : start + page_size]
        if status:
            subset = [l for l in subset if l["status"] == status]
        return subset

    monkeypatch.setattr(
        "utils.opensearch_utils.search_ingest_logs", fake_search_ingest_logs
    )

    at = AppTest.from_file("pages/4_ingest_logs.py", default_timeout=10)
    at.run()

    # First page shows first two log paths
    df = at.dataframe[0].value
    assert df["Path"].tolist() == ["/a.txt", "/b.txt"]

    # Filtering by status adjusts row counts
    at.selectbox[0].select("Success").run()
    assert at.dataframe[0].value["Path"].tolist() == ["/a.txt"]

    at.selectbox[0].select("Failed").run()
    assert at.dataframe[0].value["Path"].tolist() == ["/b.txt"]

    # Reset filter and switch to second page
    at.selectbox[0].select("All").run()
    page["value"] = 1
    at.run()
    assert at.dataframe[0].value["Path"].tolist() == ["/c.txt", "/d.txt"]
