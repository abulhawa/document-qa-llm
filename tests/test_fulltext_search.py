from utils.fulltext_search import build_query

def test_build_query_basic():
    body = build_query("hello")
    assert body["query"]["bool"]["must"][0]["simple_query_string"]["query"] == "hello"
    assert body["highlight"]["fields"]["text_full"]["fragment_size"] == 200
    assert body["sort"][0] == {"_score": {"order": "desc"}}

def test_build_query_with_filters_and_sort():
    body = build_query(
        "test",
        from_=5,
        size=20,
        sort="modified",
        filetypes=["pdf", "txt"],
        modified_from="2020-01-01",
        modified_to="2021-01-01",
        created_from="2019-01-01",
        path_prefix="/docs",
        size_gte=100,
        size_lte=2000,
        fragment_size=300,
        num_fragments=2,
    )
    filters = body["query"]["bool"]["filter"]
    assert {"terms": {"filetype": ["pdf", "txt"]}} in filters
    assert {"prefix": {"path": "/docs"}} in filters
    assert any("modified_at" in f.get("range", {}) for f in filters)
    assert body["sort"] == [{"modified_at": {"order": "desc"}}]
    assert body["from"] == 5
    assert body["size"] == 20
    assert body["highlight"]["fields"]["text_full"]["fragment_size"] == 300
