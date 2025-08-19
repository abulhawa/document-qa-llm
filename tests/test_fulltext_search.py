from utils.fulltext_search import build_query

def test_build_query_basic():
    body = build_query("hello", fragment_size=200)

    # query string check
    must = body["query"]["function_score"]["query"]["bool"]["must"]
    assert must[0]["simple_query_string"]["query"] == "hello"

    # highlight default fragment_size
    assert body["highlight"]["fields"]["text_full"]["fragment_size"] == 200

    # default sort order
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
        path_contains="docs",
        size_gte=100,
        size_lte=2000,
        fragment_size=300,
        num_fragments=2,
    )

    # bool filters are inside function_score
    bool_filters = body["query"]["function_score"]["query"]["bool"]["filter"]

    # file types live in post_filter so aggs don't shrink
    post_filters = body.get("post_filter", {}).get("bool", {}).get("filter", [])

    # file type terms are applied via post_filter
    assert {"terms": {"filetype": ["pdf", "txt"]}} in post_filters

    # path prefix & ranges remain in the main bool filter
    assert any("path.ngram" in f.get("match", {}) for f in bool_filters)
    assert any("modified_at" in f.get("range", {}) for f in bool_filters)
    assert any("created_at" in f.get("range", {}) for f in bool_filters)
    assert any("size_bytes" in f.get("range", {}) for f in bool_filters)

    # sort & paging
    assert body["sort"] == [{"modified_at": {"order": "desc"}}]
    assert body["from"] == 5
    assert body["size"] == 20

    # highlight settings
    assert body["highlight"]["fields"]["text_full"]["fragment_size"] == 300
    assert body["highlight"]["fields"]["text_full"]["number_of_fragments"] == 2

    # sanity: totals are tracked
    assert body["track_total_hits"] is True

