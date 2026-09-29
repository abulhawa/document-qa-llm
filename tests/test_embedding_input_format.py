import pytest

from embedder_api_multilingual.input_format import prepare_embedding_texts


def test_e5_roles_add_expected_prefixes():
    assert prepare_embedding_texts(
        ["alpha"], input_type="passage", input_format="e5"
    ) == ["passage: alpha"]
    assert prepare_embedding_texts(
        ["alpha"], input_type="query", input_format="e5"
    ) == ["query: alpha"]


def test_e5_role_prefix_is_not_duplicated():
    assert prepare_embedding_texts(
        ["passage: alpha"], input_type="passage", input_format="e5"
    ) == ["passage: alpha"]


def test_raw_input_format_preserves_text():
    assert prepare_embedding_texts(
        ["alpha"], input_type="query", input_format="raw"
    ) == ["alpha"]


def test_unknown_input_format_fails():
    with pytest.raises(ValueError):
        prepare_embedding_texts(["alpha"], input_type="passage", input_format="bad")
