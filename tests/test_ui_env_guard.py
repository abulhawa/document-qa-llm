import importlib
import pytest

@pytest.mark.parametrize(
    "mod", ["ftfy", "langchain_core", "langchain_community", "langchain_text_splitters"]
)
def test_ui_does_not_require_ingest_libs(mod):
    with pytest.raises(Exception):
        importlib.import_module(mod)
