import importlib.util
import pathlib
import sys

from qa_pipeline.types import RetrievedDocument


_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_qa_handoff_eval.py"
)
_SPEC = importlib.util.spec_from_file_location("run_qa_handoff_eval", _SCRIPT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Failed to load run_qa_handoff_eval.py")
eval_script = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("run_qa_handoff_eval", eval_script)
_SPEC.loader.exec_module(eval_script)


def test_pack_docs_by_token_budget_respects_min_chunks():
    docs = [
        RetrievedDocument(text="a" * 400, path="doc-a", checksum="a"),
        RetrievedDocument(text="b" * 400, path="doc-b", checksum="b"),
        RetrievedDocument(text="c" * 400, path="doc-c", checksum="c"),
    ]

    packed, used = eval_script.pack_docs_by_token_budget(
        docs,
        token_budget=120,
        min_chunks=2,
    )

    assert [doc.checksum for doc in packed] == ["a", "b"]
    assert used > 120


def test_pack_docs_by_token_budget_greedily_adds_within_budget():
    docs = [
        RetrievedDocument(text="a" * 200, path="doc-a", checksum="a"),
        RetrievedDocument(text="b" * 200, path="doc-b", checksum="b"),
        RetrievedDocument(text="c" * 200, path="doc-c", checksum="c"),
    ]

    packed, _ = eval_script.pack_docs_by_token_budget(
        docs,
        token_budget=120,
        min_chunks=1,
    )

    assert [doc.checksum for doc in packed] == ["a", "b"]


def test_answer_classification_helpers():
    assert eval_script._is_error_answer("[LLM Error: timeout] service timed out")
    assert eval_script._is_fallback_answer("I don't know.")
    assert not eval_script._is_fallback_answer("The city is Berlin.")
