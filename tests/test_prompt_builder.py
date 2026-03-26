from qa_pipeline.prompt_builder import (
    CHAT_SYSTEM_PROMPT,
    STRICT_QA_INSTRUCTIONS,
    build_prompt,
)
from qa_pipeline.types import RetrievalResult, RetrievedDocument


def _retrieval_result() -> RetrievalResult:
    return RetrievalResult(
        query="q",
        documents=[
            RetrievedDocument(text="alpha context", path="doc-a"),
            RetrievedDocument(text="beta context", path="doc-b"),
        ],
    )


def test_build_prompt_chat_uses_strict_system_prompt():
    req = build_prompt(
        _retrieval_result(),
        "What is alpha?",
        mode="chat",
        chat_history=[{"role": "assistant", "content": "Earlier answer"}],
    )

    assert req.mode == "chat"
    assert isinstance(req.prompt, list)
    assert req.prompt[0] == {"role": "system", "content": CHAT_SYSTEM_PROMPT}
    assert req.prompt[0]["content"] == STRICT_QA_INSTRUCTIONS
    assert req.prompt[1] == {"role": "assistant", "content": "Earlier answer"}
    assert req.prompt[-1]["role"] == "user"
    assert "Context:\n[1] Source: doc-a\nalpha context\n\n[2] Source: doc-b\nbeta context" in req.prompt[-1]["content"]
    assert "Question: What is alpha?" in req.prompt[-1]["content"]


def test_build_prompt_completion_includes_same_instructions():
    req = build_prompt(_retrieval_result(), "What is beta?", mode="completion")

    assert req.mode == "completion"
    assert isinstance(req.prompt, str)
    assert f"Instructions:\n{STRICT_QA_INSTRUCTIONS}" in req.prompt
    assert "Context:\n[1] Source: doc-a\nalpha context\n\n[2] Source: doc-b\nbeta context" in req.prompt
    assert "Question: What is beta?" in req.prompt
    assert req.prompt.endswith("Answer:")


def test_build_prompt_includes_source_metadata_when_present():
    retrieval = RetrievalResult(
        query="q",
        documents=[
            RetrievedDocument(
                text="cv details",
                path="john_cv.pdf",
                doc_type="cv",
                person_name="John Doe",
                authority_rank=1.0,
            )
        ],
    )

    req = build_prompt(retrieval, "Who is this?", mode="completion")

    assert isinstance(req.prompt, str)
    assert (
        "Context:\n[1] Source: john_cv.pdf | type=cv | person=John Doe | authority=1.00\ncv details"
        in req.prompt
    )
