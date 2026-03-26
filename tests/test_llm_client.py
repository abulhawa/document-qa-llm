from qa_pipeline.llm_client import generate_answer
from qa_pipeline.types import PromptRequest


def test_generate_answer_returns_content(monkeypatch):
    def fake_ask_llm_with_status(**kwargs):
        assert kwargs["mode"] == "completion"
        return {"content": "final answer", "error": None, "prompt_length": 12}

    monkeypatch.setattr(
        "qa_pipeline.llm_client.ask_llm_with_status", fake_ask_llm_with_status
    )

    result = generate_answer(
        PromptRequest(prompt="p", mode="completion"),
        temperature=0.1,
        model=None,
    )

    assert result == "final answer"


def test_generate_answer_returns_descriptive_error(monkeypatch):
    def fake_ask_llm_with_status(**kwargs):
        assert kwargs["mode"] == "completion"
        return {
            "content": "",
            "error": {
                "type": "http_error",
                "status_code": 401,
                "summary": "Unauthorized",
            },
            "prompt_length": 12,
        }

    monkeypatch.setattr(
        "qa_pipeline.llm_client.ask_llm_with_status", fake_ask_llm_with_status
    )

    result = generate_answer(
        PromptRequest(prompt="p", mode="completion"),
        temperature=0.1,
        model=None,
    )

    assert result == "[LLM Error: http_error, HTTP 401] Unauthorized"
