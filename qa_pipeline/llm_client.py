from core.llm import ask_llm_with_status
from qa_pipeline.types import PromptRequest


def _format_llm_error(error: dict) -> str:
    error_type = str(error.get("type") or "unknown_error")
    status_code = error.get("status_code")
    summary = str(error.get("summary") or "Unknown LLM error.")
    status_part = f", HTTP {status_code}" if status_code is not None else ""
    return f"[LLM Error: {error_type}{status_part}] {summary}"


def generate_answer(
    prompt_request: PromptRequest,
    temperature: float,
    model: str | None,
    use_cache: bool = True,
) -> str:
    if prompt_request.mode == "chat":
        result = ask_llm_with_status(
            prompt=prompt_request.prompt,
            mode="chat",
            temperature=temperature,
            model=model,
            use_cache=use_cache,
        )
    else:
        result = ask_llm_with_status(
            prompt=prompt_request.prompt,
            mode="completion",
            temperature=temperature,
            model=model,
            use_cache=use_cache,
        )

    if result["error"]:
        return _format_llm_error(result["error"])

    return result["content"]
