from core.llm import ask_llm
from qa_pipeline.types import PromptRequest


def generate_answer(
    prompt_request: PromptRequest,
    temperature: float,
    model: str | None,
    use_cache: bool = True,
) -> str:
    if prompt_request.mode == "chat":
        return ask_llm(
            prompt=prompt_request.prompt,
            mode="chat",
            temperature=temperature,
            model=model,
            use_cache=use_cache,
        )

    return ask_llm(
        prompt=prompt_request.prompt,
        mode="completion",
        temperature=temperature,
        model=model,
        use_cache=use_cache,
    )
