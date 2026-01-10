import requests
from typing import Any, List, Union, Dict, Optional, TypedDict, Literal
from tracing import get_current_span
from config import (
    logger,
    LLM_BASE_URL,
    LLM_COMPLETION_ENDPOINT,
    LLM_CHAT_ENDPOINT,
    LLM_MODEL_LIST_ENDPOINT,
    LLM_MODEL_LOAD_ENDPOINT,
    LLM_MODEL_INFO_ENDPOINT,
)
from core import llm_cache

# Constants
TIMEOUT = 30  # seconds
STOP_TOKENS = ["</s>", "###", "---"]
PROMPT_LENGTH_WARN_THRESHOLD = 600


class LLMCallError(TypedDict):
    type: Literal["timeout", "http_error", "invalid_json", "request_exception", "empty_response"]
    status_code: int | None
    summary: str


class LLMCallResult(TypedDict):
    content: str
    error: LLMCallError | None
    prompt_length: int | None


def get_available_models() -> List[str]:
    """Fetch the list of available models from the LLM server."""
    span = get_current_span()
    try:
        response = requests.get(LLM_MODEL_LIST_ENDPOINT, timeout=5)
        span.set_attribute("llm.model_list.requested", True)
        response.raise_for_status()
        models = response.json().get("model_names", [])
        span.set_attribute("llm.model_list.count", len(models))
        return models
    except requests.RequestException as e:
        logger.error("Failed to fetch model list: %s", e)
        span.set_attribute("llm.model_list.error", str(e))
        return []


def get_loaded_model_name() -> str | None:
    """Return the name of the currently loaded model, or None."""
    span = get_current_span()
    try:
        response = requests.get(LLM_MODEL_INFO_ENDPOINT, timeout=5)
        if response.status_code == 200:
            model_name = response.json().get("model_name")
            span.set_attribute("llm.model.loaded", model_name)
            if model_name and model_name.lower() != "none":
                return model_name
        else:
            span.set_attribute("llm.model.info_error", response.text)
    except requests.RequestException as e:
        logger.error("Failed to retrieve loaded model name: %s", e)
        span.set_attribute("llm.model.info_exception", str(e))
        return "Error connecting to server"
    return None


def is_model_loaded() -> bool:
    return get_loaded_model_name() is not None


def load_model(model_name: str) -> bool:
    span = get_current_span()
    span.set_attribute("llm.model.load.requested", model_name)
    try:
        response = requests.post(
            LLM_MODEL_LOAD_ENDPOINT, json={"model_name": model_name}, timeout=TIMEOUT
        )
        response.raise_for_status()
        logger.info("✅ Model '%s' loaded successfully.", model_name)
        span.set_attribute("llm.model.load.success", True)
        return True
    except requests.RequestException as e:
        logger.error("❌ Error loading model '%s': %s", model_name, e)
        span.set_attribute("llm.model.load.error", str(e))
        return False


def ask_llm_with_status(
    prompt: Union[str, List[Dict[str, str]]],
    mode: str = "completion",
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> LLMCallResult:
    """Unified LLM query function for chat and completion modes."""

    span = get_current_span()
    prompt_length = None
    cache_key: str | None = None
    canonical: Dict[str, Any] | None = None
    prompt_text: str | None = None
    prompt_hash: str | None = None
    model_id: str | None = None
    endpoint_id: str | None = None
    try:
        if mode == "chat":
            endpoint = LLM_CHAT_ENDPOINT
            payload = {
                "messages": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": STOP_TOKENS,
            }
            if model:
                payload["model"] = model
            logger.info("🧠 Sending chat prompt (%d messages)", len(prompt))  # type: ignore
            prompt_length = len(str(prompt))
        else:  # mode == "completion"
            endpoint = LLM_COMPLETION_ENDPOINT
            prompt_words = len(prompt.split())  # type: ignore
            if prompt_words > PROMPT_LENGTH_WARN_THRESHOLD:
                logger.warning(
                    "🚨 Prompt is long (%d words). Consider trimming.", prompt_words
                )
            logger.info("🧠 Sending completion prompt (%d words)", prompt_words)
            prompt_length = len(prompt)  # type: ignore[arg-type]
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": STOP_TOKENS,
            }
            if model:
                payload["model"] = model

        if llm_cache.is_cache_enabled(use_cache):
            decoding_params = {
                k: v for k, v in payload.items() if k not in {"prompt", "messages", "model"}
            }
            model_id = model or "default"
            endpoint_id = f"{LLM_BASE_URL}:{mode}"
            (
                cache_key,
                canonical,
                prompt_text,
                prompt_hash,
                _system_prompt,
            ) = llm_cache.build_cache_key(
                prompt=prompt,
                mode=mode,
                model_id=model_id,
                endpoint_id=endpoint_id,
                decoding_params=decoding_params,
            )
            cached = llm_cache.get_cached_response(cache_key)
            if cached is not None:
                logger.info("🧠 LLM cache hit")
                return {"content": cached, "error": None, "prompt_length": prompt_length}
            logger.info("🧠 LLM cache miss")

        response = requests.post(endpoint, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            logger.error("LLM response JSON parse failed: %s", exc)
            return {
                "content": "",
                "error": {
                    "type": "invalid_json",
                    "status_code": response.status_code,
                    "summary": str(exc),
                },
                "prompt_length": prompt_length,
            }
        span.set_attribute("llm.prompt_length", prompt_length or 0)
        span.set_attribute("llm.model", model or "default")
        span.set_attribute("llm.mode", mode)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.max_tokens", max_tokens)

        if mode == "chat":
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        else:
            content = data.get("choices", [{}])[0].get("text", "").strip()
        span.set_attribute("llm.response_length", len(content))

        if not content:
            logger.warning("⚠️ LLM response was empty or malformed: %s", response.text)
            return {
                "content": "",
                "error": {
                    "type": "empty_response",
                    "status_code": response.status_code,
                    "summary": "Empty LLM response",
                },
                "prompt_length": prompt_length,
            }

        for stop_token in STOP_TOKENS:
            if stop_token in content:
                logger.warning(
                    "🛑 LLM response stopped early due to stop token: %r", stop_token
                )
                break

        if (
            cache_key
            and canonical is not None
            and prompt_text is not None
            and prompt_hash is not None
            and model_id is not None
            and endpoint_id is not None
            and llm_cache.is_cache_enabled(use_cache)
        ):
            llm_cache.store_cache_entry(
                cache_key=cache_key,
                canonical=canonical,
                prompt_text=prompt_text,
                prompt_hash=prompt_hash,
                response_text=content,
                model_id=model_id,
                endpoint_id=endpoint_id,
                status="ok",
            )
            logger.info("🧠 LLM cache store")

        return {"content": content, "error": None, "prompt_length": prompt_length}

    except requests.Timeout as e:
        logger.error("LLM request timed out: %s", e)
        return {
            "content": "",
            "error": {"type": "timeout", "status_code": None, "summary": str(e)},
            "prompt_length": prompt_length,
        }
    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else None
        logger.error("LLM request failed (HTTP %s): %s", status_code, e)
        return {
            "content": "",
            "error": {
                "type": "http_error",
                "status_code": status_code,
                "summary": str(e),
            },
            "prompt_length": prompt_length,
        }
    except requests.RequestException as e:
        logger.error("LLM request failed: %s", e)
        return {
            "content": "",
            "error": {
                "type": "request_exception",
                "status_code": None,
                "summary": str(e),
            },
            "prompt_length": prompt_length,
        }


def ask_llm(
    prompt: Union[str, List[Dict[str, str]]],
    mode: str = "completion",
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    use_cache: bool = True,
) -> str:
    """Unified LLM query function for chat and completion modes."""
    result = ask_llm_with_status(
        prompt,
        mode=mode,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        use_cache=use_cache,
    )
    if result["error"]:
        return "[LLM Error]"
    return result["content"]


class LLMStatus(TypedDict):
    server_online: bool
    model_loaded: bool
    current_model: str | None
    status_message: str
    active: bool


def check_llm_status(timeout: float = 0.3) -> LLMStatus:
    """Check if LLM server is online and a model is loaded.
    Returns:
        Dict with keys:
            - server_online: bool
            - model_loaded: bool
            - current_model: str or None
            - status_message: str
            - active: bool (server and model are ready)
    """
    result: LLMStatus = {
        "server_online": False,
        "model_loaded": False,
        "current_model": None,
        "status_message": "LLM server is offline!",
        "active": False,
    }
    try:
        resp = requests.get(LLM_MODEL_INFO_ENDPOINT, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            model_name = data.get("model_name")

            result["server_online"] = True
            result["current_model"] = model_name
            result["status_message"] = "No LLM model is loaded!"

            if model_name and model_name.lower() != "none":
                result["model_loaded"] = True
                result["status_message"] = "LLM is ready!"
                result["active"] = True
    except requests.RequestException as e:
        logger.warning(f"LLM server unreachable: {e}")
    return result
