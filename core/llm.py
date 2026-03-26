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
    USE_GROQ,
    GROQ_API_KEY,
    GROQ_MODEL,
)
from core import llm_cache

# Constants
TIMEOUT = 30  # seconds
STOP_TOKENS = ["</s>"]
PROMPT_LENGTH_WARN_THRESHOLD = 2000


class LLMCallError(TypedDict):
    type: Literal["timeout", "http_error", "invalid_json", "request_exception", "empty_response"]
    status_code: int | None
    summary: str


class LLMCallResult(TypedDict):
    content: str
    error: LLMCallError | None
    prompt_length: int | None


def _request_headers() -> Dict[str, str] | None:
    if not USE_GROQ:
        return None
    headers = {"Content-Type": "application/json"}
    if GROQ_API_KEY:
        headers["Authorization"] = f"Bearer {GROQ_API_KEY}"
    return headers


def _extract_model_names(payload: Dict[str, Any]) -> List[str]:
    model_names = payload.get("model_names")
    if isinstance(model_names, list):
        return [str(model_name) for model_name in model_names]

    models_data = payload.get("data")
    if isinstance(models_data, list):
        names: List[str] = []
        for item in models_data:
            if isinstance(item, dict):
                model_id = item.get("id")
                if model_id:
                    names.append(str(model_id))
        return names

    return []


def _resolve_model(requested_model: Optional[str]) -> Optional[str]:
    if requested_model:
        return requested_model
    if USE_GROQ:
        return GROQ_MODEL or None
    return None


def _is_chat_response(mode: str) -> bool:
    # Groq compatibility: completion prompts are served through chat completions.
    return mode == "chat" or (USE_GROQ and mode == "completion")


def get_available_models() -> List[str]:
    """Fetch the list of available models from the LLM server."""
    span = get_current_span()
    if USE_GROQ and not GROQ_API_KEY:
        logger.warning("Groq mode enabled but GROQ_API_KEY is missing.")
        return [GROQ_MODEL] if GROQ_MODEL else []
    try:
        if USE_GROQ:
            response = requests.get(
                LLM_MODEL_LIST_ENDPOINT, timeout=5, headers=_request_headers()
            )
        else:
            response = requests.get(LLM_MODEL_LIST_ENDPOINT, timeout=5)
        span.set_attribute("llm.model_list.requested", True)
        response.raise_for_status()
        models = _extract_model_names(response.json())
        if USE_GROQ and not models and GROQ_MODEL:
            models = [GROQ_MODEL]
        span.set_attribute("llm.model_list.count", len(models))
        return models
    except requests.RequestException as e:
        logger.error("Failed to fetch model list: %s", e)
        span.set_attribute("llm.model_list.error", str(e))
        if USE_GROQ and GROQ_MODEL:
            return [GROQ_MODEL]
        return []


def get_loaded_model_name() -> str | None:
    """Return the name of the currently loaded model, or None."""
    span = get_current_span()
    if USE_GROQ:
        if not GROQ_API_KEY:
            span.set_attribute("llm.model.info_error", "missing_groq_api_key")
            return None
        span.set_attribute("llm.model.loaded", GROQ_MODEL or "")
        return GROQ_MODEL or None
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
    if USE_GROQ:
        if not GROQ_API_KEY:
            logger.error("Groq mode requires GROQ_API_KEY before calling load_model.")
            span.set_attribute("llm.model.load.error", "missing_groq_api_key")
            return False
        logger.info("Groq mode active; skipping local model load step.")
        span.set_attribute("llm.model.load.noop", True)
        return True
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
    if USE_GROQ and not GROQ_API_KEY:
        summary = "GROQ_API_KEY is required when USE_GROQ=true."
        logger.error(summary)
        return {
            "content": "",
            "error": {
                "type": "request_exception",
                "status_code": None,
                "summary": summary,
            },
            "prompt_length": None,
        }
    prompt_length = None
    cache_key: str | None = None
    canonical: Dict[str, Any] | None = None
    prompt_text: str | None = None
    prompt_hash: str | None = None
    model_id: str | None = None
    endpoint_id: str | None = None
    resolved_model = _resolve_model(model)
    try:
        if mode == "chat":
            endpoint = LLM_CHAT_ENDPOINT
            payload = {
                "messages": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": STOP_TOKENS,
            }
            if resolved_model:
                payload["model"] = resolved_model
            logger.info("🧠 Sending chat prompt (%d messages)", len(prompt))  # type: ignore
            prompt_length = len(str(prompt))
        else:  # mode == "completion"
            prompt_words = len(prompt.split())  # type: ignore
            if prompt_words > PROMPT_LENGTH_WARN_THRESHOLD:
                logger.warning(
                    "🚨 Prompt is long (%d words). Consider trimming.", prompt_words
                )
            logger.info("🧠 Sending completion prompt (%d words)", prompt_words)
            prompt_length = len(prompt)  # type: ignore[arg-type]
            if USE_GROQ:
                endpoint = LLM_CHAT_ENDPOINT
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": STOP_TOKENS,
                }
            else:
                endpoint = LLM_COMPLETION_ENDPOINT
                payload = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stop": STOP_TOKENS,
                }
            if resolved_model:
                payload["model"] = resolved_model

        if llm_cache.is_cache_enabled(use_cache):
            decoding_params = {
                k: v for k, v in payload.items() if k not in {"prompt", "messages", "model"}
            }
            model_id = resolved_model or "default"
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

        request_kwargs: Dict[str, Any] = {"json": payload, "timeout": TIMEOUT}
        headers = _request_headers()
        if headers is not None:
            request_kwargs["headers"] = headers
        response = requests.post(endpoint, **request_kwargs)
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
        span.set_attribute("llm.model", resolved_model or "default")
        span.set_attribute("llm.mode", mode)
        span.set_attribute("llm.temperature", temperature)
        span.set_attribute("llm.max_tokens", max_tokens)

        if _is_chat_response(mode):
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

    if USE_GROQ:
        if not GROQ_API_KEY:
            result["status_message"] = "Groq API key is missing."
            return result
        try:
            resp = requests.get(
                LLM_MODEL_LIST_ENDPOINT, timeout=timeout, headers=_request_headers()
            )
            resp.raise_for_status()
            data = resp.json()
            models = _extract_model_names(data)
            model_name = GROQ_MODEL or (models[0] if models else None)

            result["server_online"] = True
            result["current_model"] = model_name
            if model_name:
                result["model_loaded"] = True
                result["status_message"] = "LLM is ready!"
                result["active"] = True
            else:
                result["status_message"] = "Groq is reachable but no model is configured."
        except requests.RequestException as e:
            logger.warning(f"Groq endpoint unreachable: {e}")
            result["status_message"] = "Groq endpoint is unreachable."
        return result

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
