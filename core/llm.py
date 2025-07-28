import requests
import json
from typing import List, Union, Dict, Optional
from tracing import get_current_span, record_span_error, OUTPUT_VALUE
from config import (
    logger,
    LLM_COMPLETION_ENDPOINT,
    LLM_CHAT_ENDPOINT,
    LLM_MODEL_LIST_ENDPOINT,
    LLM_MODEL_LOAD_ENDPOINT,
    LLM_MODEL_INFO_ENDPOINT,
)

# Constants
TIMEOUT = 30  # seconds
STOP_TOKENS = ["</s>", "###", "---"]
PROMPT_LENGTH_WARN_THRESHOLD = 600


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


def ask_llm(
    prompt: Union[str, List[Dict[str, str]]],
    mode: str = "completion",
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """Unified LLM query function for chat and completion modes."""

    span = get_current_span()
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

        else:  # mode == "completion"
            endpoint = LLM_COMPLETION_ENDPOINT
            prompt_words = len(prompt.split())  # type: ignore
            if prompt_words > PROMPT_LENGTH_WARN_THRESHOLD:
                logger.warning(
                    "🚨 Prompt is long (%d words). Consider trimming.", prompt_words
                )
            logger.info("🧠 Sending completion prompt (%d words)", prompt_words)

            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": STOP_TOKENS,
            }
            if model:
                payload["model"] = model

        response = requests.post(endpoint, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        span.set_attribute("llm.prompt_length", len(prompt))
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

        for stop_token in STOP_TOKENS:
            if stop_token in content:
                logger.warning(
                    "🛑 LLM response stopped early due to stop token: %r", stop_token
                )
                break

        return content

    except requests.RequestException as e:
        logger.error("LLM request failed: %s", e)
        return "[LLM Error]"


def check_llm_status(timeout: float = 0.3) -> Dict[str, Optional[str]]:
    """Check if LLM server is online and a model is loaded.

    Returns:
        Dict with keys:
            - server_online: bool
            - model_loaded: bool
            - current_model: str or None
            - status_message: str
            - active: bool (server and model are ready)
    """
    span = get_current_span()
    result = {
        "server_online": False,
        "model_loaded": False,
        "current_model": None,
        "status_message": "LLM server is offline!",
        "active": False,
    }

    span.set_attribute("llm.check.timeout", timeout)

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
        else:
            span.set_attribute("llm.check.bad_status", resp.status_code)

    except requests.RequestException as e:
        logger.warning(f"LLM server unreachable: {e}")
        record_span_error(span, e)

    # Trace the result
    span.set_attribute("llm.server_online", result["server_online"])
    span.set_attribute("llm.model_loaded", result["model_loaded"])
    span.set_attribute("llm.model_name", result["current_model"] or "None")
    span.set_attribute(OUTPUT_VALUE, json.dumps(result))

    return result
