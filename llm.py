import requests
from typing import List, Union, Dict, Optional
from config import (
    logger,
    LLM_COMPLETION_ENDPOINT,
    LLM_CHAT_ENDPOINT,
    LLM_MODEL_LIST_ENDPOINT,
    LLM_MODEL_LOAD_ENDPOINT,
)

# Constants
TIMEOUT = 30  # seconds
STOP_TOKENS = ["</s>", "###", "---"]
PROMPT_LENGTH_WARN_THRESHOLD = 600


def get_available_models() -> List[str]:
    try:
        response = requests.get(LLM_MODEL_LIST_ENDPOINT, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json().get("model_names", [])
    except requests.RequestException as e:
        logger.error("Failed to get available models from LLM: %s", e)
        return []


def load_model(model_name: str) -> bool:
    try:
        response = requests.post(
            LLM_MODEL_LOAD_ENDPOINT, json={"model_name": model_name}, timeout=TIMEOUT
        )
        response.raise_for_status()
        logger.info("✅ Model '%s' loaded successfully.", model_name)
        return True
    except requests.RequestException as e:
        logger.error("❌ Error loading model '%s': %s", model_name, e)
        return False


def ask_llm(
    prompt: Union[str, List[Dict[str, str]]],
    mode: str = "completion",
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    """Unified LLM query function for chat and completion modes."""

    try:
        if mode == "chat":
            endpoint = LLM_CHAT_ENDPOINT
            payload = {
                "messages": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": STOP_TOKENS
            }
            if model:
                payload["model"] = model
            logger.info("🧠 Sending chat prompt (%d messages)", len(prompt))  # type: ignore

        else:  # mode == "completion"
            endpoint = LLM_COMPLETION_ENDPOINT
            prompt_words = len(prompt.split())  # type: ignore
            if prompt_words > PROMPT_LENGTH_WARN_THRESHOLD:
                logger.warning("🚨 Prompt is long (%d words). Consider trimming.", prompt_words)
            logger.info("🧠 Sending completion prompt (%d words)", prompt_words)

            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": STOP_TOKENS
            }
            if model:
                payload["model"] = model

        response = requests.post(endpoint, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if mode == "chat":
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            content = data.get("choices", [{}])[0].get("text", "").strip()

        if not content:
            logger.warning("⚠️ LLM response was empty or malformed: %s", response.text)

        for stop_token in STOP_TOKENS:
            if stop_token in content:
                logger.warning("🛑 LLM response stopped early due to stop token: %r", stop_token)
                break

        return content

    except requests.RequestException as e:
        logger.error("LLM request failed: %s", e)
        return "[LLM Error]"
