import requests
from typing import List
from config import (
    logger,
    LLM_COMPLETION_ENDPOINT,
    LLM_MODEL_LIST_ENDPOINT,
    LLM_MODEL_LOAD_ENDPOINT,
)

TIMEOUT = 30  # seconds


def get_available_models() -> List[str]:
    """Return list of available models from the LLM server."""
    try:
        response = requests.get(LLM_MODEL_LIST_ENDPOINT, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json().get("model_names", [])
    except requests.RequestException as e:
        logger.error("Failed to get available models from LLM: %s", e)
        return []


def load_model(model_name: str) -> bool:
    """Trigger model loading on the LLM server."""
    try:
        response = requests.post(LLM_MODEL_LOAD_ENDPOINT, json={"model_name": model_name}, timeout=TIMEOUT)
        response.raise_for_status()
        logger.info("✅ Model '%s' loaded successfully.", model_name)
        return True
    except requests.RequestException as e:
        logger.error("❌ Error loading model '%s': %s", model_name, e)
        return False


def ask_llm(prompt: str) -> str:
    """Send prompt to local LLM and return the generated response."""
    try:
        response = requests.post(LLM_COMPLETION_ENDPOINT, json={
            "prompt": prompt,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "stop": [
                "</s>",
                "###",
                "---"
            ]
        }, timeout=TIMEOUT)
        response.raise_for_status()

        print(response.json())  # Debugging line to see the full response
        text = response.json().get("choices", [{}])[0].get("text", "").strip()
        if not text:
            logger.warning("LLM response was empty or malformed: %s", response.text)
        return text

    except requests.RequestException as e:
        logger.error("LLM request failed for prompt: %s | Error: %s", prompt, e)
        return "[LLM Error]"
