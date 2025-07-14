import requests
from typing import List
from config import (
    logger,
    LLM_COMPLETION_ENDPOINT,
    LLM_MODEL_LIST_ENDPOINT,
    LLM_MODEL_LOAD_ENDPOINT,
)

TIMEOUT = 30  # seconds
STOP_TOKENS = ["</s>", "###", "---"]
PROMPT_LENGTH_WARN_THRESHOLD = 600  # Warn if prompt exceeds this many words

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
        response = requests.post(
            LLM_MODEL_LOAD_ENDPOINT, json={"model_name": model_name}, timeout=TIMEOUT
        )
        response.raise_for_status()
        logger.info("✅ Model '%s' loaded successfully.", model_name)
        return True
    except requests.RequestException as e:
        logger.error("❌ Error loading model '%s': %s", model_name, e)
        return False


def ask_llm(prompt: str) -> str:
    """Send prompt to local LLM and return the generated response."""

    prompt_words = len(prompt.split())
    print(f"prompt_words: {prompt_words}")
    print(f"prompt: {prompt}")
    if prompt_words > PROMPT_LENGTH_WARN_THRESHOLD:
        print("\n" + "=" * 80)
        print(
            f"🚨 WARNING: Prompt is very long ({prompt_words} words). Consider trimming context."
        )
        print("=" * 80 + "\n")

    try:
        response = requests.post(
            LLM_COMPLETION_ENDPOINT,
            json={
                "prompt": prompt,
                "max_new_tokens": 512,
                "temperature": 0.7,
                "stop": STOP_TOKENS,
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()

        print(response.json())  # Debugging line to see the full response
        text = response.json().get("choices", [{}])[0].get("text", "").strip()
        if not text:
            logger.warning("LLM response was empty or malformed: %s", response.text)
        
        for stop_token in STOP_TOKENS:
            if stop_token in text:
                logger.warning("🛑 LLM response stopped early due to stop token: %r", stop_token)
                break
            
        return text

    except requests.RequestException as e:
        logger.error("LLM request failed for prompt: %s | Error: %s", prompt, e)
        return "[LLM Error]"
