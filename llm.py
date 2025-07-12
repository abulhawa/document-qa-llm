import requests
from config import logger, LLM_API_URL

def ask_llm(prompt: str) -> str:
    logger.info("Sending prompt to TGW (/api/v1/generate): %d chars", len(prompt))
    try:
        response = requests.post(LLM_API_URL, json={
            "prompt": prompt,
            "max_new_tokens": 512,
            "temperature": 0.7,
            "stop": ["</s>", "###", "---"]
        })
        response.raise_for_status()
        result = response.json()
        return result["results"][0]["text"].strip()
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        return "⚠️ Error: Failed to get response from LLM."

def get_model_status():
    try:
        r = requests.get("http://localhost:5000/v1/models")
        r.raise_for_status()
        data = r.json()
        if data.get("data"):
            return data["data"][0]["id"]
    except Exception as e:
        logger.warning("Could not get model status: %s", e)
    return None

def get_available_models():
    try:
        r = requests.get("http://localhost:5000/v1/internal/model/list")
        r.raise_for_status()
        return r.json().get("model_names", [])
    except Exception as e:
        logger.warning("Could not fetch available models: %s", e)
        return []

def load_model(model_name):
    try:
        r = requests.post("http://localhost:5000/v1/internal/model/load", json={"model_name": model_name})
        r.raise_for_status()
        return True
    except Exception as e:
        logger.warning("Could not load model %s: %s", model_name, e)
        return False
