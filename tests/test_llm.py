import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.llm import ask_llm, get_available_models, load_model

print("Available models:", get_available_models())

print("Loading model...")
load_model("mistral-7b-instruct-v0.1.Q4_K_M.gguf")  # Replace with one that exists

print("Testing LLM...")
response = ask_llm("What is the capital of Germany?")
print("Response:", response)
