import subprocess
import json
import streamlit as st

def run_picker(mode: str) -> list[str]:
    try:
        result = subprocess.run(
            ["python", "file_picker.py", mode],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except Exception as e:
        st.error(f"❌ Picker failed: {e}")
        return []

def run_file_picker() -> list[str]:
    return run_picker("files")

def run_folder_picker() -> list[str]:
    return run_picker("folder")

def run_root_picker() -> list[str]:
    return run_picker("root")
