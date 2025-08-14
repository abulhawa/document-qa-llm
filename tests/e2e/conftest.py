import subprocess
import time
import socket
from pathlib import Path

import requests
import pytest


def _get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def streamlit_app() -> str:
    """Launch the Streamlit app on a free port and yield its base URL."""
    port = _get_free_port()
    repo_root = Path(__file__).resolve().parents[2]
    cmd = ["streamlit", "run", "main.py", "--server.port", str(port)]
    proc = subprocess.Popen(cmd, cwd=repo_root)

    base_url = f"http://localhost:{port}"
    for _ in range(60):
        try:
            resp = requests.get(base_url)
            if "Document QA" in resp.text:
                break
        except requests.ConnectionError:
            pass
        time.sleep(0.5)
    else:
        proc.terminate()
        raise RuntimeError("Streamlit app did not start")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
