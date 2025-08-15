# tests/e2e/conftest.py
import os
import sys
import json
import time
import socket
import threading
import subprocess
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests
from playwright.sync_api import sync_playwright
from playwright._impl._errors import Error as PlaywrightError


def _get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ───────────────────────────────
# Mock LLM (no external deps)
# ───────────────────────────────
class MockLLMHandler(BaseHTTPRequestHandler):
    def _send(self, status=200, payload=None):
        body = json.dumps(payload or {}).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/internal/model/info":
            return self._send(200, {"model_name": "mock-llm", "loaded": True})
        if self.path == "/v1/internal/model/list":
            return self._send(200, {"model_names": ["mock-llm"]})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        # drain body (ignore)
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n:
                self.rfile.read(n)
        except Exception:
            pass

        if self.path == "/v1/completions":
            return self._send(200, {"choices": [{"text": "Generic response"}]})

        if self.path == "/v1/chat/completions":
            # Answer shape friendly to your chat UI
            return self._send(200, {
                "id": "cmpl-mock",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant",
                                "content": {"rewritten": "mock rewritten query"}},
                    "finish_reason": "stop"
                }],
                "model": "mock-llm"
            })

        return self._send(404, {"error": "not found"})

    def log_message(self, *args, **kwargs):
        # silence noisy HTTPServer logs
        return


@pytest.fixture(scope="session")
def mock_llm_server():
    """
    Start a tiny mock LLM server.

    - In CI: bind to :5000 (the app is launched by workflow and expects that).
    - Locally: bind to a random free port and we'll pass env vars to the app proc.
    """
    if os.getenv("CI") == "true":
        host, port = "127.0.0.1", 5000
    else:
        host, port = "127.0.0.1", _get_free_port()

    server = HTTPServer((host, port), MockLLMHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Wait briefly for readiness
    import urllib.request
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://{host}:{port}/v1/internal/model/info", timeout=0.2)
            break
        except Exception:
            time.sleep(0.05)

    yield (host, port)

    server.shutdown()


# ───────────────────────────────
# Streamlit app launcher
# ───────────────────────────────
@pytest.fixture(scope="session")
def streamlit_app(mock_llm_server) -> str:
    """
    Yield the base URL of the app.

    - In CI: app is already started by the workflow on :8501 → just return that URL.
    - Locally: start the app on a random free port and *inject LLM endpoints*
      via environment variables that point to the mock server.
    """
    if os.getenv("CI") == "true":
        # The workflow already: streamlit run main.py --server.port 8501
        return os.getenv("E2E_APP_URL", "http://127.0.0.1:8501")

    # Local run: we fully control the process & env
    app_port = _get_free_port()
    base_url = f"http://127.0.0.1:{app_port}"

    llm_host, llm_port = mock_llm_server
    llm_base = f"http://{llm_host}:{llm_port}"
    print('--------------------------------------')
    print(llm_base)
    print('--------------------------------------')
    env = os.environ.copy()
    env.setdefault("BROWSER", "none")  # don't open a real browser
    # Wire all endpoints to our mock (no import from config.py needed)
    env["LLM_BASE_URL"] = llm_base
    env["LLM_MODEL_INFO_ENDPOINT"] = f"{llm_base}/v1/internal/model/info"
    env["LLM_MODEL_LIST_ENDPOINT"] = f"{llm_base}/v1/internal/model/list"
    env["LLM_MODEL_LOAD_ENDPOINT"] = f"{llm_base}/v1/internal/model/load"
    env["LLM_CHAT_ENDPOINT"] = f"{llm_base}/v1/chat/completions"
    env["LLM_COMPLETION_ENDPOINT"] = f"{llm_base}/v1/completions"
    env["LLM_GENERATE_ENDPOINT"] = f"{llm_base}/api/v1/generate"
    
    proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            "main.py",
            "--server.headless", "true",
            "--server.port", str(app_port),
        ],
        env=env,
    )

    # Wait until responsive
    for _ in range(120):
        try:
            r = requests.get(base_url, timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("Streamlit did not start in time")

    yield base_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ───────────────────────────────
# Playwright fixtures
# ───────────────────────────────
@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError:
            # Last-resort: install missing bits on developer machines
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context()
    pg = context.new_page()
    pg.console_logs = []
    pg.on("console", lambda msg: pg.console_logs.append(f"{msg.type}: {msg.text}"))
    yield pg
    context.close()


# Save screenshot + console logs on failure
@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(autouse=True)
def capture_artifacts(request, page):
    yield
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        out = Path("artifacts"); out.mkdir(exist_ok=True)
        page.screenshot(path=str(out / f"{request.node.name}.png"))
        (out / f"{request.node.name}.log").write_text("\n".join(getattr(page, "console_logs", [])))
