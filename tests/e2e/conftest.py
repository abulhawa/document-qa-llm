# tests/e2e/conftest.py

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

import pytest

ART_DIR = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    # Let pytest run the test and get the report
    outcome = yield
    rep = outcome.get_result()
    # Stash the report on the item so fixtures can inspect it
    setattr(item, "rep_" + rep.when, rep)

@pytest.fixture(autouse=True)
def _dump_on_failure(request):
    yield
    # After the test: if it used Playwright and failed at call stage, dump artifacts
    if "page" in request.fixturenames and getattr(request.node, "rep_call", None) and request.node.rep_call.failed:
        page = request.getfixturevalue("page")
        ART_DIR.mkdir(parents=True, exist_ok=True)
        testname = request.node.name.replace("/", "_")
        page.screenshot(path=str(ART_DIR / f"{testname}.png"), full_page=True)
        (ART_DIR / f"{testname}.html").write_text(page.content(), encoding="utf-8")
# ─────────────────────────────
# Minimal threaded mock LLM
# ─────────────────────────────

STUB_TEXT = "Stub answer."

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args, **_kwargs):  # silence default logs
        return

    def _read_json(self):
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    def _send_json(self, obj: dict, code: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/internal/model/info":
            self._send_json({"model_name": "stub-model"}); return
        if self.path == "/v1/internal/model/list":
            self._send_json({"data": [{"id": "stub-model"}]}); return
        self.send_response(404); self.end_headers()

    def do_POST(self):
        _ = self._read_json()  # drain body

        if self.path == "/v1/chat/completions":
            self._send_json({
                "id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": "stub-model",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": STUB_TEXT}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 2},
            }); return

        if self.path == "/v1/completions":
            self._send_json({
                "id": "cmpl-stub", "object": "text_completion", "created": 0, "model": "stub-model",
                "choices": [{"index": 0, "text": STUB_TEXT, "logprobs": None, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 2, "total_tokens": 2},
            }); return

        if self.path == "/api/v1/generate":
            self._send_json({"results": [{"text": STUB_TEXT}]}); return

        if self.path == "/v1/internal/model/load":
            self._send_json({"status": "ok", "model_name": "stub-model"}); return

        self.send_response(404); self.end_headers()


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http_ok(url: str, timeout_s: float = 5.0):
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5):
                return
        except Exception as e:
            last_err = e
            time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for {url}. Last error: {last_err}")


def _repo_root() -> Path:
    # this file is tests/e2e/conftest.py → repo root is 2 levels up
    return Path(__file__).resolve().parents[2]

# ─────────────────────────────
# Fixtures
# ─────────────────────────────

@pytest.fixture(scope="session")
def llm_stub_base():
    """Start the mock LLM on a free port and return base URL."""
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True); t.start()

    base = f"http://127.0.0.1:{port}"
    _wait_http_ok(f"{base}/v1/internal/model/info", timeout_s=5.0)

    yield base

    server.shutdown()
    t.join(timeout=3)


@pytest.fixture(scope="session")
def streamlit_app(llm_stub_base):
    """Launch Streamlit with env pointing to the mock; return app URL."""
    root = _repo_root()
    entry = root / os.getenv("STREAMLIT_APP", "main.py")  # change if your entry is different
    assert entry.exists(), f"Streamlit entry not found: {entry}"

    app_port = _free_port()
    app_url = f"http://127.0.0.1:{app_port}"

    env = os.environ.copy()
    base = llm_stub_base
    env.update({
        "LLM_BASE": base,
        "LLM_GENERATE_ENDPOINT":   f"{base}/api/v1/generate",
        "LLM_COMPLETION_ENDPOINT": f"{base}/v1/completions",
        "LLM_CHAT_ENDPOINT":       f"{base}/v1/chat/completions",
        "LLM_MODEL_LIST_ENDPOINT": f"{base}/v1/internal/model/list",
        "LLM_MODEL_LOAD_ENDPOINT": f"{base}/v1/internal/model/load",
        "LLM_MODEL_INFO_ENDPOINT": f"{base}/v1/internal/model/info",
        # Streamlit stability
        "PYTHONUNBUFFERED": "1",
        "BROWSER": "none",
        "STREAMLIT_SERVER_FILEWATCHERTYPE": "none",
        "STREAMLIT_SERVER_ADDRESS": "127.0.0.1",
    })
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", str(entry),
            "--server.headless", "true",
            "--server.port", str(app_port),
            "--server.address", "127.0.0.1",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false",
        ],
        cwd=str(root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # wait until serving
    start = time.time()
    timeout = 90 if os.name == "nt" else 45
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(app_url, timeout=0.5):
                break
        except Exception:
            if proc.poll() is not None:
                if proc.stdout:
                    print("==== Streamlit stdout (crashed) ====")
                    print(proc.stdout.read())
                raise RuntimeError("Streamlit exited before ready.")
            time.sleep(0.2)
    else:
        if proc.stdout:
            print("==== Streamlit stdout (timeout) ====")
            print(proc.stdout.read())
        raise RuntimeError("Streamlit app failed to start in time.")

    yield app_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


# ─────────────────────────────
# Playwright timeouts
# (assumes pytest-playwright is installed)
# ─────────────────────────────
@pytest.fixture(autouse=True)
def _playwright_timeouts(page, context):
    context.set_default_timeout(3000)
    context.set_default_navigation_timeout(30000)
    page.set_default_timeout(3000)
    page.set_default_navigation_timeout(30000)
