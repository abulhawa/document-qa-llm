import os, socket, threading, pytest, sys
import subprocess
import time
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright
from playwright._impl._errors import Error as PlaywrightError

from http.server import HTTPServer, BaseHTTPRequestHandler


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/v1/internal/model/info":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"model_name":"stub-model"}')
        elif self.path == "/v1/internal/model/list":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"stub-model"}]}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path in ("/v1/chat/completions", "/v1/completions", "/api/v1/generate"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"content":"stub"}}]}')
        else:
            self.send_response(404)
            self.end_headers()


def _run_mock_llm_server(port: int):
    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.fixture(scope="session")
def llm_stub_base():
    # Start a primary stub on a free port
    port = _find_free_port()
    server, thread = _run_mock_llm_server(port)
    base = f"http://127.0.0.1:{port}"

    # CI guardrail: also try to bind :5000 as a fallback target if it's free
    try:
        s5000, t5000 = _run_mock_llm_server(5000)
    except OSError:
        s5000 = t5000 = None  # some other process might be using it locally

    # Sanity-check stub is reachable
    for _ in range(20):
        try:
            r = requests.get(f"{base}/v1/internal/model/info", timeout=0.2)
            if r.ok:
                break
        except Exception:
            time.sleep(0.05)

    yield base

    server.shutdown()
    thread.join(timeout=3)
    if s5000:
        s5000.shutdown()
    if t5000:
        t5000.join(timeout=3)


@pytest.fixture(scope="session")
def streamlit_app(llm_stub_base, tmp_path_factory):
    # Pick a port for the web app itself
    app_port = _find_free_port()
    app_url = f"http://127.0.0.1:{app_port}"

    # Build the exact env we'll give to Streamlit (and all its child processes)
    base = llm_stub_base
    child_env = os.environ.copy()
    child_env.update(
        {
            "LLM_BASE": base,
            "LLM_GENERATE_ENDPOINT": f"{base}/api/v1/generate",
            "LLM_COMPLETION_ENDPOINT": f"{base}/v1/completions",
            "LLM_CHAT_ENDPOINT": f"{base}/v1/chat/completions",
            "LLM_MODEL_LIST_ENDPOINT": f"{base}/v1/internal/model/list",
            "LLM_MODEL_LOAD_ENDPOINT": f"{base}/v1/internal/model/load",
            "LLM_MODEL_INFO_ENDPOINT": f"{base}/v1/internal/model/info",
            # Streamlit niceties for CI:
            "PYTHONUNBUFFERED": "1",
            "STREAMLIT_BROWSER_GATHERUSAGESTATS": "false",
        }
    )

    # Start Streamlit as a subprocess with the env explicitly set
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "main.py",
            "--server.headless",
            "true",
            "--server.port",
            str(app_port),
            "--browser.gatherUsageStats",
            "false",
        ],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait until it's serving
    import urllib.request, time

    start = time.time()
    last_log = ""
    while time.time() - start < 30:
        try:
            with urllib.request.urlopen(app_url, timeout=0.5) as _:
                break
        except Exception:
            # Read a bit of output to aid debugging on CI
            if proc.stdout:
                try:
                    last_log = proc.stdout.readline().strip() or last_log
                except Exception:
                    pass
            time.sleep(0.2)
    else:
        # Dump logs if it failed to boot
        if proc.stdout:
            print(proc.stdout.read())
        raise RuntimeError("Streamlit app failed to start in time.")

    yield app_url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def browser():
    """Provide a headless Chromium browser for tests.

    If the required browser binaries are missing (e.g. when running locally
    without having executed ``playwright install``), they will be installed on
    the fly.
    """
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError:
            # Install required browser binaries and system dependencies if missing.
            subprocess.run(
                ["playwright", "install", "--with-deps", "chromium"],
                check=True,
            )
            browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context()
    page = context.new_page()
    page.console_logs = []
    page.on("console", lambda msg: page.console_logs.append(f"{msg.type}: {msg.text}"))
    yield page
    context.close()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(autouse=True)
def capture_artifacts(request, page):
    yield
    if request.node.rep_call.failed:
        artifacts = Path("artifacts")
        artifacts.mkdir(exist_ok=True)
        page.screenshot(path=str(artifacts / f"{request.node.name}.png"))
        log_file = artifacts / f"{request.node.name}.log"
        log_file.write_text("\n".join(page.console_logs))
