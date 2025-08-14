import os, socket, threading, pytest
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


@pytest.fixture(scope="session", autouse=True)
def _mock_llm_server():
    # start stub on a free port
    try:
        port = int(os.getenv("TEST_LLM_PORT", "0")) or _find_free_port()
        server, thread = _run_mock_llm_server(port)
    except OSError:
        port = _find_free_port()
        server, thread = _run_mock_llm_server(port)

    base = f"http://127.0.0.1:{port}"

    # session-scoped monkeypatch via the class, not the fixture
    mp = pytest.MonkeyPatch()
    mp.setenv("LLM_BASE", base)
    mp.setenv("LLM_GENERATE_ENDPOINT", f"{base}/api/v1/generate")
    mp.setenv("LLM_COMPLETION_ENDPOINT", f"{base}/v1/completions")
    mp.setenv("LLM_CHAT_ENDPOINT", f"{base}/v1/chat/completions")
    mp.setenv("LLM_MODEL_LIST_ENDPOINT", f"{base}/v1/internal/model/list")
    mp.setenv("LLM_MODEL_LOAD_ENDPOINT", f"{base}/v1/internal/model/load")
    mp.setenv("LLM_MODEL_INFO_ENDPOINT", f"{base}/v1/internal/model/info")

    yield

    server.shutdown()
    thread.join(timeout=3)
    mp.undo()


def _get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def streamlit_app() -> str:
    """Start the Streamlit app on a free port and yield its URL.

    This allows running the E2E tests locally without having to manually
    launch the application first. The server is terminated once the test
    session finishes.
    """

    port = _get_free_port()
    url = f"http://localhost:{port}"

    env = os.environ.copy()
    # Prevent Streamlit from opening a real browser window.
    env.setdefault("BROWSER", "none")

    proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            "main.py",
            "--server.headless",
            "true",
            "--server.port",
            str(port),
        ],
        env=env,
    )

    # Wait for the server to become responsive.
    for _ in range(90):
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                break
        except Exception:
            time.sleep(1)
    else:
        proc.kill()
        raise RuntimeError("Streamlit did not start in time")

    yield url

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
