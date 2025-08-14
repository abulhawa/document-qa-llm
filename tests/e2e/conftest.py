import subprocess
import time
import socket
from pathlib import Path

import requests
import pytest
from playwright.sync_api import sync_playwright


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


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch()
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
