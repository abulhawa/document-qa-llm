import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Generator

import pytest
import requests
pytest.importorskip("playwright")
from playwright.sync_api import sync_playwright
from playwright._impl._errors import Error as PlaywrightError


def _get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def streamlit_app() -> Generator[str, None, None]:
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
def browser() -> Generator[Any, None, None]:
    """Provide a headless Chromium browser for tests.

    If the required browser binaries are missing (e.g. when running locally
    without having executed ``playwright install``), they will be installed on
    the fly.
    """
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except PlaywrightError:
            # Install missing browser binaries without attempting system package installs
            subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
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
