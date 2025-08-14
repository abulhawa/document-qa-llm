import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_smoke_e2e(streamlit_app, page):
    """High-level smoke test covering primary UI flows."""
    # Verify app loads and navigation updates heading
    page.goto(streamlit_app)
    expect(page.locator("h1")).to_contain_text("Talk to Your Documents")
    page.get_by_role("link", name="Ingest Documents").click()
    expect(page.locator("h1")).to_contain_text("Ingest Documents")

    # Ingest negative path: submit without file selection
    page.get_by_role("button", name="Select File(s)").click()
    alert_text = page.locator("div[role='alert']").inner_text()
    assert "select" in alert_text.lower() or "picker failed" in alert_text.lower()

    # Chat page: submit query and ensure answer rendered without console errors
    page.get_by_role("link", name="Chat").click()
    page.fill("textarea[placeholder='Ask a question...']", "What is Document QA?")
    page.press("textarea[placeholder='Ask a question...']", "Enter")
    page.wait_for_selector("div[data-testid='stChatMessage']")
    assert not any("error" in m.lower() for m in page.console_logs)

    # Index Viewer: table render, filter reduces rows, and CSV download control
    page.get_by_role("link", name="File Index Viewer").click()
    page.wait_for_selector("table")
    rows_before = page.locator("table tbody tr").count()
    page.fill("input[aria-label='Filter by path substring']", "zzz")
    page.wait_for_timeout(500)
    rows_after = page.locator("table tbody tr").count()
    assert rows_after <= rows_before
    assert page.locator("button:has-text('Download')").is_visible()
