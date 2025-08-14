import pytest
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_smoke_e2e(streamlit_app, page):
    """High-level smoke test covering primary UI flows."""
    # Verify app loads and navigation updates heading
    page.goto(streamlit_app)
    main = page.locator("#page_0")
    expect(main.locator("h1")).to_contain_text("Talk to Your Documents")
    page.get_by_role("link", name="Ingest Documents").click()
    expect(main.locator("h1")).to_contain_text("Ingest Documents")

    # Ingest negative path: submit without file selection
    page.get_by_role("button", name="Select File(s)").click()
    alert_text = main.locator("div[role='alert']").inner_text()
    assert "select" in alert_text.lower() or "picker failed" in alert_text.lower()

    # Chat page: navigate and optionally submit a query if chat is available
    page.get_by_role("link", name="Ask Your Documents").click()
    page.wait_for_timeout(500)
    chat_box = main.locator("textarea[placeholder='Ask a question...']")
    if chat_box.count() > 0:
        chat_box.fill("What is Document QA?")
        chat_box.press("Enter")
        page.wait_for_selector("#page_0 div[data-testid='stChatMessage']")
        assert not any("error" in m.lower() for m in page.console_logs)

    # Index Viewer: navigate and exercise basic controls if data is present
    page.get_by_role("link", name="File Index Viewer").click()
    page.wait_for_timeout(500)
    if main.locator("table").count() > 0:
        rows_before = main.locator("table tbody tr").count()
        main.locator("input[aria-label='Filter by path substring']").fill("zzz")
        page.wait_for_timeout(500)
        rows_after = main.locator("table tbody tr").count()
        assert rows_after <= rows_before
        assert main.locator("button:has-text('Download')").is_visible()
