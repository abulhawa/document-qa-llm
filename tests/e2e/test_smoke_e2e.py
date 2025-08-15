import pytest, os
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e


def test_smoke_e2e(streamlit_app, page):
    """High-level smoke test covering primary UI flows."""
    # Verify app loads on the chat page and navigation updates heading
    page.goto(f"{streamlit_app}/?page=0_chat")
    expect(page.locator("h1")).to_contain_text("Talk to Your Documents", timeout=10_000)
    page.locator("a[href$='/ingest']").click()
    expect(page.locator("h1")).to_contain_text("Ingest")

    # Ingest negative path: submit without file selection
    if os.getenv("CI") == "true":
        page.get_by_role("button", name="Select File(s)").click()
        alert_text = page.locator("div[role='alert']").inner_text()
        assert "select" in alert_text.lower() or "picker failed" in alert_text.lower()
    else:
        print("Skipping file picker test locally due to native dialog issues")

    # Chat page: navigate and optionally submit a query if chat is available
    page.locator(f"a[href='{streamlit_app}/']").click()
    page.set_default_timeout(1_000)
    page.wait_for_timeout(500)
    page.get_by_role("button", name="Get Answer", exact=True).wait_for(timeout=3_000)
    question_input = page.get_by_label("Your question", exact=True)
    question_input.fill("What is Document QA?")
    page.get_by_role("button", name="Get Answer", exact=True).click()
    page.get_by_role("heading", name="📝 Answer", exact=True).wait_for(timeout=3_000)
    if any("error" in m.lower() for m in page.console_logs):
        print("Console errors:", page.console_logs)

    # Index Viewer: navigate and exercise basic controls if data is present
    page.locator("a[href$='/index_viewer']").click()
    expect(page.locator("h1")).to_contain_text("Index Viewer")
    page.wait_for_timeout(500)
    filter_box = page.locator("input[aria-label='Filter by path substring']")
    if filter_box.count() > 0:
        rows_before = page.locator("table tbody tr").count()
        filter_box.fill("zzz")
        page.wait_for_timeout(500)
        rows_after = page.locator("table tbody tr").count()
        assert rows_after <= rows_before
        assert page.locator("button:has-text('Download')").is_visible()
    else:
        print("Skipping index viewer interactions; index unavailable")
