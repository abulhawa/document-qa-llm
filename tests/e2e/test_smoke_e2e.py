import pytest, os
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
    if os.getenv("CI") == "true":
        page.get_by_role("button", name="Select File(s)").click()
        alert_text = page.locator("div[role='alert']").inner_text()
        assert "select" in alert_text.lower() or "picker failed" in alert_text.lower()
    else:
        print("Skipping file picker test locally due to native dialog issues")

    # Chat page: navigate and optionally submit a query if chat is available
    page.get_by_role("link", name="Ask Your Documents").click()
    page.set_default_timeout(1_000)
    page.wait_for_timeout(500)
    page.get_by_role("button", name="Get Answer", exact=True).wait_for(timeout=3_000)
    page.get_by_label("Your question", exact=True).fill("What is Document QA?")
    page.get_by_role("button", name="Get Answer", exact=True).click()
    page.get_by_role("heading", name="📝 Answer", exact=True).wait_for(timeout=3_000)
    assert not any("error" in m.lower() for m in page.console_logs)

    # Index Viewer: navigate and exercise basic controls if data is present
    page.get_by_role("link", name="File Index Viewer").click()
    page.wait_for_timeout(500)
    rows_before = page.locator("table tbody tr").count()
    page.fill("input[aria-label='Filter by path substring']", "zzz")
    page.wait_for_timeout(500)
    rows_after = page.locator("table tbody tr").count()
    assert rows_after <= rows_before
    assert page.locator("button:has-text('Download')").is_visible()
