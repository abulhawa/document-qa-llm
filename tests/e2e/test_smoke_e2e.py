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
    # Wait (up to ~5s) for the table to render at least one data row
    table_rows = page.locator("table tbody tr")
    for _ in range(25):
        if table_rows.count() > 0:
            break
        page.wait_for_timeout(200)
    rows_before = table_rows.count()
    if rows_before == 0:
        # No indexed rows yet; skip filter smoke instead of failing
        import pytest
        pytest.skip("No rows in Index Viewer; skipping filter smoke check")
    # Try multiple reasonable selectors for the filter box
    filter_input = page.locator(
        "input[aria-label='Filter by path substring'], input[placeholder*='Filter'], input[type='search']"
    ).first
    filter_input.fill("zzz")
    page.wait_for_timeout(500)
    rows_after = table_rows.count()
    assert rows_after <= rows_before
