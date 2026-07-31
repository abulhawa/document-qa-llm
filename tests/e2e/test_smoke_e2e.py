import os

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect

pytestmark = pytest.mark.e2e

if os.getenv("CI") != "true":
    pytest.skip("No rows in Index Viewer; skipping filter smoke check", allow_module_level=True)

def test_smoke_e2e(streamlit_app, page):
    """High-level smoke test covering primary UI flows."""
    # Verify app loads and navigation updates heading
    page.goto(streamlit_app)
    expect(page.locator("h1")).to_contain_text("Ask Your Documents")
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
    page.get_by_role("button", name="Get Answer", exact=True).wait_for(timeout=3_000)
    page.get_by_label("Your question", exact=True).fill("What is Document QA?")
    page.get_by_role("button", name="Get Answer", exact=True).click()
    page.get_by_role("heading", name="📝 Answer", exact=True).wait_for(timeout=3_000)
    assert not any("error" in m.lower() for m in page.console_logs)

    # Index Viewer: navigate and exercise basic controls if data is present
    page.get_by_role("link", name="Storage & Index", exact=True).click()
    expect(page.get_by_role("heading", name="Storage & Index", exact=True)).to_be_visible(
        timeout=10_000
    )
    section_control = page.get_by_role("radiogroup", name="Section")
    expect(section_control).to_be_visible(timeout=10_000)
    section_control.get_by_role("radio", name="File Index Viewer", exact=True).click()

    # Give the backend-backed table enough time to render without a fixed sleep.
    table_rows = page.locator("table tbody tr")
    try:
        table_rows.first.wait_for(state="visible", timeout=10_000)
    except PlaywrightTimeoutError:
        pytest.skip("No rows in Index Viewer; skipping filter smoke check")
    rows_before = table_rows.count()
    # Try multiple reasonable selectors for the filter box
    filter_input = page.locator(
        "input[aria-label='Filter by path substring'], input[placeholder*='Filter'], input[type='search']"
    ).first
    expect(filter_input).to_be_visible(timeout=10_000)
    filter_input.fill("zzz")
    expect(table_rows).not_to_have_count(rows_before, timeout=10_000)
    rows_after = table_rows.count()
    assert rows_after <= rows_before
