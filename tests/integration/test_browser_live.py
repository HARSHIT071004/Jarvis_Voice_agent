"""Live browser integration test (Phase 6 §37/§41).

Skipped unless RUN_LIVE_TESTS=1, so the main suite never depends on the
network. Uses the real Chromium against https://example.com.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS") != "1",
    reason="live network test; set RUN_LIVE_TESTS=1 to enable",
)


def test_live_open_read_click():
    from app.browser import BrowserController, BrowserPolicy

    async def main():
        controller = BrowserController(
            BrowserPolicy(), headless=True, navigation_timeout_ms=20000
        )  # default policy: http/https only, no private hosts
        try:
            result = await controller.open_url("https://example.com")
            assert result.status == "verified"
            snapshot, data = await controller.read_page()
            assert "Example Domain" in data["title"] or "example" in data["text"].lower()
            assert data["links"], "expected at least one link on example.com"
            # real click that navigates; verification compares url/title/text
            link = data["links"][0]
            click = await controller.click(link["id"])
            assert click.status == "verified"
            after, _ = await controller.read_page()
            assert after.url != snapshot.url  # navigation observed
        finally:
            await controller.close()

    asyncio.run(main())


def test_live_blocked_navigation_stays_local():
    """Metadata/private hosts stay blocked even on a live browser."""
    from app.browser import BrowserController, BrowserError, BrowserPolicy

    async def main():
        controller = BrowserController(BrowserPolicy(), headless=True)
        try:
            with pytest.raises(BrowserError) as exc:
                await controller.open_url("http://169.254.169.254/latest/meta-data/")
            assert exc.value.code == "BLOCKED_DOMAIN"
        finally:
            await controller.close()

    asyncio.run(main())
