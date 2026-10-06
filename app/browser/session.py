"""BrowserSession — Playwright lifecycle (§5/§29).

One session owns playwright -> browser -> context -> page. A task reuses
the same session for every tool call (no relaunch per action); close() is
idempotent and always cleans up so browser processes never leak.
"""

from __future__ import annotations

import logging

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from app.browser.errors import BrowserError

logger = logging.getLogger("jarvis.browser")


class BrowserSession:
    """Lazy-start Playwright session with a single active page."""

    def __init__(self, headless: bool = True, navigation_timeout_ms: int = 15000) -> None:
        self.headless = headless
        self.navigation_timeout_ms = navigation_timeout_ms
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    @property
    def page(self) -> Page | None:
        return self._page

    @property
    def started(self) -> bool:
        return self._page is not None

    async def start(self) -> Page:
        if self._page is not None and not self._page.is_closed():
            return self._page
        try:
            # Recover from externally closed pages / crashed browsers (§15):
            # recreate only what is actually gone.
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            if self._browser is None or not self._browser.is_connected():
                self._browser = await self._playwright.chromium.launch(headless=self.headless)
            if self._context is None or self._context.is_closed():
                self._context = await self._browser.new_context()
                self._context.set_default_navigation_timeout(self.navigation_timeout_ms)
                self._context.set_default_timeout(self.navigation_timeout_ms)
            if self._page is None or self._page.is_closed():
                self._page = await self._context.new_page()
        except Exception as exc:
            await self.close()
            raise BrowserError(
                "BROWSER_START_FAILED",
                "The browser could not be started. Try again in a moment.",
            ) from exc
        logger.info("[BROWSER] session_started headless=%s", self.headless)
        return self._page

    async def new_page(self) -> Page:
        """Isolated page inside the same browser (used between tasks)."""
        page = await self.start()
        assert self._context is not None
        self._page = await self._context.new_page()
        await page.close()
        assert self._page is not None
        self._page.set_default_navigation_timeout(self.navigation_timeout_ms)
        return self._page

    async def close(self) -> None:
        """Idempotent cleanup: page -> context -> browser -> playwright."""
        for closer, label, close in (
            (self._page, "page", lambda obj: obj.close()),
            (self._context, "context", lambda obj: obj.close()),
            (self._browser, "browser", lambda obj: obj.close()),
            (self._playwright, "playwright", lambda obj: obj.stop()),  # Playwright has stop(), not close()
        ):
            if closer is None:
                continue
            try:
                await close(closer)
            except Exception:
                logger.warning("[BROWSER] cleanup failed for %s (ignored)", label)
        self._page = None
        self._context = None
        self._browser = None
        self._playwright = None
        logger.info("[BROWSER] session_closed")
