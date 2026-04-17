from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from schemas import Evidence


@dataclass
class BrowserFetchResult:
    evidence: Optional[Evidence]
    error: Optional[str] = None


class BrowserFallback:
    """Optional Playwright-based fallback kept isolated from normal HTTP tools."""

    def fetch(self, url: str, max_chars: int = 5000) -> BrowserFetchResult:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # noqa: BLE001
            return BrowserFetchResult(None, f"playwright is not installed: {exc}")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="networkidle", timeout=20000)
                title = page.title()
                text = page.locator("body").inner_text(timeout=5000)
                browser.close()
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            return BrowserFetchResult(Evidence(source=url, title=title or url, content=text, metadata={"kind": "browser_page"}))
        except Exception as exc:  # noqa: BLE001
            return BrowserFetchResult(None, str(exc))

