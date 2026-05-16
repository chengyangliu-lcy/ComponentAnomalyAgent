from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from schemas import Evidence


@dataclass
class BrowserFetchResult:
    evidence: Optional[Evidence]
    error: Optional[str] = None


class BrowserFallback:
    """Optional Playwright-based fallback kept isolated from normal HTTP tools."""

    def __init__(
        self,
        timeout_ms: int = 20000,
        wait_after_load_ms: int = 2500,
        channels: tuple[str | None, ...] = (None, "chrome", "msedge"),
    ) -> None:
        self.timeout_ms = timeout_ms
        self.wait_after_load_ms = wait_after_load_ms
        self.channels = channels

    def fetch(self, url: str, max_chars: int = 5000) -> BrowserFetchResult:
        if not self._is_public_http_url(url):
            return BrowserFetchResult(None, f"blocked invalid or non-public URL: {url}")
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # noqa: BLE001
            return BrowserFetchResult(None, f"playwright is not installed: {exc}")

        try:
            with sync_playwright() as p:
                browser = None
                launch_errors: list[str] = []
                for channel in self.channels:
                    try:
                        kwargs = {"headless": True}
                        if channel:
                            kwargs["channel"] = channel
                        browser = p.chromium.launch(**kwargs)
                        break
                    except Exception as exc:  # noqa: BLE001
                        launch_errors.append(f"{channel or 'chromium'}: {exc}")
                if browser is None:
                    return BrowserFetchResult(None, "playwright browser unavailable: " + " | ".join(launch_errors))

                try:
                    page = browser.new_page(
                        locale="zh-CN",
                        user_agent=(
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                        ),
                    )
                    response = page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                    if self.wait_after_load_ms > 0:
                        page.wait_for_timeout(self.wait_after_load_ms)
                    title = page.title()
                    text = page.locator("body").inner_text(timeout=5000)
                    status_code = response.status if response else None
                    final_url = page.url
                finally:
                    browser.close()

            raw_limit = max(max_chars * 3, max_chars)
            if len(text) > raw_limit:
                text = text[:raw_limit] + "..."
            if not text:
                return BrowserFetchResult(None, "playwright browser returned empty page content")
            return BrowserFetchResult(
                Evidence(
                    source=final_url or url,
                    title=title or url,
                    content=text,
                    metadata={
                        "kind": "browser_page",
                        "read_backend": "playwright_browser",
                        "status_code": status_code,
                        "max_chars": max_chars,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            return BrowserFetchResult(None, str(exc))

    def _is_public_http_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        host = parsed.hostname or ""
        blocked = ("localhost", "127.", "10.", "192.168.", "172.16.", "0.0.0.0")
        return not any(host.startswith(prefix) for prefix in blocked)
