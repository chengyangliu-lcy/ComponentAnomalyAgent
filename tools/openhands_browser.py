from __future__ import annotations

from dataclasses import dataclass
import os
from urllib.parse import urlparse

from schemas import Evidence
from tools.browser import BrowserFetchResult
from tools.utils import compact_text


@dataclass
class OpenHandsBrowserConfig:
    timeout_seconds: float = 15.0
    max_chars: int = 6000
    require_installed: bool = True


class OpenHandsBrowserFetcher:
    """Thin adapter around OpenHands browser-use tools.

    Imports and initializes OpenHands lazily so normal HTTP reading remains
    cheap and environments without Chromium can fall back cleanly.
    """

    def __init__(self, config: OpenHandsBrowserConfig | None = None) -> None:
        self.config = config or OpenHandsBrowserConfig()
        self._executor = None
        self._init_error: str | None = None

    def fetch(self, url: str, max_chars: int | None = None) -> BrowserFetchResult:
        if not self._is_public_http_url(url):
            return BrowserFetchResult(None, f"blocked invalid or non-public URL: {url}")
        executor, error = self._executor_or_error()
        if error:
            return BrowserFetchResult(None, error)
        assert executor is not None
        try:
            os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
            from openhands.tools.browser_use import BrowserGetContentAction, BrowserNavigateAction

            navigation = executor(BrowserNavigateAction(url=url, new_tab=False))
            if getattr(navigation, "is_error", False):
                return BrowserFetchResult(None, _observation_text(navigation) or "OpenHands browser navigation failed")

            content = executor(BrowserGetContentAction(extract_links=False, start_from_char=0))
            if getattr(content, "is_error", False):
                return BrowserFetchResult(None, _observation_text(content) or "OpenHands browser content extraction failed")

            text = compact_text(_observation_text(content), max_chars=max_chars or self.config.max_chars)
            if not text:
                return BrowserFetchResult(None, "OpenHands browser returned empty page content")
            return BrowserFetchResult(
                Evidence(
                    source=url,
                    title=url,
                    content=text,
                    metadata={
                        "kind": "openhands_browser_page",
                        "fallback_provider": "openhands_browser",
                        "read_backend": "openhands_browser",
                        "max_chars": max_chars or self.config.max_chars,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            return BrowserFetchResult(None, f"OpenHands browser fetch failed: {exc}")

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None and hasattr(executor, "close"):
            try:
                executor.close()
            except Exception:
                pass

    def __del__(self) -> None:
        self.close()

    def _executor_or_error(self):
        if self._init_error:
            return None, self._init_error
        if self._executor is not None:
            return self._executor, None
        try:
            os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
            from openhands.tools.browser_use.impl import BrowserToolExecutor

            self._executor = BrowserToolExecutor(
                headless=True,
                init_timeout_seconds=int(self.config.timeout_seconds),
                action_timeout_seconds=float(self.config.timeout_seconds),
            )
            return self._executor, None
        except Exception as exc:  # noqa: BLE001
            self._init_error = f"OpenHands browser unavailable: {exc}"
            return None, self._init_error

    def _is_public_http_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        host = parsed.hostname or ""
        blocked = ("localhost", "127.", "10.", "192.168.", "172.16.", "0.0.0.0")
        return not any(host.startswith(prefix) for prefix in blocked)


def _observation_text(observation) -> str:
    text = getattr(observation, "text", None)
    if text is None:
        text = str(observation or "")
    return str(text).strip()
