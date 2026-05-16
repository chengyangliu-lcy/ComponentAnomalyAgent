from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Global lock: MCP server runs in a single process but tools may be called
# concurrently by the LLM. This lock serializes all HTTP requests to avoid
# triggering the API's rate limiter.
_REQUEST_LOCK = threading.Lock()


class RateLimitError(Exception):
    """Raised when the API returns 403 due to rate limiting."""

    def __init__(self, retry_after: float = 30.0):
        self.retry_after = retry_after
        super().__init__(f"Rate limited, retry after {retry_after}s")


class LCSCClient:
    """Client for querying LCSC and JLCPCB component APIs.

    Uses:
    - JLCPCB API for component search (returns C-numbers, stock, prices)
    - LCSC API for detailed product info and datasheets

    Rate limiting strategy:
    - Global request lock to serialize all HTTP calls
    - Minimum 2s gap between any two requests
    - Exponential backoff on 403: 5s, 10s, 20s (max 3 retries)
    - Cache results to avoid repeated calls for the same query
    """

    LCSC_DETAIL_URL = "https://wmsc.lcsc.com/ftps/wm/product/detail"
    JLCPCB_SEARCH_URL = (
        "https://jlcpcb.com/api/overseas-pcb-order/v1"
        "/shoppingCart/smtGood/selectSmtComponentList"
    )

    MIN_INTERVAL = 2.0       # Minimum seconds between any two requests
    MAX_RETRIES = 3           # Max retries on 403
    BACKOFF_BASE = 5.0        # Initial backoff seconds
    COOLDOWN_AFTER_403 = 15.0 # Extra cooldown after a 403 response

    def __init__(self, cache_ttl: int = 3600):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Accept": "application/json",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
        })
        self._last_request_time = 0.0
        self._cache: dict[str, tuple[float, Any]] = {}
        self._cache_ttl = cache_ttl

    def _wait_rate_limit(self):
        """Wait until minimum interval has elapsed since last request."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.MIN_INTERVAL:
            time.sleep(self.MIN_INTERVAL - elapsed)
        self._last_request_time = time.time()

    def _get_cached(self, key: str) -> Any | None:
        if key in self._cache:
            ts, data = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Any):
        self._cache[key] = (time.time(), data)

    def _request_with_retry(
        self, method: str, url: str, headers: dict | None = None,
        params: dict | None = None, json_body: dict | None = None,
    ) -> requests.Response:
        """Make an HTTP request with rate limiting and 403 retry logic.

        All requests are serialized through _REQUEST_LOCK to prevent
        concurrent calls from triggering rate limits.
        """
        with _REQUEST_LOCK:
            for attempt in range(self.MAX_RETRIES + 1):
                self._wait_rate_limit()
                try:
                    if method == "GET":
                        resp = self.session.get(
                            url, params=params, headers=headers, timeout=15
                        )
                    else:
                        resp = self.session.post(
                            url, json=json_body, headers=headers, timeout=15
                        )
                except requests.RequestException as e:
                    raise RuntimeError(f"网络请求失败: {e}") from e

                self._last_request_time = time.time()

                if resp.status_code == 403:
                    if attempt < self.MAX_RETRIES:
                        backoff = self.BACKOFF_BASE * (2 ** attempt)
                        logger.warning(
                            "403 Forbidden (attempt %d/%d), backing off %.1fs",
                            attempt + 1, self.MAX_RETRIES + 1, backoff,
                        )
                        time.sleep(backoff)
                        continue
                    raise RateLimitError(retry_after=self.COOLDOWN_AFTER_403)

                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 10))
                    raise RateLimitError(retry_after=retry_after)

                resp.raise_for_status()
                return resp

        raise RateLimitError(retry_after=self.COOLDOWN_AFTER_403)

    def search(self, keyword: str, page: int = 1, page_size: int = 20) -> dict:
        """Search components via JLCPCB API. Returns normalized result dict."""
        cache_key = f"search:{keyword}:{page}:{page_size}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        headers = {
            "Origin": "https://jlcpcb.com",
            "Referer": "https://jlcpcb.com/parts",
        }
        payload = {
            "keyword": keyword,
            "pageSize": page_size,
            "currentPage": page,
        }
        resp = self._request_with_retry(
            "POST", self.JLCPCB_SEARCH_URL, headers=headers, json_body=payload
        )
        raw = resp.json()

        data_raw = raw.get("data", {})
        page_info = data_raw.get("componentPageInfo", {})
        products = page_info.get("list", [])
        total = page_info.get("total", 0)

        normalized = {
            "total": total,
            "products": [_normalize_jlcpcb_product(p) for p in products],
        }
        self._set_cache(cache_key, normalized)
        return normalized

    def get_detail(self, product_code: str) -> dict:
        """Get product detail from LCSC by C-number."""
        cache_key = f"detail:{product_code}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        resp = self._request_with_retry(
            "GET", self.LCSC_DETAIL_URL, params={"productCode": product_code}
        )
        raw = resp.json()
        result = raw.get("result", {})
        if not result:
            return {}

        normalized = _normalize_lcsc_detail(result)
        self._set_cache(cache_key, normalized)
        return normalized

    def get_datasheet_url(self, product_code: str) -> str | None:
        """Get datasheet PDF URL for a product."""
        detail = self.get_detail(product_code)
        return detail.get("datasheet_url")


def _extract_c_number(url: str) -> str:
    m = re.search(r"C\d+", url)
    return m.group() if m else ""


def _normalize_jlcpcb_product(p: dict) -> dict:
    c_number = p.get("componentCode", "")
    if not c_number:
        c_number = _extract_c_number(p.get("lcscGoodsUrl", ""))

    prices = p.get("componentPrices", [])
    price_str = _format_jlcpcb_prices(prices)

    attrs = p.get("attributes", [])
    attr_lines = []
    for a in attrs:
        name = a.get("attributeNameEn", "")
        value = a.get("attributeValueEn", "")
        if name and value:
            attr_lines.append(f"  - {name}: {value}")

    return {
        "c_number": c_number,
        "name": p.get("componentNameEn") or p.get("componentName", ""),
        "model": p.get("componentModelEn", ""),
        "brand": p.get("componentBrandEn", ""),
        "category": p.get("componentTypeEn", ""),
        "stock": p.get("stockCount", 0),
        "library_type": p.get("componentLibraryType", ""),
        "is_basic": p.get("componentLibraryType") == "base",
        "price_str": price_str,
        "datasheet_url": p.get("dataManualUrl") or p.get("dataManualOfficialLink", ""),
        "lcsc_url": p.get("lcscGoodsUrl", ""),
        "image_url": p.get("componentImageUrl", ""),
        "description": p.get("describe", ""),
        "specification": p.get("componentSpecificationEn", ""),
        "attributes": attr_lines,
        "min_purchase": p.get("minPurchaseNum", 0),
        "rohs": p.get("rohsFlag", False),
    }


def _normalize_lcsc_detail(r: dict) -> dict:
    prices = r.get("productPriceList", [])
    price_str = _format_lcsc_prices(prices)

    params = r.get("paramVOList", [])
    param_lines = []
    for p in params:
        name = p.get("paramNameEn", "")
        value = p.get("paramValue", "")
        if name and value:
            param_lines.append(f"  - {name}: {value}")

    return {
        "c_number": r.get("productCode", ""),
        "name": r.get("productNameEn", ""),
        "model": r.get("productModel", ""),
        "brand": r.get("brandNameEn", ""),
        "category": r.get("catalogName", ""),
        "stock": r.get("stockNumber", 0),
        "min_order": r.get("minPacketNumber", 0),
        "price_str": price_str,
        "datasheet_url": r.get("pdfUrl", ""),
        "description": r.get("productDescEn") or r.get("productIntroEn", ""),
        "rohs": r.get("isEnvironment", False),
        "weight": r.get("productWeight", ""),
        "params": param_lines,
        "raw": r,
    }


def _format_jlcpcb_prices(prices: list) -> str:
    if not prices:
        return "暂无报价"
    parts = []
    for tier in prices[:4]:
        start = tier.get("startNumber", "?")
        price = tier.get("productPrice", "?")
        parts.append(f"{start}+: ${price}")
    return " | ".join(parts)


def _format_lcsc_prices(prices: list) -> str:
    if not prices:
        return "暂无报价"
    parts = []
    for tier in prices[:4]:
        qty = tier.get("ladder", "?")
        price = tier.get("productPrice", "?")
        currency = tier.get("currencySymbol", "$")
        parts.append(f"{qty}+: {currency}{price}")
    return " | ".join(parts)


def format_search_results(data: dict) -> str:
    products = data.get("products", [])
    total = data.get("total", 0)
    if not products:
        return "未找到匹配的元器件。"

    lines = [f"共找到 {total} 个结果，当前显示 {len(products)} 个：\n"]
    for i, p in enumerate(products, 1):
        code = p["c_number"]
        lib_tag = " [Basic]" if p["is_basic"] else " [Extended]"
        lines.append(
            f"{i}. [{code}] {p['name']}{lib_tag}\n"
            f"   型号: {p['model']} | 厂商: {p['brand']} | 分类: {p['category']}\n"
            f"   库存: {p['stock']} | 价格: {p['price_str']}"
        )
        if p["datasheet_url"]:
            lines.append(f"   数据手册: {p['datasheet_url']}")
    return "\n".join(lines)


def format_detail(data: dict) -> str:
    if not data:
        return "未找到该元器件的详细信息。"

    lines = [
        f"=== {data['c_number']} ===",
        f"名称: {data['name']}",
        f"型号: {data['model']}",
        f"厂商: {data['brand']}",
        f"分类: {data['category']}",
        f"库存: {data['stock']}",
        f"最小起订: {data.get('min_order', 'N/A')}",
        f"价格: {data['price_str']}",
        f"数据手册: {data['datasheet_url'] or '无'}",
        f"描述: {data['description']}",
    ]
    params = data.get("params", [])
    if params:
        lines.append("\n技术参数:")
        lines.extend(params)

    return "\n".join(lines)


def format_jlcpcb_results(data: dict) -> str:
    return format_search_results(data)
