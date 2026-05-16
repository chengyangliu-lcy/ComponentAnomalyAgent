"""Demo: SearXNG vs HTML(DuckDuckGo) 搜索对比 + Jina Reader 完整 Markdown 输出"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tools.evidence_tools import APIWebSearchExecutor, RobustWebReadExecutor

QUERY = "LM358 datasheet voltage comparator circuit"
FETCH_TOP_N = 3
JINA_MAX_CHARS = 8000
SEARXNG_URL = "http://127.0.0.1:9980"


def search(provider: str, query: str, limit: int = 8):
    kwargs = dict(provider_order=[provider], api_key_envs={}, api_keys={}, timeout=20)
    if provider == "searxng":
        kwargs["searxng_url"] = SEARXNG_URL
    return APIWebSearchExecutor(**kwargs).run(query, limit=limit)


def is_pdf_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(".pdf") or "/pdf/" in lowered


def fetch_with_jina(url: str, title: str, snippet: str):
    reader = RobustWebReadExecutor(
        enable_openhands_browser_primary=False,
        enable_jina_reader=True,
        jina_max_chars=JINA_MAX_CHARS,
    )
    return reader.run(url, title, snippet)


def print_search_results(label: str, result):
    print(f"\n{'='*70}")
    print(f"  [{label}]  success={result.success}  results={len(result.evidence)}")
    print(f"{'='*70}")
    for i, ev in enumerate(result.evidence):
        pdf_tag = " [PDF]" if is_pdf_url(ev.source) else ""
        print(f"  {i+1}. {ev.title[:80]}{pdf_tag}")
        print(f"     URL: {ev.source}")
        print(f"     Snippet: {ev.content[:200]}")
        print()


def fetch_and_print(label: str, ev, idx: int):
    pdf_tag = " [PDF]" if is_pdf_url(ev.source) else ""
    print(f"\n{'#'*70}")
    print(f"  {label} Result {idx}: {ev.title[:60]}{pdf_tag}")
    print(f"  URL: {ev.source}")
    print(f"{'#'*70}")

    if is_pdf_url(ev.source):
        print(f"\n  >> PDF 跳过抓取，保留 snippet:")
        print(f"  >> {ev.content}")
        return

    read = fetch_with_jina(ev.source, ev.title, ev.content)
    backend = read.metadata.get("read_backend")
    print(f"\n  >> Fetch backend: {backend}")

    if read.evidence:
        md = read.evidence[0].content
        print(f"  >> Content length: {len(md)} chars")
        print(f"\n{'─'*70}")
        print(f"  >>> BEGIN MARKDOWN ({backend})")
        print(f"{'─'*70}")
        print(md)
        print(f"{'─'*70}")
        print(f"  >>> END MARKDOWN")
        print(f"{'─'*70}")
    else:
        print(f"  >> Fetch FAILED: {read.errors}")


def main():
    print(f"Query: {QUERY}")
    print(f"Fetch top {FETCH_TOP_N} non-PDF results with Jina Reader (max {JINA_MAX_CHARS} chars)\n")

    # ── 1. SearXNG 搜索 ──
    sx = search("searxng", QUERY)
    print_search_results("SearXNG", sx)

    # ── 2. HTML (DuckDuckGo) 搜索 ──
    html = search("html", QUERY)
    print_search_results("HTML/DuckDuckGo", html)

    # ── 3. 对 SearXNG 结果用 Jina Reader 抓取（跳过 PDF）──
    print(f"\n\n{'#'*70}")
    print(f"  Jina Reader fetch on SearXNG results (top {FETCH_TOP_N} non-PDF)")
    print(f"{'#'*70}")
    count = 0
    for ev in sx.evidence:
        if count >= FETCH_TOP_N:
            break
        if is_pdf_url(ev.source):
            continue
        count += 1
        fetch_and_print("SearXNG", ev, count)

    # ── 4. 对 HTML 结果用 Jina Reader 抓取（跳过 PDF）──
    print(f"\n\n{'#'*70}")
    print(f"  Jina Reader fetch on HTML/DuckDuckGo results (top {FETCH_TOP_N} non-PDF)")
    print(f"{'#'*70}")
    count = 0
    for ev in html.evidence:
        if count >= FETCH_TOP_N:
            break
        if is_pdf_url(ev.source):
            continue
        count += 1
        fetch_and_print("HTML", ev, count)

    # ── 5. 总结 ──
    print(f"\n\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  SearXNG:        {len(sx.evidence)} results")
    print(f"  HTML/DDG:       {len(html.evidence)} results")
    print(f"  Jina Reader:    enabled, max {JINA_MAX_CHARS} chars")
    print(f"  OpenHands:      disabled for this test")


if __name__ == "__main__":
    main()
