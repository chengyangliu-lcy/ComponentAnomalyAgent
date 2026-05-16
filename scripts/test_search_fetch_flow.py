from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.dataset_parser import DatasetParser
from tools.evidence_tools import APIWebSearchExecutor, RobustWebReadExecutor, ToolRun


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SearXNG search -> web fetch flow with verbose output.")
    parser.add_argument("--query", action="append", help="Search query. Can be passed multiple times.")
    parser.add_argument("--url", action="append", help="Fetch a URL directly. Can be passed multiple times.")
    parser.add_argument("--sample-id", action="append", help="Dataset sample_id/post_id to build a query from.")
    parser.add_argument("--dataset", default="2025_dataset.jsonl")
    parser.add_argument("--image-root", default="2025")
    parser.add_argument("--searxng-url", default="http://127.0.0.1:9980")
    parser.add_argument("--search-limit", type=int, default=5)
    parser.add_argument("--fetch-limit", type=int, default=3)
    parser.add_argument("--jina-max-chars", type=int, default=8000)
    parser.add_argument("--crawl4ai-max-chars", type=int, default=6000)
    parser.add_argument("--crawl4ai-primary", action="store_true", help="Use crawl4ai before Jina in non-compare mode.")
    parser.add_argument("--crawl4ai-content-filter-type", default="bm25", choices=["pruning", "bm25"])
    parser.add_argument("--crawl4ai-markdown-mode", default="best", choices=["best", "fit", "raw"])
    parser.add_argument("--crawl4ai-content-source", default="markdown", choices=["best", "markdown", "cleaned_html"])
    parser.add_argument("--crawl4ai-word-count-threshold", type=int, default=10)
    parser.add_argument("--crawl4ai-css-selector", default="", help="Optional strict crawl4ai CSS selector, e.g. article.")
    parser.add_argument(
        "--crawl4ai-target-element",
        action="append",
        default=None,
        help="crawl4ai target element selector. Can be passed multiple times.",
    )
    parser.add_argument("--crawl4ai-excluded-selector", default="", help="Override crawl4ai excluded selector list.")
    parser.add_argument("--content-chars", type=int, default=3000)
    parser.add_argument("--full-content", action="store_true", help="Print full fetched content.")
    parser.add_argument("--include-pdf", action="store_true", help="Fetch PDF-like results instead of skipping them.")
    parser.add_argument("--disable-browser", action="store_true", help="Disable Playwright browser fallback.")
    parser.add_argument("--enable-openhands", action="store_true", help="Enable OpenHands browser fallback.")
    parser.add_argument("--browser-wait-ms", type=int, default=2500)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--compare", action="store_true", help="Run comparison mode: crawl4ai vs jina on every URL.")
    parser.add_argument("--competitive", action="store_true", help="Use production competitive mode in non-compare fetches.")
    parser.add_argument("--crawl4ai-only", action="store_true", help="Use only crawl4ai (disable jina).")
    parser.add_argument("--jina-only", action="store_true", help="Use only jina (disable crawl4ai).")
    parser.add_argument("--scrapling-only", action="store_true", help="Use only Scrapling (disable crawl4ai and jina).")
    parser.add_argument("--scrapling-max-chars", type=int, default=8000)
    parser.add_argument("--scrapling-mode", default="dynamic", choices=["fetcher", "dynamic", "stealthy"])
    parser.add_argument("--scrapling-content-source", default="html", choices=["html", "text"])
    parser.add_argument("--scrapling-no-auto-match", action="store_true", help="Disable Scrapling adaptive selector matching.")
    parser.add_argument("--scrapling-no-network-idle", action="store_true", help="Disable network idle wait for dynamic/stealthy modes.")
    parser.add_argument("--scrapling-wait-ms", type=int, default=0)
    parser.add_argument("--scrapling-wait-selector", default="")
    parser.add_argument("--scrapling-disable-resources", action="store_true")
    parser.add_argument("--scrapling-load-images", action="store_true", help="Do not block images in stealthy mode.")
    parser.add_argument("--scrapling-no-google-search", action="store_true", help="Disable Google referer behavior.")
    parser.add_argument("--scrapling-real-chrome", action="store_true", help="Use real Chrome in dynamic mode if available.")
    parser.add_argument("--scrapling-css-selector", default="", help="Optional strict Scrapling CSS selector, e.g. article.")
    parser.add_argument(
        "--scrapling-target-element",
        action="append",
        default=None,
        help="Scrapling target element selector. Can be passed multiple times.",
    )
    parser.add_argument("--scrapling-excluded-selector", default="", help="Override Scrapling excluded selector list.")
    parser.add_argument("--comparison-output", default=None, help="Path to write comparison JSONL results.")
    return parser.parse_args()


def main() -> None:
    _configure_stdout()
    args = parse_args()
    queries = list(args.query or [])

    if args.sample_id:
        samples = list(DatasetParser(Path(args.dataset), Path(args.image_root)).iter_samples())
        by_id = {sample.sample_id: sample for sample in samples}
        by_id.update({sample.post_id: sample for sample in samples})
        for sample_id in args.sample_id:
            sample = by_id.get(sample_id)
            if not sample:
                print(f"[sample] not found: {sample_id}")
                continue
            query = build_query_from_sample(sample.question_text)
            queries.append(query)
            print("=" * 100)
            print(f"[sample] id={sample.sample_id}")
            print(f"[sample] question={one_line(sample.question_text, 500)}")
            print(f"[sample] query={query}")

    if not queries and not args.url:
        queries = ["LM358 datasheet voltage comparator circuit"]

    comparison_path = args.comparison_output
    if args.compare and not comparison_path:
        comparison_path = str(REPO_ROOT / "outputs" / "backend_comparison.jsonl")
        args.comparison_output = comparison_path

    searcher = APIWebSearchExecutor(
        provider_order=["searxng"],
        api_key_envs={},
        api_keys={},
        timeout=args.timeout,
        searxng_url=args.searxng_url,
    )
    reader = make_reader(
        args,
        enable_crawl4ai=not args.jina_only and not args.scrapling_only,
        enable_jina=not args.crawl4ai_only and not args.scrapling_only,
        enable_scrapling=args.scrapling_only or (args.competitive and not args.crawl4ai_only and not args.jina_only),
        compare_mode=False,
        comparison_path=comparison_path,
    )

    try:
        all_stats: list[dict] = []
        if args.url:
            for url in args.url:
                stats = run_direct_fetch(url, reader, args)
                if stats:
                    all_stats.extend(stats)
        for query in queries:
            stats = run_flow(query, searcher, reader, args)
            if stats:
                all_stats.extend(stats)
        if all_stats:
            print_summary(all_stats, args.compare)
    finally:
        reader.close()


def make_reader(
    args: argparse.Namespace,
    *,
    enable_crawl4ai: bool,
    enable_jina: bool,
    enable_scrapling: bool = False,
    compare_mode: bool = False,
    comparison_path: str | None = None,
) -> RobustWebReadExecutor:
    return RobustWebReadExecutor(
        timeout=args.timeout,
        enable_openhands_browser_primary=args.enable_openhands,
        enable_browser_fallback=not args.disable_browser,
        browser_fallback_wait_ms=args.browser_wait_ms,
        enable_crawl4ai=enable_crawl4ai,
        crawl4ai_max_chars=args.crawl4ai_max_chars,
        crawl4ai_primary=args.crawl4ai_primary,
        crawl4ai_content_filter_type=args.crawl4ai_content_filter_type,
        crawl4ai_markdown_mode=args.crawl4ai_markdown_mode,
        crawl4ai_content_source=args.crawl4ai_content_source,
        crawl4ai_word_count_threshold=args.crawl4ai_word_count_threshold,
        crawl4ai_css_selector=args.crawl4ai_css_selector,
        crawl4ai_target_elements=args.crawl4ai_target_element,
        crawl4ai_excluded_selector=args.crawl4ai_excluded_selector,
        enable_scrapling=enable_scrapling,
        scrapling_timeout_seconds=args.timeout,
        scrapling_max_chars=args.scrapling_max_chars,
        scrapling_mode=args.scrapling_mode,
        scrapling_content_source=args.scrapling_content_source,
        scrapling_auto_match=not args.scrapling_no_auto_match,
        scrapling_network_idle=not args.scrapling_no_network_idle,
        scrapling_wait_ms=args.scrapling_wait_ms,
        scrapling_wait_selector=args.scrapling_wait_selector,
        scrapling_disable_resources=args.scrapling_disable_resources,
        scrapling_block_images=not args.scrapling_load_images,
        scrapling_google_search=not args.scrapling_no_google_search,
        scrapling_real_chrome=args.scrapling_real_chrome,
        scrapling_css_selector=args.scrapling_css_selector,
        scrapling_target_elements=args.scrapling_target_element,
        scrapling_excluded_selector=args.scrapling_excluded_selector,
        enable_jina_reader=enable_jina,
        jina_max_chars=args.jina_max_chars,
        web_read_competitive_mode=args.competitive,
        web_read_competitive_timeout_seconds=args.timeout,
        web_read_min_quality_score=0.30,
        web_read_min_clean_chars=300,
        web_read_compare_mode=compare_mode,
        backend_comparison_path=comparison_path,
    )


def run_direct_fetch(url: str, reader: RobustWebReadExecutor, args: argparse.Namespace) -> list[dict]:
    print("\n" + "=" * 100)
    print(f"[direct_url] {url}")
    if args.compare and not args.crawl4ai_only and not args.jina_only and not args.scrapling_only:
        return run_compare_fetch(url, url, "", args, label="fetch 1")

    read = reader.run(url, url, "")
    read_backend = str(read.metadata.get("read_backend") or "")
    fetch_status = "FETCH_OK" if read_backend != "snippet_fallback" and read.success else "FETCH_FAILED_WITH_SNIPPET_FALLBACK"
    print(f"[fetch 1] success={read.success} summary={read.summary}")
    print(f"[fetch 1] status={fetch_status}")
    print(f"[fetch 1] read_backend={read_backend}")
    print(f"[fetch 1] errors={read.errors}")
    print(f"[fetch 1] metadata={read.metadata}")

    page_chars = 0
    quality_score = 0.0
    if read.evidence:
        page = read.evidence[0]
        page_chars = len(page.content)
        quality_score = float(page.metadata.get("quality_score") or 0.0)
        print(f"[fetch 1] evidence_source={page.source}")
        print(f"[fetch 1] evidence_title={page.title}")
        print(f"[fetch 1] evidence_metadata={page.metadata}")
        content = page.content if args.full_content else page.content[: args.content_chars]
        print(f"[fetch 1] content_chars_total={page_chars} printed={len(content)}")
        print(f"[fetch 1] quality_score={quality_score:.4f}")
        print(f"[fetch 1] BEGIN_CONTENT")
        print(content)
        print(f"[fetch 1] END_CONTENT")
    else:
        print("[fetch 1] no evidence")

    return [
        {
            "query": "",
            "url": url,
            "read_backend": read_backend,
            "fetch_status": fetch_status,
            "success": read.success,
            "page_chars": page_chars,
            "quality_score": quality_score,
            "errors": read.errors,
        }
    ]


def run_compare_fetch(url: str, title: str, snippet: str, args: argparse.Namespace, *, label: str) -> list[dict]:
    """Script-only comparison: print both backend outputs for human inspection."""

    crawl4ai_reader = make_reader(args, enable_crawl4ai=True, enable_jina=False, enable_scrapling=False)
    jina_reader = make_reader(args, enable_crawl4ai=False, enable_jina=True, enable_scrapling=False)
    scrapling_reader = make_reader(args, enable_crawl4ai=False, enable_jina=False, enable_scrapling=True)
    try:
        crawl4ai_read = run_crawl4ai_candidate(crawl4ai_reader, url, title, snippet, args)
        jina_read = run_jina_candidate(jina_reader, url, title, snippet, args)
        scrapling_read = run_scrapling_candidate(scrapling_reader, url, title, snippet, args)
    finally:
        crawl4ai_reader.close()
        jina_reader.close()
        scrapling_reader.close()

    crawl4ai_stat = print_backend_fetch(label, "crawl4ai", crawl4ai_read, args)
    jina_stat = print_backend_fetch(label, "jina_reader", jina_read, args)
    scrapling_stat = print_backend_fetch(label, "scrapling", scrapling_read, args)

    crawl4ai_score = crawl4ai_stat["quality_score"]
    jina_score = jina_stat["quality_score"]
    scrapling_score = scrapling_stat["quality_score"]
    scores = {"crawl4ai": crawl4ai_score, "jina": jina_score, "scrapling": scrapling_score}
    if all(score <= 0.0 for score in scores.values()):
        winner = "tie_failed"
    else:
        winner = max(scores.items(), key=lambda item: item[1])[0]

    comparison = {
        "url": url,
        "crawl4ai_score": round(crawl4ai_score, 4),
        "crawl4ai_chars": crawl4ai_stat["page_chars"],
        "crawl4ai_error": "; ".join(crawl4ai_stat["errors"]),
        "jina_score": round(jina_score, 4),
        "jina_chars": jina_stat["page_chars"],
        "jina_error": "; ".join(jina_stat["errors"]),
        "scrapling_score": round(scrapling_score, 4),
        "scrapling_chars": scrapling_stat["page_chars"],
        "scrapling_error": "; ".join(scrapling_stat["errors"]),
        "winner": winner,
    }
    print(f"[{label}] comparison={comparison}")
    if args.comparison_output:
        path = Path(args.comparison_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(comparison, ensure_ascii=False) + "\n")

    return [
        {
            "query": "",
            "url": url,
            "read_backend": "compare_detail",
            "fetch_status": (
                "FETCH_OK"
                if crawl4ai_stat["success"] or jina_stat["success"] or scrapling_stat["success"]
                else "FETCH_FAILED_WITH_SNIPPET_FALLBACK"
            ),
            "success": crawl4ai_stat["success"] or jina_stat["success"] or scrapling_stat["success"],
            "page_chars": max(crawl4ai_stat["page_chars"], jina_stat["page_chars"], scrapling_stat["page_chars"]),
            "quality_score": max(crawl4ai_score, jina_score, scrapling_score),
            "errors": crawl4ai_stat["errors"] + jina_stat["errors"] + scrapling_stat["errors"],
            "compare_mode": True,
            "crawl4ai_score": crawl4ai_score,
            "crawl4ai_chars": crawl4ai_stat["page_chars"],
            "jina_score": jina_score,
            "jina_chars": jina_stat["page_chars"],
            "scrapling_score": scrapling_score,
            "scrapling_chars": scrapling_stat["page_chars"],
            "winner": winner,
        }
    ]


def run_crawl4ai_candidate(reader: RobustWebReadExecutor, url: str, title: str, snippet: str, args: argparse.Namespace) -> ToolRun:
    result = reader.crawl4ai_fetcher.fetch(
        url,
        max_chars=args.crawl4ai_max_chars,
        title=title,
        snippet=snippet,
    )
    if not result.evidence:
        return ToolRun([], summary=f"crawl4ai failed {url}", success=False, errors=[result.error or "crawl4ai: empty"], metadata={"read_backend": "crawl4ai"})
    evidence = reader._clean_evidence(
        result.evidence,
        title=title or url,
        snippet=snippet,
        max_chars=args.crawl4ai_max_chars,
        source_format="text",
    )
    evidence.title = title or evidence.title or url
    evidence.metadata["read_backend"] = "crawl4ai"
    return ToolRun([evidence], summary=f"crawl4ai candidate {url}", metadata={"read_backend": "crawl4ai"})


def run_jina_candidate(reader: RobustWebReadExecutor, url: str, title: str, snippet: str, args: argparse.Namespace) -> ToolRun:
    result = reader._fetch_jina(url, title=title, snippet=snippet, max_chars=args.jina_max_chars)
    if not result:
        return ToolRun([], summary=f"jina failed {url}", success=False, errors=["jina_reader: empty or failed"], metadata={"read_backend": "jina_reader"})
    evidence = reader._jina_evidence(url, title, result)
    return ToolRun([evidence], summary=f"jina candidate {url}", metadata={"read_backend": "jina_reader"})


def run_scrapling_candidate(reader: RobustWebReadExecutor, url: str, title: str, snippet: str, args: argparse.Namespace) -> ToolRun:
    result = reader.scrapling_fetcher.fetch(
        url,
        max_chars=args.scrapling_max_chars,
        title=title,
        snippet=snippet,
    )
    if not result.evidence:
        return ToolRun([], summary=f"scrapling failed {url}", success=False, errors=[result.error or "scrapling: empty"], metadata={"read_backend": "scrapling"})
    evidence = reader._clean_evidence(
        result.evidence,
        title=title or url,
        snippet=snippet,
        max_chars=args.scrapling_max_chars,
        source_format="text",
    )
    evidence.title = title or evidence.title or url
    evidence.metadata["read_backend"] = "scrapling"
    return ToolRun([evidence], summary=f"scrapling candidate {url}", metadata={"read_backend": "scrapling"})


def print_backend_fetch(label: str, backend_label: str, read, args: argparse.Namespace) -> dict:
    read_backend = str(read.metadata.get("read_backend") or "")
    fetch_status = "FETCH_OK" if read_backend not in {"snippet_fallback", "failed", ""} and read.success else "FETCH_FAILED_WITH_SNIPPET_FALLBACK"
    prefix = f"[{label}][{backend_label}]"
    print("-" * 100)
    print(f"{prefix} success={read.success} summary={read.summary}")
    print(f"{prefix} status={fetch_status}")
    print(f"{prefix} read_backend={read_backend}")
    print(f"{prefix} errors={read.errors}")
    print(f"{prefix} metadata={read.metadata}")

    page_chars = 0
    quality_score = 0.0
    if read.evidence:
        page = read.evidence[0]
        page_chars = len(page.content)
        quality_score = float(page.metadata.get("quality_score") or 0.0)
        print(f"{prefix} evidence_source={page.source}")
        print(f"{prefix} evidence_title={page.title}")
        print(f"{prefix} evidence_metadata={page.metadata}")
        content = page.content if args.full_content else page.content[: args.content_chars]
        print(f"{prefix} content_chars_total={page_chars} printed={len(content)}")
        print(f"{prefix} quality_score={quality_score:.4f}")
        print(f"{prefix} BEGIN_CONTENT")
        print(content)
        print(f"{prefix} END_CONTENT")
    else:
        print(f"{prefix} no evidence")

    return {
        "read_backend": read_backend,
        "fetch_status": fetch_status,
        "success": read.success and bool(read.evidence),
        "page_chars": page_chars,
        "quality_score": quality_score,
        "errors": read.errors,
    }


def run_flow(query: str, searcher: APIWebSearchExecutor, reader: RobustWebReadExecutor, args: argparse.Namespace) -> list[dict]:
    print("\n" + "=" * 100)
    print(f"[query] {query}")
    search = searcher.run(query, limit=args.search_limit)
    print(f"[search] success={search.success} summary={search.summary}")
    print(f"[search] errors={search.errors}")
    print(f"[search] metadata={search.metadata}")
    print(f"[search] results={len(search.evidence)}")

    for idx, evidence in enumerate(search.evidence, 1):
        print("-" * 100)
        print(f"[result {idx}] title={evidence.title}")
        print(f"[result {idx}] url={evidence.source}")
        print(f"[result {idx}] score={evidence.score}")
        print(f"[result {idx}] metadata={evidence.metadata}")
        print(f"[result {idx}] snippet={evidence.content}")

    stats: list[dict] = []
    fetched = 0
    for idx, evidence in enumerate(search.evidence, 1):
        if fetched >= args.fetch_limit:
            break
        if is_pdf_like(evidence.source) and not args.include_pdf:
            print("-" * 100)
            print(f"[fetch skip result {idx}] pdf-like url={evidence.source}")
            continue

        fetched += 1
        print("\n" + "#" * 100)
        print(f"[fetch {fetched}] from result={idx}")
        print(f"[fetch {fetched}] title={evidence.title}")
        print(f"[fetch {fetched}] url={evidence.source}")
        if args.compare and not args.crawl4ai_only and not args.jina_only and not args.scrapling_only:
            stats.extend(run_compare_fetch(evidence.source, evidence.title, evidence.content, args, label=f"fetch {fetched}"))
            continue

        read = reader.run(evidence.source, evidence.title, evidence.content)
        read_backend = str(read.metadata.get("read_backend") or "")
        fetch_status = "FETCH_OK" if read_backend != "snippet_fallback" and read.success else "FETCH_FAILED_WITH_SNIPPET_FALLBACK"
        print(f"[fetch {fetched}] success={read.success} summary={read.summary}")
        print(f"[fetch {fetched}] status={fetch_status}")
        print(f"[fetch {fetched}] read_backend={read_backend}")
        print(f"[fetch {fetched}] errors={read.errors}")
        print(f"[fetch {fetched}] metadata={read.metadata}")

        page_chars = 0
        quality_score = 0.0
        if read.evidence:
            page = read.evidence[0]
            page_chars = len(page.content)
            quality_score = float(page.metadata.get("quality_score") or 0.0)
            print(f"[fetch {fetched}] evidence_source={page.source}")
            print(f"[fetch {fetched}] evidence_title={page.title}")
            print(f"[fetch {fetched}] evidence_metadata={page.metadata}")
            content = page.content if args.full_content else page.content[: args.content_chars]
            print(f"[fetch {fetched}] content_chars_total={page_chars} printed={len(content)}")
            print(f"[fetch {fetched}] quality_score={quality_score:.4f}")
            print(f"[fetch {fetched}] BEGIN_CONTENT")
            print(content)
            print(f"[fetch {fetched}] END_CONTENT")
        else:
            print(f"[fetch {fetched}] no evidence")

        stat = {
            "query": query,
            "url": evidence.source,
            "read_backend": read_backend,
            "fetch_status": fetch_status,
            "success": read.success,
            "page_chars": page_chars,
            "quality_score": quality_score,
            "errors": read.errors,
        }
        if read.metadata.get("compare_mode"):
            comparison = read.metadata.get("comparison", {})
            stat["compare_mode"] = True
            stat["crawl4ai_score"] = comparison.get("crawl4ai_score", 0)
            stat["crawl4ai_chars"] = comparison.get("crawl4ai_chars", 0)
            stat["jina_score"] = comparison.get("jina_score", 0)
            stat["jina_chars"] = comparison.get("jina_chars", 0)
            stat["scrapling_score"] = comparison.get("scrapling_score", 0)
            stat["scrapling_chars"] = comparison.get("scrapling_chars", 0)
            stat["winner"] = comparison.get("winner", "")
        stats.append(stat)

    if fetched == 0:
        print("[fetch] no pages fetched; all selected results were skipped or search returned nothing")
    return stats


def print_summary(stats: list[dict], compare_mode: bool) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    total = len(stats)
    ok = sum(1 for s in stats if s["fetch_status"] == "FETCH_OK")
    print(f"Total fetches: {total}, Success: {ok}, Snippet fallback: {total - ok}")

    backends: dict[str, list[dict]] = {}
    for s in stats:
        backends.setdefault(s["read_backend"], []).append(s)

    print(f"\nBackend distribution:")
    for backend, items in sorted(backends.items()):
        avg_chars = sum(i["page_chars"] for i in items) / max(len(items), 1)
        avg_score = sum(i["quality_score"] for i in items) / max(len(items), 1)
        print(f"  {backend}: {len(items)} fetches, avg_chars={avg_chars:.0f}, avg_quality={avg_score:.4f}")

    if compare_mode:
        compare_items = [s for s in stats if s.get("compare_mode")]
        if compare_items:
            c4a_wins = sum(1 for s in compare_items if s.get("winner") == "crawl4ai")
            jina_wins = sum(1 for s in compare_items if s.get("winner") == "jina")
            scrapling_wins = sum(1 for s in compare_items if s.get("winner") == "scrapling")
            c4a_scores = [s.get("crawl4ai_score", 0) for s in compare_items]
            jina_scores = [s.get("jina_score", 0) for s in compare_items]
            scrapling_scores = [s.get("scrapling_score", 0) for s in compare_items]
            c4a_chars = [s.get("crawl4ai_chars", 0) for s in compare_items]
            jina_chars = [s.get("jina_chars", 0) for s in compare_items]
            scrapling_chars = [s.get("scrapling_chars", 0) for s in compare_items]

            print(f"\n--- crawl4ai vs Jina vs Scrapling Comparison ({len(compare_items)} URLs) ---")
            print(f"  crawl4ai wins: {c4a_wins}, Jina wins: {jina_wins}, Scrapling wins: {scrapling_wins}")
            print(f"  crawl4ai avg_score:  {sum(c4a_scores)/len(c4a_scores):.4f}, avg_chars: {sum(c4a_chars)/len(c4a_chars):.0f}")
            print(f"  Jina avg_score:      {sum(jina_scores)/len(jina_scores):.4f}, avg_chars: {sum(jina_chars)/len(jina_chars):.0f}")
            print(f"  Scrapling avg_score: {sum(scrapling_scores)/len(scrapling_scores):.4f}, avg_chars: {sum(scrapling_chars)/len(scrapling_chars):.0f}")

            print(f"\n  Per-URL breakdown:")
            for s in compare_items:
                winner_mark = " <--" if True else ""
                c4a = s.get("crawl4ai_score", 0)
                jina = s.get("jina_score", 0)
                scrapling = s.get("scrapling_score", 0)
                w = s.get("winner", "?")
                print(f"    {s['url'][:80]}")
                print(
                    f"      crawl4ai: score={c4a:.4f} chars={s.get('crawl4ai_chars', 0):>5}"
                    f"  |  jina: score={jina:.4f} chars={s.get('jina_chars', 0):>5}"
                    f"  |  scrapling: score={scrapling:.4f} chars={s.get('scrapling_chars', 0):>5}"
                    f"  |  winner: {w}"
                )


def build_query_from_sample(question: str) -> str:
    first_line = next((line.strip() for line in question.splitlines() if line.strip()), question.strip())
    tokens = re.findall(
        r"\b(?:[A-Z]{1,5}\d{1,5}[A-Z]*|TL431A?|IRFZ44N|LM358|MOSFET|MOS|LED|LLC|UC384\d|R\d+|C\d+)\b",
        question,
    )
    suffix = " ".join(dict.fromkeys(tokens[:8]))
    query = f"{first_line} {suffix}".strip()
    return re.sub(r"\s+", " ", query)


def is_pdf_like(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(".pdf") or "/pdf/" in lowered or "/lit/" in lowered


def one_line(text: str, max_chars: int) -> str:
    return " ".join((text or "").split())[:max_chars]


if __name__ == "__main__":
    main()
