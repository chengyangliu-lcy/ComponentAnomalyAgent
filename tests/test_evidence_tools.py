from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from uuid import uuid4
import unittest
from unittest.mock import patch, AsyncMock, MagicMock

import yaml

from schemas import Evidence
from tools.browser import BrowserFetchResult
from tools.content_extractor import extract_llm_markdown
from tools.crawl4ai_fetcher import Crawl4AIConfig, Crawl4AIFetcher, Crawl4AIFetchResult
from tools.scrapling_fetcher import ScraplingFetchResult, ScraplingFetcher
from tools.evidence_tools import (
    APIWebSearchExecutor,
    DomainSkillExecutor,
    FinishAnswerExecutor,
    ImageInspectAction,
    ImageInspectExecutor,
    QwenSearchExecutor,
    RobustWebReadExecutor,
)


class FakeLLM:
    available = True

    def chat(self, messages, temperature=None):
        return SimpleNamespace(
            content='{"components":["R1"],"values":["10K"],"topology":"feedback"}',
            error=None,
        )


class CaptureLLM:
    available = True

    def __init__(self) -> None:
        self.messages = None

    def chat(self, messages, temperature=None, **kwargs):
        self.messages = messages
        return SimpleNamespace(content="结论：测试答案", error=None)


class CaptureSearchLLM:
    available = True

    def __init__(self) -> None:
        self.search_options = None

    def search_chat(self, messages, temperature=None, search_options=None):
        self.search_options = search_options
        return SimpleNamespace(content="搜索结果摘要", error=None, token_usage={"total_tokens": 12})


class FakeBrowser:
    def __init__(self, result: BrowserFetchResult) -> None:
        self.result = result
        self.calls: list[tuple[str, int | None]] = []

    def fetch(self, url: str, max_chars: int | None = None) -> BrowserFetchResult:
        self.calls.append((url, max_chars))
        return self.result


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code


class EvidenceToolTests(unittest.TestCase):
    def test_domain_skills_are_generic_and_do_not_embed_sample_points(self) -> None:
        skill_path = Path(__file__).resolve().parents[1] / "knowledge_base" / "domain_skills.yaml"
        text = skill_path.read_text(encoding="utf-8")
        forbidden_terms = [
            "R161",
            "C24",
            "R8",
            "R9",
            "C5",
            "CON1",
            "IRFZ44N",
            "TL431A",
            "205",
            "502",
            "0.5",
            "100k",
            "470k",
            "0.1mW",
            "1nF",
            "expected_terms",
        ]
        lowered = text.lower()
        for term in forbidden_terms:
            self.assertNotIn(term.lower(), lowered)

        payload = yaml.safe_load(text)
        self.assertIsInstance(payload.get("skills"), list)
        self.assertGreaterEqual(len(payload["skills"]), 4)
        for skill in payload["skills"]:
            self.assertEqual(
                sorted(skill.keys()),
                ["content", "id", "query_terms", "title", "triggers"],
            )

    def test_web_search_falls_back_to_html_when_api_key_missing(self) -> None:
        executor = APIWebSearchExecutor(
            provider_order=["tavily", "html"],
            api_key_envs={"tavily": "TAVILY_API_KEY"},
        )
        html_result = Evidence(
            source="https://example.com/a",
            title="RC compensation",
            content="feedback loop compensation",
            metadata={"kind": "web_search_result"},
        )
        executor.html_search.search = lambda query, limit=5: ([html_result], None)

        with patch.dict("os.environ", {}, clear=True):
            run = executor.run("feedback compensation", limit=3)

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["provider"], "html")
        self.assertEqual(run.evidence[0].source, "https://example.com/a")

    def test_web_search_uses_api_provider_before_html(self) -> None:
        executor = APIWebSearchExecutor(
            provider_order=["tavily", "html"],
            api_key_envs={"tavily": "TAVILY_API_KEY"},
        )
        executor._search_provider = lambda provider, key, query, limit: [
            Evidence(
                source="https://api.example.com",
                title="Current sense API result",
                content="current sense resistor content",
                metadata={"kind": "web_search_result", "provider": provider},
            )
        ]
        executor.html_search.search = lambda query, limit=5: ([], "should not be called")

        with patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"}, clear=True):
            run = executor.run("current sense", limit=3)

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["provider"], "tavily")
        self.assertEqual(run.evidence[0].source, "https://api.example.com")

    def test_web_search_prefers_direct_config_key_over_environment(self) -> None:
        executor = APIWebSearchExecutor(
            provider_order=["tavily", "html"],
            api_key_envs={"tavily": "TAVILY_API_KEY"},
            api_keys={"tavily": "direct-key"},
        )
        used_keys: list[str] = []

        def fake_provider(provider, key, query, limit):
            used_keys.append(key)
            return [
                Evidence(
                    source="https://direct.example.com",
                    title="Feedback compensation direct key result",
                    content="feedback compensation loop content",
                    metadata={"kind": "web_search_result", "provider": provider},
                )
            ]

        executor._search_provider = fake_provider

        with patch.dict("os.environ", {"TAVILY_API_KEY": "env-key"}, clear=True):
            run = executor.run("feedback compensation", limit=3)

        self.assertTrue(run.success)
        self.assertEqual(used_keys, ["direct-key"])
        self.assertEqual(run.evidence[0].source, "https://direct.example.com")

    def test_web_search_filters_low_value_pages_and_zero_overlap_results(self) -> None:
        executor = APIWebSearchExecutor(provider_order=["searxng"], api_key_envs={}, searxng_url="http://searxng")
        executor._search_searxng = lambda query, limit: [
            Evidence(
                source="https://jianghu.taobao.com/detail/47799_67930536",
                title="Lg电视遥控器平替居然能语音 工程模式 全系兼容？",
                content="欢迎来到淘宝网 店铺 商品规格 登录查看更多优惠",
                score=0.5,
                metadata={"kind": "web_search_result", "provider": "searxng"},
            ),
            Evidence(
                source="https://www.rcmoy.com/sitemap.html",
                title="网站地图 - 火狐电竞",
                content="unrelated product index",
                score=2.5,
                metadata={"kind": "web_search_result", "provider": "searxng"},
            ),
            Evidence(
                source="https://example.com/uc3842-repair",
                title="UC3842 MOS switch power supply repair",
                content="UC3842 PWM controller MOSFET startup resistor feedback current sense",
                score=0.0,
                metadata={"kind": "web_search_result", "provider": "searxng"},
            ),
        ]

        run = executor.run("UC3842 MOS switch power supply", limit=3)

        self.assertTrue(run.success)
        self.assertEqual([item.source for item in run.evidence], ["https://example.com/uc3842-repair"])
        self.assertGreater(run.evidence[0].score, 0.0)

    def test_content_extractor_rejects_sitemap_and_mojibake(self) -> None:
        sitemap = "\n".join(
            f"[product {idx}](https://example.com/product/{idx}) | 0.8 | Daily | 2026-01-01"
            for idx in range(40)
        )
        sitemap_result = extract_llm_markdown(sitemap, title="网站地图", snippet="UC3842 MOS")
        self.assertEqual(sitemap_result.quality_score, 0.0)
        self.assertEqual(sitemap_result.quality_reason, "site_map_or_link_index")

        mojibake = " ".join(["é¦–é¡µ æœŸåˆŠ æ‚å¿— å¼€å…³ç”µæº èŠ¯ç‰‡"] * 80)
        mojibake_result = extract_llm_markdown(mojibake, title="开关电源芯片", snippet="UC3842")
        self.assertEqual(mojibake_result.quality_score, 0.0)
        self.assertEqual(mojibake_result.quality_reason, "mojibake_or_wrong_encoding")

    def test_content_extractor_prefers_article_body_over_navigation_boilerplate(self) -> None:
        html = """
        <html><body>
          <nav><a>Home</a><a>Products</a><a>Login</a></nav>
          <aside>Related posts advertisement subscribe newsletter</aside>
          <article>
            <h1>LM358 LED dimming fault</h1>
            <p>The LM358 op amp output swing and input offset voltage should be checked in this LED dimming circuit.</p>
            <p>The potentiometer voltage changes, so inspect the MOSFET gate drive, feedback path, and current limiting resistor.</p>
          </article>
          <footer>privacy policy terms of use</footer>
        </body></html>
        """

        result = extract_llm_markdown(
            html,
            title="LM358 LED dimming fault",
            snippet="LM358 potentiometer MOSFET LED dimming",
            source_format="html",
        )

        self.assertIn(result.extractor, {"readability", "trafilatura", "bs4_main_text"})
        self.assertIn("LM358 op amp output swing", result.content)
        self.assertIn("MOSFET gate drive", result.content)
        self.assertNotIn("Related posts advertisement", result.content)
        self.assertIn("candidate_scores", result.metadata)

    def test_web_read_keeps_snippet_on_page_failure(self) -> None:
        executor = RobustWebReadExecutor(enable_openhands_browser_primary=False, enable_jina_reader=False, enable_crawl4ai=False)
        executor.reader.read = lambda url, max_chars=6000: SimpleNamespace(evidence=None, error="403 forbidden")

        run = executor.run("https://example.com/blocked", "Blocked page", "important snippet")

        self.assertTrue(run.success)
        self.assertIn("important snippet", run.evidence[0].content)
        self.assertEqual(run.evidence[0].metadata["read_error"], "403 forbidden")
        self.assertEqual(run.evidence[0].metadata["read_confidence"], "low")
        self.assertFalse(run.evidence[0].metadata["effective_read_success"])

    def test_web_read_prefers_openhands_browser(self) -> None:
        browser_evidence = Evidence(
            source="https://example.com/page",
            title="Browser page",
            content="browser extracted content",
            metadata={"kind": "openhands_browser_page"},
        )
        browser = FakeBrowser(BrowserFetchResult(browser_evidence))
        executor = RobustWebReadExecutor(openhands_browser=browser, enable_jina_reader=False, enable_crawl4ai=False)
        executor.reader.read = lambda url, max_chars=6000: self.fail("WebReader should not be called")

        run = executor.run("https://example.com/page", "Preferred title", "snippet")

        self.assertTrue(run.success)
        self.assertEqual(run.evidence[0].content, "browser extracted content")
        self.assertEqual(run.evidence[0].title, "Preferred title")
        self.assertEqual(run.evidence[0].metadata["kind"], "openhands_browser_page")
        self.assertEqual(run.metadata["read_backend"], "openhands_browser")
        self.assertEqual(browser.calls, [("https://example.com/page", 6000)])

    def test_web_read_uses_playwright_browser_fallback(self) -> None:
        browser_evidence = Evidence(
            source="https://example.com/page",
            title="Rendered page",
            content="LLC ripple thread rendered by browser with feedback loop replies.",
            metadata={"kind": "browser_page"},
        )
        browser = FakeBrowser(BrowserFetchResult(browser_evidence))
        openhands = FakeBrowser(BrowserFetchResult(None, "should not be called"))
        executor = RobustWebReadExecutor(
            openhands_browser=openhands,
            browser_fallback=browser,
            enable_browser_fallback=True,
            enable_jina_reader=False,
            enable_crawl4ai=False,
        )
        executor.reader.read = lambda url, max_chars=6000: self.fail("WebReader should not be called")

        run = executor.run("https://example.com/page", "LLC page", "LLC ripple")

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["read_backend"], "playwright_browser")
        self.assertEqual(run.evidence[0].metadata["read_backend"], "playwright_browser")
        self.assertEqual(browser.calls, [("https://example.com/page", 6000)])
        self.assertEqual(openhands.calls, [])

    def test_web_read_falls_back_to_requests_when_openhands_unavailable(self) -> None:
        browser = FakeBrowser(BrowserFetchResult(None, "chromium missing"))
        executor = RobustWebReadExecutor(openhands_browser=browser, enable_jina_reader=False, enable_crawl4ai=False)
        executor.reader.read = lambda url, max_chars=6000: SimpleNamespace(
            evidence=Evidence(
                source=url,
                title="Requests page",
                content="requests content",
                metadata={"kind": "web_page"},
            ),
            error=None,
        )

        run = executor.run("https://example.com/readable", "Readable page", "snippet")

        self.assertTrue(run.success)
        self.assertEqual(run.evidence[0].metadata["read_backend"], "requests_bs4")
        self.assertEqual(run.evidence[0].metadata["openhands_error"], "chromium missing")
        self.assertEqual(run.metadata["read_backend"], "requests_bs4")

    def test_web_read_keeps_snippet_after_openhands_and_requests_fail(self) -> None:
        browser = FakeBrowser(BrowserFetchResult(None, "browser failed"))
        executor = RobustWebReadExecutor(openhands_browser=browser, enable_jina_reader=False, enable_crawl4ai=False)
        executor.reader.read = lambda url, max_chars=6000: SimpleNamespace(evidence=None, error="403 forbidden")

        run = executor.run("https://example.com/blocked", "Blocked page", "important snippet")

        self.assertTrue(run.success)
        self.assertEqual(run.evidence[0].metadata["read_backend"], "snippet_fallback")
        self.assertEqual(run.evidence[0].metadata["openhands_error"], "browser failed")
        self.assertEqual(run.evidence[0].metadata["web_reader_error"], "403 forbidden")
        self.assertEqual(run.evidence[0].metadata["read_confidence"], "low")
        self.assertFalse(run.evidence[0].metadata["effective_read_success"])
        self.assertIn("browser failed", run.errors)
        self.assertIn("403 forbidden", run.errors)

    def test_web_read_skips_openhands_for_pdf_or_known_slow_pages(self) -> None:
        browser = FakeBrowser(BrowserFetchResult(None, "should not be called"))
        executor = RobustWebReadExecutor(openhands_browser=browser, enable_jina_reader=False, enable_crawl4ai=False)
        executor.reader.read = lambda url, max_chars=6000: self.fail("WebReader should not be called")

        run = executor.run("https://example.com/file.pdf", "PDF", "pdf snippet")

        self.assertTrue(run.success)
        self.assertEqual(browser.calls, [])
        self.assertEqual(run.evidence[0].metadata["read_backend"], "snippet_fallback")
        self.assertEqual(run.evidence[0].metadata["read_skipped"], "pdf_binary_or_known_slow")
        self.assertEqual(run.evidence[0].metadata["read_confidence"], "low")
        self.assertFalse(run.evidence[0].metadata["effective_read_success"])

    def test_web_read_falls_back_when_jina_content_is_low_quality(self) -> None:
        low_quality_jina = "\n".join(["Products Support Resources Applications Company Menu"] * 40)
        browser_evidence = Evidence(
            source="https://example.com/page",
            title="Browser page",
            content="LM358 input offset voltage and supply current details for comparator circuit diagnosis.",
            metadata={"kind": "openhands_browser_page"},
        )
        browser = FakeBrowser(BrowserFetchResult(browser_evidence))
        executor = RobustWebReadExecutor(
            openhands_browser=browser,
            jina_use_readerlm_fallback=False,
            jina_min_clean_chars=120,
            jina_min_quality_score=0.55,
            enable_crawl4ai=False,
        )
        executor.reader.read = lambda url, max_chars=6000: self.fail("WebReader should not be called")

        with patch("tools.evidence_tools.requests.get", return_value=FakeResponse(low_quality_jina)):
            run = executor.run(
                "https://example.com/page",
                "LM358 datasheet",
                "LM358 input offset voltage supply current",
            )

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["read_backend"], "openhands_browser")
        self.assertIn("jina_reader: low quality", run.evidence[0].metadata["jina_error"])
        self.assertEqual(browser.calls, [("https://example.com/page", 6000)])

    def test_web_read_uses_jina_readerlm_when_markdown_quality_is_low(self) -> None:
        low_quality_jina = "\n".join(["Products Support Resources Applications Company Menu"] * 40)
        readerlm_content = "\n".join(
            [
                "LM358 dual operational amplifier datasheet",
                "The LM358 device includes two independent high-gain operational amplifiers.",
                "It specifies input offset voltage, supply current, common-mode input voltage, and output swing.",
                "This information is useful for comparator circuit and power supply diagnosis.",
            ]
            * 8
        )
        executor = RobustWebReadExecutor(
            enable_openhands_browser_primary=False,
            jina_use_readerlm_fallback=True,
            jina_min_clean_chars=120,
            jina_min_quality_score=0.35,
            enable_crawl4ai=False,
        )
        executor.reader.read = lambda url, max_chars=6000: self.fail("WebReader should not be called")

        with patch(
            "tools.evidence_tools.requests.get",
            side_effect=[FakeResponse(low_quality_jina), FakeResponse(readerlm_content)],
        ) as mocked_get:
            run = executor.run(
                "https://example.com/lm358",
                "LM358 datasheet",
                "LM358 input offset voltage supply current",
            )

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["read_backend"], "jina_reader")
        self.assertIn("LM358", run.evidence[0].content)
        self.assertIn("readerlm_v2", run.evidence[0].metadata["extractor"])
        self.assertEqual(mocked_get.call_count, 2)

    def test_image_inspect_uses_mock_llm(self) -> None:
        with _workspace_tempdir() as tmp:
            image_path = Path(tmp) / "image.png"
            image_path.write_bytes(b"not-a-real-image-but-readable")
            executor = ImageInspectExecutor(FakeLLM(), enabled=True)

            run = executor.run(
                ImageInspectAction(
                    sample_id="sample",
                    question="R1 起什么作用",
                    image_paths=[str(image_path)],
                )
            )

        self.assertTrue(run.success)
        self.assertGreaterEqual(len(run.evidence), 2)
        self.assertIn("components", run.evidence[-1].content)

    def test_domain_skill_rejects_forbidden_test_root(self) -> None:
        with _workspace_tempdir() as tmp:
            root = Path(tmp)
            skill_path = root / "2025" / "domain_skills.yaml"
            skill_path.parent.mkdir()
            skill_path.write_text("skills: []", encoding="utf-8")

            with self.assertRaises(ValueError):
                DomainSkillExecutor(skill_path, forbidden_roots=[root / "2025"])

    def test_domain_skill_loads_clean_generic_skill_file(self) -> None:
        skill_path = Path(__file__).resolve().parents[1] / "knowledge_base" / "domain_skills.yaml"
        executor = DomainSkillExecutor(skill_path, forbidden_roots=[])

        run = executor.run("开关电源反馈环路纹波异常，应该怎么排查")

        self.assertTrue(run.success)
        self.assertTrue(all("expected_terms" not in item.metadata for item in run.evidence))
        self.assertTrue(any(item.source.startswith("domain_skill:") for item in run.evidence))

    def test_qwen_search_passes_forced_search_options_to_llm(self) -> None:
        llm = CaptureSearchLLM()
        executor = QwenSearchExecutor(llm, enabled=True, search_options={"forced_search": True})

        run = executor.run("TL431 feedback abnormal")

        self.assertTrue(run.success)
        self.assertEqual(llm.search_options, {"forced_search": True})
        self.assertEqual(run.evidence[0].metadata["search_options"], {"forced_search": True})
        self.assertTrue(run.evidence[0].metadata["forced_search"])

    def test_finish_answer_compacts_and_deduplicates_evidence(self) -> None:
        llm = CaptureLLM()
        executor = FinishAnswerExecutor(llm, max_evidence=4)
        duplicate = Evidence(
            source="domain_skill:test",
            title="Skill",
            content="feedback compensation " * 80,
            metadata={"kind": "domain_skill"},
        )
        image = Evidence(
            source="image_inspect",
            title="Image",
            content="image detail " * 120,
            metadata={"kind": "image_inspection"},
        )

        run = executor.run("反馈纹波怎么处理", [duplicate, duplicate, image])

        self.assertTrue(run.success)
        user_text = llm.messages[1]["content"]
        self.assertEqual(user_text.count("domain_skill:test"), 1)
        self.assertIn("类型:domain_skill", user_text)
        self.assertIn("类型:image_inspection", user_text)
        self.assertLess(len(user_text), 3000)


class FakeCrawl4AIFetcher:
    def __init__(self, result: Crawl4AIFetchResult) -> None:
        self.result = result
        self.calls: list[tuple[str, int | None]] = []

    def fetch(self, url: str, max_chars: int | None = None, **kwargs) -> Crawl4AIFetchResult:
        self.calls.append((url, max_chars))
        return self.result


class FakeScraplingFetcher:
    def __init__(self, result: ScraplingFetchResult) -> None:
        self.result = result
        self.calls: list[tuple[str, int | None]] = []

    def fetch(self, url: str, max_chars: int | None = None, **kwargs) -> ScraplingFetchResult:
        self.calls.append((url, max_chars))
        return self.result


class SequencedFetcher:
    def __init__(self, results) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, int | None]] = []

    def fetch(self, url: str, max_chars: int | None = None, **kwargs):
        self.calls.append((url, max_chars))
        if not self.results:
            raise AssertionError("no result queued")
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class Crawl4AITests(unittest.TestCase):
    def test_scrapling_fetch_blocks_localhost(self) -> None:
        fetcher = ScraplingFetcher()
        result = fetcher.fetch("http://localhost/test")
        self.assertIsNotNone(result.error)
        self.assertIn("blocked", result.error)
        self.assertIsNone(result.evidence)

    def test_crawl4ai_fetch_blocks_localhost(self) -> None:
        fetcher = Crawl4AIFetcher()
        result = fetcher.fetch("http://localhost/test")
        self.assertIsNotNone(result.error)
        self.assertIn("blocked", result.error)
        self.assertIsNone(result.evidence)

    def test_crawl4ai_fetch_blocks_private_ip(self) -> None:
        fetcher = Crawl4AIFetcher()
        for url in ["http://192.168.1.1/test", "http://10.0.0.1/test", "http://172.16.0.1/test"]:
            result = fetcher.fetch(url)
            self.assertIsNotNone(result.error, f"should block {url}")
            self.assertIn("blocked", result.error)

    def test_crawl4ai_blocks_non_http(self) -> None:
        fetcher = Crawl4AIFetcher()
        result = fetcher.fetch("ftp://example.com/file")
        self.assertIsNotNone(result.error)
        self.assertIn("blocked", result.error)

    def test_crawl4ai_best_markdown_prefers_richer_raw_content(self) -> None:
        fetcher = Crawl4AIFetcher(Crawl4AIConfig(markdown_mode="best"))
        selected = fetcher._select_markdown(
            fit_markdown="Login Register Menu",
            raw_markdown="\n".join(
                [
                    "LM358 dual operational amplifier datasheet",
                    "Input offset voltage, supply current, output swing, and comparator circuit notes.",
                    "Feedback loop compensation details for LED dimming repair.",
                ]
                * 8
            ),
            title="LM358 datasheet",
            snippet="input offset voltage supply current",
            max_chars=6000,
        )

        self.assertIn("Input offset voltage", selected)

    def test_web_read_prefers_jina_and_skips_crawl4ai_when_jina_is_good(self) -> None:
        crawl4ai = FakeCrawl4AIFetcher(Crawl4AIFetchResult(error="should not be called"))
        executor = RobustWebReadExecutor(
            crawl4ai_fetcher=crawl4ai,
            enable_openhands_browser_primary=False,
            jina_min_clean_chars=120,
            jina_min_quality_score=0.35,
        )
        jina_content = "\n".join(
            [
                "LM358 dual operational amplifier datasheet",
                "The LM358 device specifies input offset voltage and supply current.",
                "This content is useful for comparator and LED dimming fault diagnosis.",
            ]
            * 8
        )

        with patch("tools.evidence_tools.requests.get", return_value=FakeResponse(jina_content)):
            run = executor.run("https://example.com/lm358", "LM358 datasheet", "LM358 input offset voltage")

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["read_backend"], "jina_reader")
        self.assertEqual(crawl4ai.calls, [])

    def test_web_read_uses_crawl4ai_when_configured_as_primary(self) -> None:
        crawl4ai_evidence = Evidence(
            source="https://example.com/page",
            title="Crawl4AI page",
            content="LM358 dual op-amp datasheet with feedback loop compensation details.",
            metadata={"kind": "web_page"},
        )
        crawl4ai = FakeCrawl4AIFetcher(Crawl4AIFetchResult(evidence=crawl4ai_evidence, quality_score=0.8))
        openhands = FakeBrowser(BrowserFetchResult(None, "should not be called"))
        executor = RobustWebReadExecutor(
            openhands_browser=openhands,
            crawl4ai_fetcher=crawl4ai,
            enable_jina_reader=False,
            crawl4ai_primary=True,
        )
        executor.reader.read = lambda url, max_chars=6000: self.fail("WebReader should not be called")

        run = executor.run("https://example.com/page", "LM358 datasheet", "LM358 op-amp")

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["read_backend"], "crawl4ai")
        self.assertEqual(run.evidence[0].metadata["read_backend"], "crawl4ai")
        self.assertEqual(crawl4ai.calls, [("https://example.com/page", 6000)])
        self.assertEqual(openhands.calls, [])

    def test_web_read_falls_back_from_crawl4ai_to_jina(self) -> None:
        crawl4ai = FakeCrawl4AIFetcher(Crawl4AIFetchResult(error="crawl4ai: timeout"))
        openhands = FakeBrowser(BrowserFetchResult(None, "should not be called"))
        executor = RobustWebReadExecutor(
            openhands_browser=openhands,
            crawl4ai_fetcher=crawl4ai,
            enable_jina_reader=False,
            enable_openhands_browser_primary=False,
        )
        good_content = "LM358 dual operational amplifier datasheet with input offset voltage and supply current specifications."
        executor.reader.read = lambda url, max_chars=6000: SimpleNamespace(
            evidence=Evidence(source=url, title="page", content=good_content, metadata={"kind": "web_page"}),
            error=None,
        )

        run = executor.run("https://example.com/page", "LM358", "LM358 datasheet")

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["read_backend"], "requests_bs4")
        self.assertEqual(run.evidence[0].metadata["crawl4ai_error"], "crawl4ai: timeout")

    def test_web_read_skips_crawl4ai_when_disabled(self) -> None:
        crawl4ai = FakeCrawl4AIFetcher(Crawl4AIFetchResult(evidence=Evidence(
            source="https://example.com", title="t", content="content", metadata={},
        ), quality_score=0.8))
        openhands = FakeBrowser(BrowserFetchResult(None, "should not be called"))
        executor = RobustWebReadExecutor(
            openhands_browser=openhands,
            crawl4ai_fetcher=crawl4ai,
            enable_crawl4ai=False,
            enable_jina_reader=False,
            enable_openhands_browser_primary=False,
        )
        good_content = "LM358 dual operational amplifier datasheet with input offset voltage specifications."
        executor.reader.read = lambda url, max_chars=6000: SimpleNamespace(
            evidence=Evidence(source=url, title="page", content=good_content, metadata={"kind": "web_page"}),
            error=None,
        )

        run = executor.run("https://example.com/page", "LM358", "LM358 datasheet")

        self.assertTrue(run.success)
        self.assertEqual(crawl4ai.calls, [])
        self.assertEqual(run.metadata["read_backend"], "requests_bs4")

    def test_web_read_uses_scrapling_after_crawl4ai_failure(self) -> None:
        crawl4ai = FakeCrawl4AIFetcher(Crawl4AIFetchResult(error="crawl4ai: timeout"))
        scrapling_evidence = Evidence(
            source="https://example.com/page",
            title="Scrapling page",
            content="LM358 dual operational amplifier datasheet input offset voltage comparator circuit.",
            metadata={"kind": "web_page"},
        )
        scrapling = FakeScraplingFetcher(ScraplingFetchResult(evidence=scrapling_evidence, quality_score=0.8))
        executor = RobustWebReadExecutor(
            crawl4ai_fetcher=crawl4ai,
            scrapling_fetcher=scrapling,
            enable_scrapling=True,
            enable_jina_reader=False,
            enable_openhands_browser_primary=False,
        )
        executor.reader.read = lambda url, max_chars=6000: self.fail("WebReader should not be called")

        run = executor.run("https://example.com/page", "LM358", "LM358 datasheet")

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["read_backend"], "scrapling")
        self.assertEqual(run.evidence[0].metadata["read_backend"], "scrapling")
        self.assertEqual(run.evidence[0].metadata["crawl4ai_error"], "crawl4ai: timeout")
        self.assertEqual(scrapling.calls, [("https://example.com/page", 8000)])

    def test_web_read_compare_mode_logs_comparison(self) -> None:
        crawl4ai_evidence = Evidence(
            source="https://example.com/page",
            title="Crawl4AI page",
            content="LM358 dual op-amp datasheet with feedback loop compensation details and voltage specifications.",
            metadata={"kind": "web_page"},
        )
        crawl4ai = FakeCrawl4AIFetcher(Crawl4AIFetchResult(evidence=crawl4ai_evidence, quality_score=0.8))
        openhands = FakeBrowser(BrowserFetchResult(None, "unused"))

        with _workspace_tempdir() as tmp:
            comparison_path = str(Path(tmp) / "comparison.jsonl")
            executor = RobustWebReadExecutor(
                openhands_browser=openhands,
                crawl4ai_fetcher=crawl4ai,
                enable_jina_reader=False,
                web_read_compare_mode=True,
                backend_comparison_path=comparison_path,
            )

            with patch("tools.evidence_tools.requests.get", return_value=FakeResponse(
                "LM358 dual operational amplifier datasheet input offset voltage supply current comparator circuit."
            )):
                run = executor.run("https://example.com/page", "LM358", "LM358 datasheet")

            self.assertTrue(run.success)
            self.assertTrue(run.evidence[0].metadata.get("compare_mode"))
            self.assertEqual(Path(comparison_path).exists(), True)
            lines = Path(comparison_path).read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["url"], "https://example.com/page")
            self.assertIn("crawl4ai_score", record)
            self.assertIn("jina_score", record)
            self.assertIn("scrapling_score", record)
            self.assertIn("winner", record)

    def test_web_read_compare_mode_marks_both_zero_scores_as_tie_failed(self) -> None:
        crawl4ai = FakeCrawl4AIFetcher(Crawl4AIFetchResult(error="crawl4ai: empty"))
        executor = RobustWebReadExecutor(
            crawl4ai_fetcher=crawl4ai,
            enable_jina_reader=True,
            web_read_compare_mode=True,
        )

        with patch("tools.evidence_tools.requests.get", return_value=FakeResponse("short", status_code=500)):
            run = executor.run("https://example.com/empty", "Empty", "snippet")

        self.assertFalse(run.success)
        self.assertEqual(run.metadata["comparison"]["winner"], "tie_failed")

    def test_web_read_competitive_mode_selects_highest_scored_backend(self) -> None:
        crawl4ai_evidence = Evidence(
            source="https://example.com/page",
            title="c4a",
            content="Login Register Menu Products Support " * 30,
            metadata={"kind": "web_page"},
        )
        scrapling_evidence = Evidence(
            source="https://example.com/page",
            title="scrapling",
            content=(
                "LM358 LED dimming circuit repair. The MOSFET gate drive, feedback resistor, "
                "input offset voltage, output swing, and current limiting resistor are useful diagnostics. "
            )
            * 10,
            metadata={"kind": "web_page"},
        )
        crawl4ai = FakeCrawl4AIFetcher(Crawl4AIFetchResult(evidence=crawl4ai_evidence, quality_score=0.4))
        scrapling = FakeScraplingFetcher(ScraplingFetchResult(evidence=scrapling_evidence, quality_score=0.8))
        executor = RobustWebReadExecutor(
            crawl4ai_fetcher=crawl4ai,
            scrapling_fetcher=scrapling,
            enable_scrapling=True,
            enable_openhands_browser_primary=False,
            web_read_competitive_mode=True,
            web_read_min_quality_score=0.2,
            web_read_min_clean_chars=100,
        )
        jina_content = (
            "LM358 datasheet generic page with voltage and current notes. "
            "Products Support Resources Company Menu "
        ) * 10

        with patch("tools.evidence_tools.requests.get", return_value=FakeResponse(jina_content)):
            run = executor.run("https://example.com/page", "LM358 LED dimming", "MOSFET feedback input offset voltage")

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["read_backend"], "scrapling_dynamic")
        self.assertEqual(run.evidence[0].metadata["competitive_winner"], "scrapling_dynamic")
        self.assertEqual(set(run.evidence[0].metadata["completed_order"]), {"jina_reader", "crawl4ai", "scrapling_dynamic"})
        self.assertEqual(len(run.evidence[0].metadata["backend_scores"]), 3)

    def test_web_read_competitive_mode_falls_back_when_all_candidates_low_quality(self) -> None:
        crawl4ai = FakeCrawl4AIFetcher(Crawl4AIFetchResult(error="crawl4ai: empty"))
        scrapling = FakeScraplingFetcher(ScraplingFetchResult(error="scrapling: empty"))
        openhands = FakeBrowser(BrowserFetchResult(None, "should not be called"))
        executor = RobustWebReadExecutor(
            openhands_browser=openhands,
            crawl4ai_fetcher=crawl4ai,
            scrapling_fetcher=scrapling,
            enable_scrapling=True,
            enable_openhands_browser_primary=False,
            enable_browser_fallback=False,
            web_read_competitive_mode=True,
            web_read_min_quality_score=0.9,
            web_read_min_clean_chars=5000,
        )
        executor.reader.read = lambda url, max_chars=6000: SimpleNamespace(
            evidence=Evidence(source=url, title="page", content="LM358 input offset voltage and current notes." * 20, metadata={"kind": "web_page"}),
            error=None,
        )

        with patch("tools.evidence_tools.requests.get", return_value=FakeResponse("short", status_code=500)):
            run = executor.run("https://example.com/page", "LM358", "snippet")

        self.assertTrue(run.success)
        self.assertEqual(run.metadata["read_backend"], "requests_bs4")


@contextmanager
def _workspace_tempdir():
    base = Path(__file__).resolve().parents[1] / "outputs" / "test_evidence_tools"
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    yield str(path)


if __name__ == "__main__":
    unittest.main()
