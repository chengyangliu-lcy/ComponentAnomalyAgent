from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
from uuid import uuid4
import unittest
from unittest.mock import patch

import yaml

from schemas import Evidence
from tools.browser import BrowserFetchResult
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

    def chat(self, messages, temperature=None):
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
                title="API result",
                content="api content",
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
                    title="Direct key result",
                    content="api content",
                    metadata={"kind": "web_search_result", "provider": provider},
                )
            ]

        executor._search_provider = fake_provider

        with patch.dict("os.environ", {"TAVILY_API_KEY": "env-key"}, clear=True):
            run = executor.run("feedback compensation", limit=3)

        self.assertTrue(run.success)
        self.assertEqual(used_keys, ["direct-key"])
        self.assertEqual(run.evidence[0].source, "https://direct.example.com")

    def test_web_read_keeps_snippet_on_page_failure(self) -> None:
        executor = RobustWebReadExecutor(enable_openhands_browser_primary=False)
        executor.reader.read = lambda url, max_chars=6000: SimpleNamespace(evidence=None, error="403 forbidden")

        run = executor.run("https://example.com/blocked", "Blocked page", "important snippet")

        self.assertTrue(run.success)
        self.assertIn("important snippet", run.evidence[0].content)
        self.assertEqual(run.evidence[0].metadata["read_error"], "403 forbidden")

    def test_web_read_prefers_openhands_browser(self) -> None:
        browser_evidence = Evidence(
            source="https://example.com/page",
            title="Browser page",
            content="browser extracted content",
            metadata={"kind": "openhands_browser_page"},
        )
        browser = FakeBrowser(BrowserFetchResult(browser_evidence))
        executor = RobustWebReadExecutor(openhands_browser=browser)
        executor.reader.read = lambda url, max_chars=6000: self.fail("WebReader should not be called")

        run = executor.run("https://example.com/page", "Preferred title", "snippet")

        self.assertTrue(run.success)
        self.assertEqual(run.evidence[0].content, "browser extracted content")
        self.assertEqual(run.evidence[0].title, "Preferred title")
        self.assertEqual(run.evidence[0].metadata["kind"], "openhands_browser_page")
        self.assertEqual(run.metadata["read_backend"], "openhands_browser")
        self.assertEqual(browser.calls, [("https://example.com/page", 6000)])

    def test_web_read_falls_back_to_requests_when_openhands_unavailable(self) -> None:
        browser = FakeBrowser(BrowserFetchResult(None, "chromium missing"))
        executor = RobustWebReadExecutor(openhands_browser=browser)
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
        executor = RobustWebReadExecutor(openhands_browser=browser)
        executor.reader.read = lambda url, max_chars=6000: SimpleNamespace(evidence=None, error="403 forbidden")

        run = executor.run("https://example.com/blocked", "Blocked page", "important snippet")

        self.assertTrue(run.success)
        self.assertEqual(run.evidence[0].metadata["read_backend"], "snippet_fallback")
        self.assertEqual(run.evidence[0].metadata["openhands_error"], "browser failed")
        self.assertEqual(run.evidence[0].metadata["web_reader_error"], "403 forbidden")
        self.assertIn("browser failed", run.errors)
        self.assertIn("403 forbidden", run.errors)

    def test_web_read_skips_openhands_for_pdf_or_known_slow_pages(self) -> None:
        browser = FakeBrowser(BrowserFetchResult(None, "should not be called"))
        executor = RobustWebReadExecutor(openhands_browser=browser)
        executor.reader.read = lambda url, max_chars=6000: self.fail("WebReader should not be called")

        run = executor.run("https://example.com/file.pdf", "PDF", "pdf snippet")

        self.assertTrue(run.success)
        self.assertEqual(browser.calls, [])
        self.assertEqual(run.evidence[0].metadata["read_backend"], "snippet_fallback")
        self.assertEqual(run.evidence[0].metadata["read_skipped"], "pdf_binary_or_known_slow")

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

@contextmanager
def _workspace_tempdir():
    base = Path(__file__).resolve().parents[1] / "outputs" / "test_evidence_tools"
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    yield str(path)


if __name__ == "__main__":
    unittest.main()
