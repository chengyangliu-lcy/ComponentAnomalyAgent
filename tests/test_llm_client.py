from __future__ import annotations

from types import SimpleNamespace
import unittest

from llm_client import LLMClient


class FakeCompletions:
    def __init__(self) -> None:
        self.payload = None

    def create(self, **payload):
        self.payload = payload
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 5}),
        )


class LLMClientSearchTests(unittest.TestCase):
    def test_search_chat_merges_search_options_into_extra_body(self) -> None:
        completions = FakeCompletions()
        client = LLMClient(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            extra_body={"search_options": {"search_strategy": "agent"}},
        )
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

        response = client.search_chat(
            [{"role": "user", "content": "query"}],
            search_options={"forced_search": True},
        )

        self.assertEqual(response.content, "ok")
        self.assertEqual(
            completions.payload["extra_body"],
            {
                "enable_search": True,
                "search_options": {
                    "search_strategy": "agent",
                    "forced_search": True,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
