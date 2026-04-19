from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class LLMResponse:
    content: str
    token_usage: Dict[str, Any]
    error: Optional[str] = None


class LLMClient:
    """Small OpenAI-compatible client with an explicit no-key fallback."""

    def __init__(
        self,
        api_key: str | None,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: float | None = None,
        max_retries: int = 0,
        extra_body: Dict[str, Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_body = dict(extra_body or {})
        self._client = (
            OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
            if api_key
            else None
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float | None = None,
        response_format: Dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self._client:
            return LLMResponse(
                content="",
                token_usage={},
                error="LLM API key is not configured; fallback logic was used.",
            )
        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": self.max_tokens,
            }
            if response_format:
                payload["response_format"] = response_format
            if self.extra_body:
                payload["extra_body"] = self.extra_body
            response = self._client.chat.completions.create(**payload)
            usage = getattr(response, "usage", None)
            token_usage = usage.model_dump() if hasattr(usage, "model_dump") else {}
            return LLMResponse(
                content=(response.choices[0].message.content or "").strip(),
                token_usage=token_usage,
            )
        except Exception as exc:  # noqa: BLE001
            return LLMResponse(content="", token_usage={}, error=str(exc))

    def json_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.1,
        response_format: Dict[str, Any] | None = None,
    ) -> tuple[Dict[str, Any], Optional[str]]:
        response = self.chat(messages, temperature=temperature, response_format=response_format)
        if response.error:
            return {}, response.error
        text = response.content.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                text = text[start:end]
            return json.loads(text), None
        except Exception as exc:  # noqa: BLE001
            return {}, f"failed to parse JSON from judge response: {exc}; raw={response.content[:200]}"
