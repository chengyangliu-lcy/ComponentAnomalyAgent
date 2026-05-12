from __future__ import annotations

import json
import re
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
        self._total_calls: int = 0
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._total_tokens: int = 0

    def _accumulate(self, token_usage: Dict[str, Any]) -> None:
        self._total_calls += 1
        self._total_prompt_tokens += int(token_usage.get("prompt_tokens", 0))
        self._total_completion_tokens += int(token_usage.get("completion_tokens", 0))
        self._total_tokens += int(token_usage.get("total_tokens", 0))

    @property
    def cumulative_usage(self) -> Dict[str, Any]:
        return {
            "calls": self._total_calls,
            "prompt_tokens": self._total_prompt_tokens,
            "completion_tokens": self._total_completion_tokens,
            "total_tokens": self._total_tokens,
        }

    @property
    def available(self) -> bool:
        return self._client is not None

    def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float | None = None,
        response_format: Dict[str, Any] | None = None,
        max_tokens: int | None = None,
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
                "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            }
            if response_format:
                payload["response_format"] = response_format
            if self.extra_body:
                payload["extra_body"] = self.extra_body
            response = self._client.chat.completions.create(**payload)
            usage = getattr(response, "usage", None)
            token_usage = usage.model_dump() if hasattr(usage, "model_dump") else {}
            self._accumulate(token_usage)
            return LLMResponse(
                content=(response.choices[0].message.content or "").strip(),
                token_usage=token_usage,
            )
        except Exception as exc:  # noqa: BLE001
            return LLMResponse(content="", token_usage={}, error=str(exc))

    def search_chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        search_options: Dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Call the LLM with DashScope's built-in internet search enabled.

        Uses extra_body={'enable_search': True} to activate the model's
        native web search capability. The model will automatically search
        the internet and include results in its response.

        Search operations typically need more time than regular chat, so
        a longer default timeout (90s) is used when the caller doesn't
        specify one.  A higher default max_tokens (2000) is used because
        search results are typically longer than planner JSON output.
        """
        if not self._client:
            return LLMResponse(
                content="",
                token_usage={},
                error="LLM API key is not configured; fallback logic was used.",
            )
        search_timeout = timeout or max(self.timeout or 0, 90)
        search_max_tokens = max_tokens or max(self.max_tokens, 2000)
        try:
            search_extra_body = dict(self.extra_body)
            search_extra_body["enable_search"] = True
            if search_options:
                search_extra_body["search_options"] = {
                    **dict(search_extra_body.get("search_options") or {}),
                    **dict(search_options),
                }
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature if temperature is None else temperature,
                "max_tokens": search_max_tokens,
                "timeout": search_timeout,
            }
            if search_extra_body:
                payload["extra_body"] = search_extra_body
            response = self._client.chat.completions.create(**payload)
            usage = getattr(response, "usage", None)
            token_usage = usage.model_dump() if hasattr(usage, "model_dump") else {}
            self._accumulate(token_usage)
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
        except json.JSONDecodeError:
            repaired = _repair_truncated_json(text)
            if repaired is not None:
                return repaired, None
            return {}, f"failed to parse JSON from judge response: raw={response.content[:200]}"


def _repair_truncated_json(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to repair truncated/malformed JSON by closing open brackets."""
    start = text.find("{")
    if start < 0:
        return None
    snippet = text[start:]
    open_brackets = 0
    open_braces = 0
    in_string = False
    escape_next = False
    for ch in snippet:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            open_braces += 1
        elif ch == "}":
            open_braces -= 1
        elif ch == "[":
            open_brackets += 1
        elif ch == "]":
            open_brackets -= 1

    # Try progressively: add closing chars, then remove trailing garbage
    suffix = "]" * open_brackets + "}" * open_braces
    candidates = [snippet + suffix]
    # Also try removing incomplete trailing value (e.g. "key": [partial...)
    for cut in range(len(snippet) - 1, max(len(snippet) - 100, start), -1):
        if snippet[cut] in (",", "\n"):
            trimmed = snippet[:cut].rstrip().rstrip(",")
            if trimmed.endswith(":"):
                trimmed = trimmed[:trimmed.rfind(",")].rstrip().rstrip(",")
            candidate = trimmed + suffix
            candidates.append(candidate)
            if len(candidates) > 20:
                break

    for candidate in candidates:
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue
    return None
