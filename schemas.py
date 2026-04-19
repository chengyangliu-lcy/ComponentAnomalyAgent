from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ImageRef:
    original_url: str
    path: Optional[Path]
    exists: bool

    def to_json(self) -> Dict[str, Any]:
        return {
            "original_url": self.original_url,
            "path": str(self.path) if self.path else None,
            "exists": self.exists,
        }


@dataclass
class StandardSample:
    sample_id: str
    post_id: str
    question_text: str
    images: List[ImageRef]
    reference_answer: str
    raw_messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, include_reference: bool = True) -> Dict[str, Any]:
        data = {
            "sample_id": self.sample_id,
            "post_id": self.post_id,
            "question_text": self.question_text,
            "images": [image.to_json() for image in self.images],
            "raw_messages": self.raw_messages,
            "metadata": self.metadata,
        }
        if include_reference:
            data["reference_answer"] = self.reference_answer
        return data


@dataclass
class ToolEvent:
    tool_name: str
    action: str
    success: bool
    elapsed_seconds: float = 0.0
    summary: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Evidence:
    source: str
    title: str
    content: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentPlan:
    question_type: str
    needs_images: bool
    needs_local_retrieval: bool
    needs_web_search: bool
    queries: List[str]
    steps: List[str]
    strategy: str = ""
    selected_actions: List[Dict[str, Any]] = field(default_factory=list)
    final_stop_reason: str = ""
    budgets: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceResult:
    sample_id: str
    question: str
    answer: str
    tools_used: List[str]
    web_searched: bool
    tool_trace: List[ToolEvent]
    reasoning_summary: str
    elapsed_seconds: float
    token_usage: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    plan: Optional[AgentPlan] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "answer": self.answer,
            "tools_used": self.tools_used,
            "web_searched": self.web_searched,
            "tool_trace": [event.to_json() for event in self.tool_trace],
            "reasoning_summary": self.reasoning_summary,
            "elapsed_seconds": self.elapsed_seconds,
            "token_usage": self.token_usage,
            "errors": self.errors,
            "plan": self.plan.to_json() if self.plan else None,
        }
