from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ToolArgSpec:
    arg_type: str
    required: bool = False


EnabledPredicate = Callable[[dict[str, Any], dict[str, Any]], bool]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    planner_name: str
    event_tool_name: str
    event_action_name: str
    description: str
    args_schema: dict[str, ToolArgSpec] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)
    budget_keys: tuple[str, ...] = ()
    recoverable_by_default: bool = True
    enabled_predicate: EnabledPredicate | None = None

    def is_enabled(self, agent_cfg: dict[str, Any], web_cfg: dict[str, Any]) -> bool:
        if self.enabled_predicate is None:
            return True
        return bool(self.enabled_predicate(agent_cfg, web_cfg))

    def validate_args(self, args: dict[str, Any]) -> str | None:
        if not isinstance(args, dict):
            return "planner action args must be a JSON object"
        for key, spec in self.args_schema.items():
            if spec.required and key not in args:
                return f"missing required arg: {key}"
            value = args.get(key)
            if value is None:
                continue
            if spec.arg_type == "string" and not isinstance(value, str):
                return f"planner action arg '{key}' must be a string"
            if spec.arg_type == "integer" and not isinstance(value, int):
                return f"planner action arg '{key}' must be an integer"
            if spec.arg_type == "boolean" and not isinstance(value, bool):
                return f"planner action arg '{key}' must be a boolean"
        return None


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def get(self, name: str) -> ToolSpec:
        return self._specs[name]

    def names(self) -> list[str]:
        return list(self._specs)

    def enabled_specs(self, agent_cfg: dict[str, Any], web_cfg: dict[str, Any]) -> list[ToolSpec]:
        return [spec for spec in self._specs.values() if spec.is_enabled(agent_cfg, web_cfg)]

    def enabled_names(self, agent_cfg: dict[str, Any], web_cfg: dict[str, Any]) -> list[str]:
        return [spec.name for spec in self.enabled_specs(agent_cfg, web_cfg)]

    def planner_tool_names(self) -> list[str]:
        return [spec.planner_name for spec in self._specs.values()]

    def planner_tool_names_text(self) -> str:
        return "|".join(self.planner_tool_names())

    def planner_descriptions(self) -> list[str]:
        return [f"- {spec.planner_name}：{spec.description}" for spec in self._specs.values()]


def _web_search_enabled(agent_cfg: dict[str, Any], web_cfg: dict[str, Any]) -> bool:
    _ = web_cfg
    return bool(agent_cfg.get("enable_web_search", True))


def _local_retrieval_enabled(agent_cfg: dict[str, Any], web_cfg: dict[str, Any]) -> bool:
    _ = web_cfg
    return bool(agent_cfg.get("enable_local_retrieval", False))


def _domain_skills_enabled(agent_cfg: dict[str, Any], web_cfg: dict[str, Any]) -> bool:
    _ = web_cfg
    return bool(agent_cfg.get("enable_domain_skills", True))


def _image_enabled(agent_cfg: dict[str, Any], web_cfg: dict[str, Any]) -> bool:
    _ = web_cfg
    return bool(agent_cfg.get("use_images", True))


def build_default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolSpec(
                name="inspect_image",
                planner_name="inspect_image",
                event_tool_name="image_inspect",
                event_action_name="multimodal_component_extract",
                description="存在图片且尚未收集图片证据时优先使用；图片检查失败或次数耗尽后不要重复调用。",
                args_schema={},
                defaults={},
                budget_keys=("max_image_inspect_attempts",),
                recoverable_by_default=True,
                enabled_predicate=_image_enabled,
            ),
            ToolSpec(
                name="match_domain_skill",
                planner_name="match_domain_skill",
                event_tool_name="domain_skill",
                event_action_name="match_electronics_skills",
                description="用于通用电子机制、排查先验、反馈环路、补偿、滤波、电源、接地、驱动、采样和器件选型问题。",
                args_schema={
                    "query": ToolArgSpec("string"),
                    "topic": ToolArgSpec("string"),
                    "details": ToolArgSpec("string"),
                },
                defaults={},
                budget_keys=("max_domain_skill_calls",),
                recoverable_by_default=True,
                enabled_predicate=_domain_skills_enabled,
            ),
            ToolSpec(
                name="local_retrieve",
                planner_name="local_retrieve",
                event_tool_name="local_retrieve",
                event_action_name="hybrid_kb_search",
                description="Search the local Hackster/Common Crawl knowledge base for project, circuit, component, code, and public reference evidence before falling back to live web search.",
                args_schema={
                    "query": ToolArgSpec("string"),
                    "limit": ToolArgSpec("integer"),
                },
                defaults={},
                budget_keys=("max_local_chunks",),
                recoverable_by_default=True,
                enabled_predicate=_local_retrieval_enabled,
            ),
            ToolSpec(
                name="web_search",
                planner_name="web_search",
                event_tool_name="web_search",
                event_action_name="api_or_html_search",
                description="仅在缺少公开资料、型号资料、数据手册、拓扑规则、失效机制或高价值通用参考时使用。",
                args_schema={
                    "query": ToolArgSpec("string"),
                    "limit": ToolArgSpec("integer"),
                },
                defaults={},
                budget_keys=("max_web_queries", "max_results_per_query"),
                recoverable_by_default=True,
                enabled_predicate=_web_search_enabled,
            ),
            ToolSpec(
                name="web_read",
                planner_name="web_read",
                event_tool_name="web_reader",
                event_action_name="read_or_keep_snippet",
                description="当搜索结果摘要有价值时，读取一个最有价值且未读过的公开网址；不要重复读取已读或失败的网址。",
                args_schema={
                    "url": ToolArgSpec("string"),
                    "title": ToolArgSpec("string"),
                    "snippet": ToolArgSpec("string"),
                },
                defaults={},
                budget_keys=("max_pages_to_read",),
                recoverable_by_default=True,
                enabled_predicate=_web_search_enabled,
            ),
            ToolSpec(
                name="rank_evidence",
                planner_name="rank_evidence",
                event_tool_name="evidence_rank",
                event_action_name="rank_and_dedupe",
                description="收集到多来源证据后使用，或在证据较杂但准备结束前使用。",
                args_schema={"max_items": ToolArgSpec("integer")},
                defaults={"max_items": 12},
                budget_keys=("max_items",),
                recoverable_by_default=True,
            ),
            ToolSpec(
                name="review_evidence",
                planner_name="review_evidence",
                event_tool_name="circuit_reviewer",
                event_action_name="coverage_check",
                description="需要检查图片、元件、原因或处理建议等证据覆盖度时使用。",
                args_schema={},
                defaults={},
                budget_keys=(),
                recoverable_by_default=True,
            ),
            ToolSpec(
                name="finish_answer",
                planner_name="finish_answer",
                event_tool_name="finish_answer",
                event_action_name="synthesize_final_answer",
                description="仅在证据足够、剩余预算较低，或重复可恢复错误让继续调用工具价值很低时使用。",
                args_schema={
                    "answer": ToolArgSpec("string"),
                    "allow_llm": ToolArgSpec("boolean"),
                },
                defaults={},
                budget_keys=("final_answer_timeout_seconds",),
                recoverable_by_default=False,
            ),
        ]
    )


DEFAULT_TOOL_REGISTRY = build_default_registry()
