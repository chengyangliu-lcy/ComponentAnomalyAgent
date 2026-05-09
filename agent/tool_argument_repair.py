from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from schemas import Evidence
from tools.circuit_kb import expand_chinese_electronics_terms


GENERIC_QUERY_TERMS = {
    "原因",
    "处理",
    "问题",
    "异常",
    "分析",
    "电路",
    "电子",
    "怎么",
    "如何",
    "troubleshooting",
    "reason",
    "reasons",
    "issue",
    "issues",
    "problem",
    "problems",
    "fix",
    "repair",
    "solution",
}

STOP_TERMS = {
    "请教",
    "这个",
    "哪些",
    "以及",
    "为什么",
    "怎么处理",
    "问题",
    "电路",
}


@dataclass
class RepairNotes:
    applied: bool = False
    reasons: list[str] = field(default_factory=list)
    query_rewrite_before: str | None = None
    query_rewrite_after: str | None = None

    def add(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)
            self.applied = True

    def to_json(self) -> dict[str, Any]:
        return {
            "repair_applied": self.applied,
            "repair_reasons": self.reasons,
            "query_rewrite_before": self.query_rewrite_before,
            "query_rewrite_after": self.query_rewrite_after,
        }


def repair_action_args(
    tool_name: str,
    args: dict[str, Any],
    *,
    question: str,
    evidence: list[Evidence],
    max_web_results: int,
    rank_limit: int,
    next_seed_query: Callable[[], str],
    select_read_target: Callable[[str], Evidence | None],
    allow_llm: bool,
    query_translator: Callable[[str], str | None] | None = None,
) -> tuple[dict[str, Any], RepairNotes]:
    repaired = dict(args)
    repaired.pop("_meta", None)
    notes = RepairNotes()

    if tool_name == "web_search":
        repaired, notes = _repair_web_search(repaired, notes, question, max_web_results, next_seed_query)
    elif tool_name == "local_retrieve":
        repaired, notes = _repair_local_retrieve(repaired, notes, question, rank_limit, next_seed_query, query_translator=query_translator)
    elif tool_name == "web_read":
        repaired, notes = _repair_web_read(repaired, notes, select_read_target)
    elif tool_name == "rank_evidence":
        repaired, notes = _repair_rank_evidence(repaired, notes, rank_limit)
    elif tool_name == "finish_answer":
        repaired, notes = _repair_finish_answer(repaired, notes, allow_llm)
    elif tool_name == "qwen_search":
        repaired, notes = _repair_qwen_search(repaired, notes, question, next_seed_query)
    elif tool_name in {"inspect_image", "review_evidence", "match_domain_skill"}:
        repaired, notes = _normalize_simple_args(tool_name, repaired, notes)

    repaired["_meta"] = notes.to_json()
    return repaired, notes


def _normalize_simple_args(tool_name: str, args: dict[str, Any], notes: RepairNotes) -> tuple[dict[str, Any], RepairNotes]:
    allowed_keys = {
        "inspect_image": set(),
        "review_evidence": set(),
        "match_domain_skill": {"query", "topic", "details"},
    }[tool_name]
    normalized = {key: value for key, value in args.items() if key in allowed_keys and value is not None}
    if normalized != args:
        notes.add(f"{tool_name} args normalized")
    return normalized, notes


def _repair_local_retrieve(
    args: dict[str, Any],
    notes: RepairNotes,
    question: str,
    max_results: int,
    next_seed_query: Callable[[], str],
    query_translator: Callable[[str], str | None] | None = None,
) -> tuple[dict[str, Any], RepairNotes]:
    query = str(args.get("query") or "").strip()
    if not query:
        query = next_seed_query().strip() or _build_question_query(question)
        if query:
            notes.add("local_retrieve query filled from question terms")
    rewritten = _rewrite_local_retrieve_query(query, question, query_translator=query_translator)
    if rewritten != query:
        notes.query_rewrite_before = query or None
        notes.query_rewrite_after = rewritten
        if query_translator is not None and _contains_cjk(query):
            notes.add("local_retrieve query LLM-translated to English")
        else:
            notes.add("local_retrieve query expanded with English circuit terms")
    args["query"] = rewritten
    limit = args.get("limit")
    if not isinstance(limit, int):
        args["limit"] = max_results
        notes.add("local_retrieve limit filled from config")
    elif limit < 1:
        args["limit"] = max_results
        notes.add("local_retrieve limit reset to config default")
    elif limit > max_results:
        args["limit"] = max_results
        notes.add("local_retrieve limit clipped to maximum")
    return args, notes


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text or ""))


def _rewrite_local_retrieve_query(
    query: str,
    question: str,
    query_translator: Callable[[str], str | None] | None = None,
) -> str:
    query = re.sub(r"\s+", " ", query or "").strip()

    # LLM-based Chinese-to-English translation (replaces, not just appends)
    if query_translator is not None and _contains_cjk(query):
        translated = query_translator(query)
        if translated and translated != query:
            query = translated

    base_text = f"{question} {query}".strip()
    expansions = expand_chinese_electronics_terms(base_text)
    if not expansions:
        return query
    items = list(dict.fromkeys([query, *expansions]))
    return " ".join(item for item in items if item).strip()


def _repair_web_search(
    args: dict[str, Any],
    notes: RepairNotes,
    question: str,
    max_web_results: int,
    next_seed_query: Callable[[], str],
) -> tuple[dict[str, Any], RepairNotes]:
    query = str(args.get("query") or "").strip()
    original_query = query or None
    if not query:
        query = next_seed_query().strip()
        if query:
            notes.add("web_search query filled from seed query")
    rewritten = _rewrite_query(query, question)
    if rewritten != (query or ""):
        notes.query_rewrite_before = original_query or query or None
        notes.query_rewrite_after = rewritten
        notes.add("web_search query rewritten with question key terms")
    query = rewritten.strip()
    if not query:
        fallback = _build_question_query(question)
        if fallback:
            query = fallback
            notes.query_rewrite_before = original_query
            notes.query_rewrite_after = fallback
            notes.add("web_search query generated from question terms")
    args["query"] = query

    limit = args.get("limit")
    if not isinstance(limit, int):
        args["limit"] = max_web_results
        notes.add("web_search limit filled from config")
    elif limit < 1:
        args["limit"] = max_web_results
        notes.add("web_search limit reset to config default")
    elif limit > max_web_results:
        args["limit"] = max_web_results
        notes.add("web_search limit clipped to config maximum")
    return args, notes


def _repair_web_read(
    args: dict[str, Any],
    notes: RepairNotes,
    select_read_target: Callable[[str], Evidence | None],
) -> tuple[dict[str, Any], RepairNotes]:
    url = str(args.get("url") or "").strip()
    selected = select_read_target(url)
    if not url and selected is not None:
        args["url"] = selected.source
        args["title"] = str(args.get("title") or selected.title or selected.source)
        args["snippet"] = str(args.get("snippet") or selected.content[:500])
        notes.add("web_read url filled from strongest unread evidence")
    elif url and selected is not None:
        if not str(args.get("title") or "").strip() and selected.title:
            args["title"] = selected.title
            notes.add("web_read title filled from selected evidence")
        if not str(args.get("snippet") or "").strip() and selected.content:
            args["snippet"] = selected.content[:500]
            notes.add("web_read snippet filled from selected evidence")
    return args, notes


def _repair_qwen_search(
    args: dict[str, Any],
    notes: RepairNotes,
    question: str,
    next_seed_query: Callable[[], str],
) -> tuple[dict[str, Any], RepairNotes]:
    query = str(args.get("query") or "").strip()
    original_query = query or None
    if not query:
        query = next_seed_query().strip()
        if query:
            notes.add("qwen_search query filled from seed query")
    rewritten = _rewrite_query(query, question)
    if rewritten != (query or ""):
        notes.query_rewrite_before = original_query or query or None
        notes.query_rewrite_after = rewritten
        notes.add("qwen_search query rewritten with question key terms")
    query = rewritten.strip()
    if not query:
        fallback = _build_question_query(question)
        if fallback:
            query = fallback
            notes.query_rewrite_before = original_query
            notes.query_rewrite_after = fallback
            notes.add("qwen_search query generated from question terms")
    args["query"] = query
    return args, notes


def _repair_rank_evidence(args: dict[str, Any], notes: RepairNotes, rank_limit: int) -> tuple[dict[str, Any], RepairNotes]:
    max_items = args.get("max_items")
    if not isinstance(max_items, int):
        args["max_items"] = rank_limit
        notes.add("rank_evidence max_items filled from config")
    elif max_items < 1:
        args["max_items"] = rank_limit
        notes.add("rank_evidence max_items reset to default")
    elif max_items > rank_limit:
        args["max_items"] = rank_limit
        notes.add("rank_evidence max_items clipped to maximum")
    return args, notes


def _repair_finish_answer(args: dict[str, Any], notes: RepairNotes, allow_llm: bool) -> tuple[dict[str, Any], RepairNotes]:
    answer = str(args.get("answer") or "").strip()
    if not answer and "answer" in args:
        args.pop("answer", None)
        notes.add("finish_answer blank planner answer removed")
    if not allow_llm:
        if args.get("allow_llm") is not False:
            args["allow_llm"] = False
            notes.add("finish_answer allow_llm disabled by remaining budget")
    elif "allow_llm" in args and not isinstance(args["allow_llm"], bool):
        args.pop("allow_llm", None)
        notes.add("finish_answer invalid allow_llm removed")
    return args, notes


def _rewrite_query(query: str, question: str) -> str:
    query = re.sub(r"\s+", " ", query or "").strip()
    question_terms = _question_terms(question)
    if not query:
        return _build_question_query(question)
    query_terms = _question_terms(query)
    if not query_terms:
        return _build_question_query(question)
    overlap = {term for term in query_terms if term in question_terms}
    generic_only = all(term.lower() in {item.lower() for item in GENERIC_QUERY_TERMS} for term in query_terms)
    if generic_only or len(overlap) < min(2, len(question_terms)):
        missing_terms = [term for term in question_terms if term not in query_terms][:4]
        enhanced = " ".join(dict.fromkeys([query, *missing_terms]))
        return enhanced.strip()
    return query


def _build_question_query(question: str) -> str:
    terms = _question_terms(question)[:6]
    if not terms:
        return ""
    return " ".join(terms + ["electronics troubleshooting"])


def _question_terms(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{1,8}\d{0,6}[A-Za-z0-9_.+-]*|\d+(?:\.\d+)?\s*[A-Za-z%]*|[\u4e00-\u9fff]{2,}", text or "")
    cleaned: list[str] = []
    for token in tokens:
        normalized = token.strip()
        if not normalized:
            continue
        if normalized in STOP_TERMS:
            continue
        if normalized.lower() in {item.lower() for item in GENERIC_QUERY_TERMS}:
            continue
        cleaned.append(normalized)
    return list(dict.fromkeys(cleaned))
