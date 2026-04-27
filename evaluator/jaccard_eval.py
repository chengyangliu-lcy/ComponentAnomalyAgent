from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal

try:  # jieba is optional at runtime; regex extraction remains the fallback.
    import jieba.analyse  # type: ignore[import-untyped]
except Exception:  # noqa: BLE001
    jieba = None  # type: ignore[assignment]


EntityType = Literal["structured", "domain_abbrev", "keyphrase"]

KEEP_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9.%+\-_/ΩμuUFVAHhmMkKnNpP℃°]+")
COMPONENT_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:R|C|L|D|Q|U|IC|J|CN|FB|F|T|ZD|VR|NTC|PTC|LED|MOS|TVS)\d+[A-Z0-9_.+-]*(?![A-Za-z0-9])",
    re.I,
)
MODEL_ID_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,}\d{2,}[A-Z0-9_.+-]*(?![A-Za-z0-9])", re.I)
NUMBER_UNIT_RE = re.compile(
    r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:v|a|w|ohm|kohm|mohm|Ω|kΩ|mΩ|ma|ua|μa|uf|μf|nf|pf|hz|khz|mhz|c|℃|°c|%|ms|us|μs|s)(?![A-Za-z0-9])",
    re.I,
)
ENGLISH_TERM_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]{2,}(?:[-_/][A-Za-z0-9]+)*(?![A-Za-z0-9])")
CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")

STOPWORDS = {
    "一个",
    "一般",
    "不是",
    "不能",
    "主要",
    "也会",
    "以及",
    "但是",
    "例如",
    "可能",
    "因此",
    "如果",
    "存在",
    "导致",
    "建议",
    "应该",
    "或者",
    "所以",
    "检查",
    "检测",
    "测量",
    "进行",
    "这个",
    "这里",
    "需要",
    "问题",
    "异常",
    "正常",
    "原因",
    "影响",
    "可以",
    "是否",
    "没有",
    "出现",
    "相关",
    "由于",
    "通过",
    "分析",
    "判断",
    "确认",
    "结论",
    "依据",
    "具体",
    "实际",
    "部分",
    "方式",
    "情况",
    "作用",
    "位置",
    "方法",
    "方案",
    "处理",
    "改善",
    "功能",
    "实现",
    "增加",
    "降低",
    "提高",
    "选择",
    "采用",
    "使用",
    "注意",
    "风险",
    "不确定性",
}
LOW_VALUE_SUFFIXES = ("情况", "问题", "原因", "作用", "现象", "建议", "方法", "位置", "方案", "处理", "部分", "功能")
PHRASE_SPLIT_RE = re.compile(
    r"(?:的|了|是|为|把|被|用|和|与|或|及|在|对|到|使|让|实现|功能|造成|导致|需要|建议|检查|检测|测量|复核|可能)"
)
TOKEN_ALIASES = {
    "mosfet": "mos",
    "场效应管": "mos",
    "mos管": "mos",
    "闸极": "栅极",
    "门极": "栅极",
    "发烫": "过热",
    "发热": "过热",
    "温升异常": "过热",
    "自激": "振荡",
    "啸振": "振荡",
    "肖特基管": "肖特基二极管",
    "欧姆": "ohm",
    "ω": "ohm",
    "Ω": "ohm",
    "μ": "u",
}
DOMAIN_ABBREVS = {
    "mos",
    "esr",
    "esl",
    "emi",
    "pwm",
    "llc",
    "ntc",
    "ptc",
    "led",
    "tvs",
    "gdt",
    "mov",
    "pcb",
    "buck",
    "boost",
    "ldo",
    "adc",
    "dac",
    "mcu",
    "bms",
    "igbt",
    "sic",
    "gan",
    "gnd",
    "vcc",
    "vdd",
    "vin",
    "vout",
    "vf",
    "sw",
    "rc",
    "lc",
    "snubber",
}
TYPE_LIMITS = {"structured": 40, "domain_abbrev": 24, "keyphrase": 48}


@dataclass(frozen=True)
class TechnicalEntity:
    token: str
    type: EntityType
    weight: float


@dataclass
class TechnicalEntityMatchResult:
    score: float
    reference_entities: List[str]
    prediction_entities: List[str]
    support_entities: List[str]
    matched_entities: List[str]
    missed_entities: List[str]
    unsupported_entities: List[str]
    precision: float
    recall: float
    f1: float
    f_beta: float
    unsupported_entity_weight: float
    unsupported_entity_rate: float
    entity_counts: Dict[str, Dict[str, int]]
    matched_by_type: Dict[str, List[str]]
    missed_by_type: Dict[str, List[str]]

    def to_json(self) -> Dict[str, object]:
        return {
            "score": self.score,
            "reference_entities": self.reference_entities,
            "prediction_entities": self.prediction_entities,
            "support_entities": self.support_entities,
            "matched_entities": self.matched_entities,
            "missed_entities": self.missed_entities,
            "unsupported_entities": self.unsupported_entities,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "f_beta": self.f_beta,
            "unsupported_entity_weight": self.unsupported_entity_weight,
            "unsupported_entity_rate": self.unsupported_entity_rate,
            "entity_counts": self.entity_counts,
            "matched_by_type": self.matched_by_type,
            "missed_by_type": self.missed_by_type,
        }


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("μ", "u").replace("Ω", "ohm").replace("ω", "ohm")
    for src, dst in TOKEN_ALIASES.items():
        text = text.replace(src, dst).replace(src.upper(), dst)
    parts = KEEP_RE.findall(text)
    return "".join(parts).lower()


def normalize_token(text: str) -> str:
    token = normalize_text(text).strip()
    return TOKEN_ALIASES.get(token, token)


def build_technical_idf(documents: Iterable[str]) -> Dict[str, float]:
    docs = list(documents)
    if not docs:
        return {}
    doc_freq: dict[str, int] = {}
    for doc in docs:
        for token in set(extract_technical_tokens(doc)):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    total = len(docs)
    return {
        token: round(math.log((total + 1) / (freq + 1)) + 1.0, 6)
        for token, freq in doc_freq.items()
    }


def extract_technical_tokens(text: str, idf: Dict[str, float] | None = None) -> List[str]:
    return [entity.token for entity in _extract_entities(text, idf=idf, support_tokens=None, role="reference")]


def token_weight(token: str, idf: Dict[str, float] | None = None) -> float:
    entity_type = _entity_type(token)
    if entity_type == "structured":
        return _structured_weight(token)
    if entity_type == "domain_abbrev":
        return 2.2
    if _is_low_value_token(token):
        return 0.0
    return max(0.8, min(float((idf or {}).get(normalize_token(token), 1.0)), 2.4))


def technical_entity_match(
    reference: str,
    prediction: str,
    support_text: str = "",
    idf: Dict[str, float] | None = None,
    support_aliases: Iterable[str] | None = None,
    beta: float = 1.5,
) -> TechnicalEntityMatchResult:
    support_blob = "\n".join([support_text or "", reference or "", " ".join(support_aliases or [])])
    support_entities = _extract_entities(support_blob, idf=idf, support_tokens=None, role="support")
    support_tokens = {entity.token for entity in support_entities}

    ref_entities = _extract_entities(reference, idf=idf, support_tokens=support_tokens, role="reference")
    ref_tokens = {entity.token for entity in ref_entities}
    support_for_prediction = support_tokens | ref_tokens
    pred_entities = _extract_entities(
        prediction,
        idf=idf,
        support_tokens=support_for_prediction,
        role="prediction",
    )
    pred_tokens = {entity.token for entity in pred_entities}

    entity_map = _merge_entity_maps(ref_entities, pred_entities, support_entities)
    matched = ref_tokens & pred_tokens
    missed = ref_tokens - pred_tokens

    if not ref_tokens and not pred_tokens:
        return _result(
            score=1.0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            f_beta=1.0,
            ref_entities=ref_entities,
            pred_entities=pred_entities,
            support_entities=support_entities,
            matched=matched,
            missed=missed,
            unsupported=[],
            unsupported_weight=0.0,
            unsupported_rate=0.0,
            entity_map=entity_map,
        )

    overlap_weight = sum(entity_map[token].weight for token in matched)
    ref_weight = sum(entity.weight for entity in ref_entities)
    supported_pred_weight = sum(entity.weight for entity in pred_entities if entity.token in support_for_prediction)
    pred_weight = supported_pred_weight or sum(entity.weight for entity in pred_entities)
    precision = overlap_weight / pred_weight if pred_weight else 0.0
    recall = overlap_weight / ref_weight if ref_weight else 0.0
    f1 = _fbeta(precision, recall, beta=1.0)
    f_beta = _fbeta(precision, recall, beta=beta)
    unsupported = _unsupported_entities(pred_entities, support_for_prediction)
    unsupported_weight = sum(entity.weight for entity in unsupported)
    high_risk_weight = sum(entity.weight for entity in pred_entities if _is_high_risk(entity))
    unsupported_rate = unsupported_weight / high_risk_weight if high_risk_weight else 0.0

    return _result(
        score=f_beta,
        precision=precision,
        recall=recall,
        f1=f1,
        f_beta=f_beta,
        ref_entities=ref_entities,
        pred_entities=pred_entities,
        support_entities=support_entities,
        matched=matched,
        missed=missed,
        unsupported=unsupported,
        unsupported_weight=unsupported_weight,
        unsupported_rate=unsupported_rate,
        entity_map=entity_map,
    )


def _extract_entities(
    text: str,
    idf: Dict[str, float] | None,
    support_tokens: set[str] | None,
    role: Literal["reference", "prediction", "support"],
) -> List[TechnicalEntity]:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("μ", "u").replace("Ω", "ohm").replace("ω", "ohm")
    candidates: dict[str, TechnicalEntity] = {}
    for pattern in [COMPONENT_ID_RE, MODEL_ID_RE, NUMBER_UNIT_RE]:
        for match in pattern.findall(normalized):
            _add_entity(candidates, match, idf)
    for term in _dynamic_terms(normalized):
        _add_entity(candidates, term, idf)

    entities = list(candidates.values())
    if role == "prediction":
        entities = [
            entity
            for entity in entities
            if entity.type != "keyphrase"
            or entity.token in (support_tokens or set())
            or entity.weight >= 1.6
        ]
    return _limit_entities(entities)


def _dynamic_terms(text: str) -> List[str]:
    terms: list[str] = []
    if jieba is not None:
        for term in jieba.analyse.extract_tags(text, topK=40, withWeight=False, allowPOS=("n", "vn", "v", "eng")):  # type: ignore[union-attr]
            _append_term(terms, str(term))
        for term in jieba.analyse.textrank(text, topK=30, withWeight=False, allowPOS=("n", "vn", "v", "eng")):  # type: ignore[union-attr]
            _append_term(terms, str(term))
    for match in ENGLISH_TERM_RE.findall(text):
        if _looks_like_domain_abbrev(match):
            _append_term(terms, match)
    for run in CJK_RUN_RE.findall(text):
        for term in _fallback_cjk_terms(run):
            _append_term(terms, term)
    return terms


def _fallback_cjk_terms(run: str) -> List[str]:
    candidates: list[str] = []
    for part in PHRASE_SPLIT_RE.split(run):
        part = _trim_cjk_phrase(part)
        if 2 <= len(part) <= 12 and part not in candidates and not _is_low_value_token(part):
            candidates.append(part)
    cleaned = _trim_cjk_phrase(run)
    if 2 <= len(cleaned) <= 12 and cleaned not in candidates and not _is_low_value_token(cleaned):
        candidates.append(cleaned)
    return candidates[:6]


def _trim_cjk_phrase(text: str) -> str:
    text = text.strip("，。；：、,.!?！？（）()[]【】 ")
    for suffix in ("的", "了", "中", "上", "下", "时", "后", "前"):
        if text.endswith(suffix) and len(text) > 2:
            text = text[: -len(suffix)]
    return text


def _append_term(terms: list[str], raw: str) -> None:
    token = normalize_token(raw)
    if token and token not in terms and not _is_low_value_token(token):
        terms.append(token)


def _add_entity(entities: dict[str, TechnicalEntity], raw: str, idf: Dict[str, float] | None) -> None:
    token = normalize_token(raw)
    if not token or _is_low_value_token(token):
        return
    if _is_cjk_token(token) and len(token) < 2:
        return
    entity_type = _entity_type(token)
    if entity_type == "keyphrase" and len(token) < 2:
        return
    weight = token_weight(token, idf)
    if weight <= 0:
        return
    current = entities.get(token)
    entity = TechnicalEntity(token=token, type=entity_type, weight=round(weight, 6))
    if current is None or entity.weight > current.weight:
        entities[token] = entity


def _entity_type(token: str) -> EntityType:
    normalized = normalize_token(token)
    if _is_structured_entity(normalized):
        return "structured"
    if _looks_like_domain_abbrev(normalized):
        return "domain_abbrev"
    return "keyphrase"


def _structured_weight(token: str) -> float:
    token = normalize_token(token)
    if COMPONENT_ID_RE.fullmatch(token):
        return 3.5
    if MODEL_ID_RE.fullmatch(token):
        return 3.0
    if NUMBER_UNIT_RE.fullmatch(token):
        return 2.7
    return 2.5


def _limit_entities(entities: List[TechnicalEntity]) -> List[TechnicalEntity]:
    grouped: dict[str, list[TechnicalEntity]] = {"structured": [], "domain_abbrev": [], "keyphrase": []}
    for entity in entities:
        grouped[entity.type].append(entity)
    limited: list[TechnicalEntity] = []
    for entity_type, items in grouped.items():
        ordered = sorted(items, key=lambda item: (-item.weight, item.token))
        limited.extend(ordered[: TYPE_LIMITS[entity_type]])
    return sorted(limited, key=lambda item: (-item.weight, item.type, item.token))


def _merge_entity_maps(*groups: List[TechnicalEntity]) -> Dict[str, TechnicalEntity]:
    merged: dict[str, TechnicalEntity] = {}
    for group in groups:
        for entity in group:
            current = merged.get(entity.token)
            if current is None or entity.weight > current.weight:
                merged[entity.token] = entity
    return merged


def _unsupported_entities(pred_entities: List[TechnicalEntity], support_tokens: set[str]) -> List[TechnicalEntity]:
    unsupported = [
        entity
        for entity in pred_entities
        if entity.token not in support_tokens and _is_high_risk(entity)
    ]
    return sorted(unsupported, key=lambda item: (-item.weight, item.token))[:12]


def _is_high_risk(entity: TechnicalEntity) -> bool:
    if entity.type == "structured":
        return True
    return entity.type == "domain_abbrev" and entity.weight >= 2.2


def _result(
    score: float,
    precision: float,
    recall: float,
    f1: float,
    f_beta: float,
    ref_entities: List[TechnicalEntity],
    pred_entities: List[TechnicalEntity],
    support_entities: List[TechnicalEntity],
    matched: set[str],
    missed: set[str],
    unsupported: List[TechnicalEntity],
    unsupported_weight: float,
    unsupported_rate: float,
    entity_map: Dict[str, TechnicalEntity],
) -> TechnicalEntityMatchResult:
    matched_ordered = _sort_tokens(matched, entity_map)
    missed_ordered = _sort_tokens(missed, entity_map)
    return TechnicalEntityMatchResult(
        score=round(score, 6),
        reference_entities=[entity.token for entity in ref_entities],
        prediction_entities=[entity.token for entity in pred_entities],
        support_entities=[entity.token for entity in support_entities],
        matched_entities=matched_ordered,
        missed_entities=missed_ordered,
        unsupported_entities=[entity.token for entity in unsupported],
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        f_beta=round(f_beta, 6),
        unsupported_entity_weight=round(unsupported_weight, 6),
        unsupported_entity_rate=round(unsupported_rate, 6),
        entity_counts={
            "reference": _counts_by_type(ref_entities),
            "prediction": _counts_by_type(pred_entities),
            "support": _counts_by_type(support_entities),
            "unsupported": _counts_by_type(unsupported),
        },
        matched_by_type=_tokens_by_type(matched_ordered, entity_map),
        missed_by_type=_tokens_by_type(missed_ordered, entity_map),
    )


def _sort_tokens(tokens: set[str], entity_map: Dict[str, TechnicalEntity]) -> List[str]:
    return sorted(tokens, key=lambda item: (-entity_map[item].weight, entity_map[item].type, item))


def _counts_by_type(entities: List[TechnicalEntity]) -> Dict[str, int]:
    return {
        "structured": sum(1 for entity in entities if entity.type == "structured"),
        "domain_abbrev": sum(1 for entity in entities if entity.type == "domain_abbrev"),
        "keyphrase": sum(1 for entity in entities if entity.type == "keyphrase"),
        "total": len(entities),
    }


def _tokens_by_type(tokens: List[str], entity_map: Dict[str, TechnicalEntity]) -> Dict[str, List[str]]:
    grouped: dict[str, list[str]] = {"structured": [], "domain_abbrev": [], "keyphrase": []}
    for token in tokens:
        grouped[entity_map[token].type].append(token)
    return grouped


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)


def _is_structured_entity(token: str) -> bool:
    return bool(
        COMPONENT_ID_RE.fullmatch(token)
        or MODEL_ID_RE.fullmatch(token)
        or NUMBER_UNIT_RE.fullmatch(token)
    )


def _looks_like_domain_abbrev(token: str) -> bool:
    normalized = normalize_token(token)
    if len(normalized) < 2 or normalized in STOPWORDS:
        return False
    if normalized in DOMAIN_ABBREVS:
        return True
    return bool(re.fullmatch(r"[a-z][a-z0-9+\-_/]{1,}", normalized)) and (
        normalized.upper() == token
        or any(ch.isdigit() for ch in normalized)
        or "/" in normalized
        or "-" in normalized
    )


def _is_cjk_token(token: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", token))


def _is_low_value_token(token: str) -> bool:
    normalized = normalize_token(token)
    if not normalized or normalized in STOPWORDS:
        return True
    if len(normalized) <= 1:
        return True
    if any(normalized.endswith(suffix) for suffix in LOW_VALUE_SUFFIXES):
        return True
    if re.fullmatch(r"\d+", normalized):
        return True
    return False
