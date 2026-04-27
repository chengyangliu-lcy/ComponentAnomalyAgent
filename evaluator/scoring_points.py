from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

from evaluator.jaccard_eval import extract_technical_tokens, normalize_text, token_weight
from evaluator.rouge_eval import split_claims


STATUS_SCORES = {"hit": 1.0, "partial": 0.5, "missed": 0.0, "contradicted": -1.0}
POINT_TYPES = {
    "core_conclusion",
    "cause_mechanism",
    "component_or_value",
    "diagnostic_step",
    "fix_suggestion",
    "caveat",
}

SENTENCE_SPLIT_RE = re.compile(r"[。；;！？!?\n]+")
RELATION_RE = re.compile(r"(?:因|由|致|使|造成|引起|导致|相关|影响|取决|反馈|补偿|耦合)")
ACTION_RE = re.compile(r"(?:查|测|看|观察|确认|验证|复核|排查|更换|调整|增大|减小|并联|串联|处理|改善|补偿)")
CAUTION_RE = re.compile(r"(?:不|未|无|非|避免|防止|注意|风险|可能|不确定|不能)")
GENERIC_RE = re.compile(r"(?:建议|检查|可能|异常|问题|原因|需要|进一步|相关|处理)")
NEGATION_RE = re.compile(r"(?:不|未|无|非|无需|不需要|不能)")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:v|a|w|ohm|Ω|kΩ|mΩ|ma|ua|μa|uf|μf|nf|pf|hz|khz|mhz|%|ms|us|μs|s)?", re.I)

DIRECTION_GROUPS = [
    {"上管", "下管"},
    {"开路", "短路"},
    {"过大", "过小", "偏大", "偏小", "增大", "减小", "升高", "降低"},
    {"导通", "关断", "截止"},
    {"输入", "输出"},
    {"高电平", "低电平"},
    {"正向", "反向"},
]


@dataclass
class StructuredScoringPoint:
    id: str
    type: str
    text: str
    aliases: List[str]
    weight: float
    required: bool
    evidence_source: str = "reference_answer"
    point_type_confidence: float = 0.0

    def to_json(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "aliases": self.aliases,
            "weight": self.weight,
            "required": self.required,
            "evidence_source": self.evidence_source,
            "point_type_confidence": self.point_type_confidence,
        }


@dataclass
class ScoringPointMatch:
    id: str
    type: str
    text: str
    status: str
    score: float
    weight: float
    required: bool
    overlap: float
    matched_aliases: List[str]
    match_evidence: Dict[str, object] = field(default_factory=dict)
    contradiction_evidence: List[str] = field(default_factory=list)
    point_type_confidence: float = 0.0

    def to_json(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "status": self.status,
            "score": self.score,
            "weight": self.weight,
            "required": self.required,
            "overlap": self.overlap,
            "matched_aliases": self.matched_aliases,
            "match_evidence": self.match_evidence,
            "contradiction_evidence": self.contradiction_evidence,
            "point_type_confidence": self.point_type_confidence,
        }


@dataclass
class ScoringPointResult:
    reference_points: List[Any]
    hit_points: List[str]
    missed_points: List[str]
    false_positive_points: List[str]
    coverage: float | None
    matches: List[ScoringPointMatch] | None = None
    critical_errors: List[str] | None = None
    required_coverage: float | None = None
    core_conclusion_hit: bool = False
    unsupported_key_tokens: List[str] | None = None

    def to_json(self) -> Dict[str, object]:
        reference_payload = [
            point.to_json() if hasattr(point, "to_json") else point for point in self.reference_points
        ]
        matches_payload = [match.to_json() for match in self.matches or []]
        return {
            "reference_points": reference_payload,
            "hit_points": self.hit_points,
            "missed_points": self.missed_points,
            "false_positive_points": self.false_positive_points,
            "coverage": self.coverage,
            "structured_points": reference_payload,
            "matches": matches_payload,
            "critical_errors": self.critical_errors or [],
            "required_coverage": self.required_coverage,
            "core_conclusion_hit": self.core_conclusion_hit,
            "unsupported_key_tokens": self.unsupported_key_tokens or [],
            "match_evidence": {match["id"]: match.get("match_evidence", {}) for match in matches_payload},
            "contradiction_evidence": {
                match["id"]: match.get("contradiction_evidence", [])
                for match in matches_payload
                if match.get("contradiction_evidence")
            },
            "point_type_confidence": {
                point["id"]: point.get("point_type_confidence", 0.0) for point in reference_payload
            },
        }


def extract_scoring_points(reference: str, max_points: int = 12) -> List[str]:
    return [point.text for point in extract_structured_scoring_points(reference, max_points=max_points)]


def extract_structured_scoring_points(reference: str, max_points: int = 12) -> List[StructuredScoringPoint]:
    candidates = _candidate_points(reference)
    points: list[StructuredScoringPoint] = []
    seen: set[str] = set()
    for text in candidates:
        key = normalize_text(text)
        if key in seen:
            continue
        seen.add(key)
        point_type, confidence = _classify_point(text, is_first=not points)
        points.append(_build_structured_point(len(points) + 1, point_type, text, confidence))
        if len(points) >= max_points:
            return points

    covered_text = "".join(point.text for point in points)
    for token in extract_technical_tokens(reference):
        if token in normalize_text(covered_text) or token_weight(token) < 2.0:
            continue
        points.append(
            StructuredScoringPoint(
                id=f"p{len(points) + 1}",
                type="component_or_value",
                text=token,
                aliases=[token],
                weight=0.8,
                required=False,
                point_type_confidence=0.75,
            )
        )
        if len(points) >= max_points:
            break
    return points


def judge_scoring_points(reference: str, prediction: str) -> ScoringPointResult:
    structured_points = extract_structured_scoring_points(reference)
    if not structured_points:
        return ScoringPointResult([], [], [], [], None, matches=[])
    return judge_structured_scoring_points(structured_points, prediction)


def judge_structured_scoring_points(
    points: List[StructuredScoringPoint],
    prediction: str,
) -> ScoringPointResult:
    normalized_prediction = normalize_text(prediction)
    pred_tokens = set(extract_technical_tokens(prediction))
    matches: list[ScoringPointMatch] = []
    weighted_score = 0.0
    total_weight = 0.0
    required_score = 0.0
    required_weight = 0.0
    hit: list[str] = []
    missed: list[str] = []
    critical_errors: list[str] = []

    for point in points:
        status, overlap, aliases, evidence = _match_point(point, normalized_prediction, pred_tokens)
        contradiction_evidence = _contradiction_evidence(point.text, prediction)
        if status != "missed" and contradiction_evidence:
            status = "contradicted"
        score = STATUS_SCORES[status]
        weighted_score += point.weight * score
        total_weight += point.weight
        if point.required:
            required_score += point.weight * max(score, 0.0)
            required_weight += point.weight
        if status in {"hit", "partial"}:
            hit.append(point.text)
        else:
            missed.append(point.text)
        if status == "contradicted" and point.type == "core_conclusion":
            critical_errors.append(f"core_conclusion_contradicted:{point.id}")
        matches.append(
            ScoringPointMatch(
                id=point.id,
                type=point.type,
                text=point.text,
                status=status,
                score=score,
                weight=point.weight,
                required=point.required,
                overlap=overlap,
                matched_aliases=aliases,
                match_evidence=evidence,
                contradiction_evidence=contradiction_evidence,
                point_type_confidence=point.point_type_confidence,
            )
        )

    coverage = max(0.0, min(1.0, weighted_score / total_weight)) if total_weight else None
    required_coverage = required_score / required_weight if required_weight else None
    reference_tokens = set(extract_technical_tokens(" ".join(point.text for point in points)))
    unsupported = [
        token
        for token in sorted(pred_tokens - reference_tokens, key=lambda item: (-token_weight(item), item))
        if token_weight(token) >= 2.0
    ][:12]
    return ScoringPointResult(
        reference_points=points,
        hit_points=hit,
        missed_points=missed,
        false_positive_points=unsupported[:8],
        coverage=round(coverage, 6) if coverage is not None else None,
        matches=matches,
        critical_errors=critical_errors,
        required_coverage=round(required_coverage, 6) if required_coverage is not None else None,
        core_conclusion_hit=any(
            match.type == "core_conclusion" and match.status in {"hit", "partial"} for match in matches
        ),
        unsupported_key_tokens=unsupported,
    )


def _candidate_points(reference: str) -> List[str]:
    candidates: list[str] = []
    first_candidate = True
    for sentence in SENTENCE_SPLIT_RE.split(reference or ""):
        text = sentence.strip(" \t\r\n:：-•*0123456789.、)")
        if not text:
            continue
        claims = split_claims(text, max_claims=4) if len(normalize_text(text)) > 60 else [text]
        for claim in claims:
            cleaned = claim.strip(" \t\r\n:：-•*0123456789.、)")
            score = _information_score(cleaned)
            if (score >= 0.35 or (first_candidate and len(normalize_text(cleaned)) >= 4)) and cleaned not in candidates:
                candidates.append(cleaned)
                first_candidate = False
    return candidates


def _information_score(text: str) -> float:
    normalized = normalize_text(text)
    if len(normalized) < 4:
        return 0.0
    entities = extract_technical_tokens(text)
    entity_strength = min(sum(token_weight(token) for token in entities) / 8.0, 1.0)
    length_strength = min(len(normalized) / 36.0, 1.0)
    relation_strength = 0.25 if RELATION_RE.search(text) else 0.0
    action_strength = 0.18 if ACTION_RE.search(text) else 0.0
    caution_strength = 0.18 if CAUTION_RE.search(text) else 0.0
    generic_penalty = 0.2 if not entities and GENERIC_RE.fullmatch(normalized) else 0.0
    return max(0.0, min(1.0, 0.45 * entity_strength + 0.35 * length_strength + relation_strength + action_strength + caution_strength - generic_penalty))


def _classify_point(sentence: str, is_first: bool) -> tuple[str, float]:
    entities = extract_technical_tokens(sentence)
    normalized = normalize_text(sentence)
    scores = {
        "core_conclusion": 0.38 if is_first else 0.0,
        "cause_mechanism": 0.0,
        "component_or_value": 0.0,
        "diagnostic_step": 0.0,
        "fix_suggestion": 0.0,
        "caveat": 0.0,
    }
    entity_strength = min(sum(token_weight(token) for token in entities) / 8.0, 1.0)
    scores["component_or_value"] += entity_strength * 0.62
    scores["cause_mechanism"] += 0.45 if RELATION_RE.search(sentence) else 0.0
    scores["diagnostic_step"] += 0.32 if ACTION_RE.search(sentence) else 0.0
    scores["fix_suggestion"] += 0.32 if ACTION_RE.search(sentence) and len(entities) > 0 else 0.0
    scores["caveat"] += 0.40 if CAUTION_RE.search(sentence) else 0.0
    scores["core_conclusion"] += 0.28 if entity_strength >= 0.45 and (RELATION_RE.search(sentence) or len(normalized) >= 16) else 0.0
    if len(normalized) <= 16 and entity_strength >= 0.5:
        scores["component_or_value"] += 0.22
    if not entities and len(normalized) >= 16:
        scores["core_conclusion"] += 0.18
    point_type = max(scores, key=scores.get)
    confidence = max(scores[point_type], 0.35)
    if point_type not in POINT_TYPES:
        point_type = "component_or_value"
    return point_type, round(min(confidence, 1.0), 6)


def _build_structured_point(
    index: int,
    point_type: str,
    text: str,
    confidence: float,
) -> StructuredScoringPoint:
    aliases = extract_technical_tokens(text)
    info = _information_score(text)
    weight = {
        "core_conclusion": 1.8,
        "cause_mechanism": 1.45,
        "component_or_value": 1.0,
        "diagnostic_step": 0.95,
        "fix_suggestion": 0.95,
        "caveat": 0.8,
    }.get(point_type, 1.0)
    weight += min(info * 0.35, 0.35)
    return StructuredScoringPoint(
        id=f"p{index}",
        type=point_type if point_type in POINT_TYPES else "component_or_value",
        text=text,
        aliases=aliases,
        weight=round(weight, 6),
        required=point_type in {"core_conclusion", "cause_mechanism"},
        point_type_confidence=confidence,
    )


def _match_point(
    point: StructuredScoringPoint,
    normalized_prediction: str,
    pred_tokens: set[str],
) -> tuple[str, float, List[str], Dict[str, object]]:
    normalized_point = normalize_text(point.text)
    matched_aliases = [
        alias for alias in point.aliases if alias in pred_tokens or normalize_text(alias) in normalized_prediction
    ]
    point_tokens = set(extract_technical_tokens(point.text))
    entity_overlap = len(point_tokens & pred_tokens) / max(len(point_tokens), 1) if point_tokens else 0.0
    char_overlap = _bigram_overlap(normalized_point, normalized_prediction)
    claim_overlap = _content_token_overlap(point.text, normalized_prediction)
    exact = bool(normalized_point and normalized_point in normalized_prediction)
    evidence_score = max(char_overlap, claim_overlap)
    if point_tokens:
        evidence_score = max(evidence_score, 0.65 * entity_overlap + 0.35 * claim_overlap)
    if matched_aliases and point.type == "component_or_value":
        evidence_score = max(evidence_score, 0.5)
    if exact:
        evidence_score = 1.0
    if evidence_score >= 0.74 or (entity_overlap >= 0.85 and claim_overlap >= 0.35):
        status = "hit"
    elif evidence_score >= 0.42 or (matched_aliases and point.type in {"component_or_value", "diagnostic_step", "fix_suggestion"}):
        status = "partial"
    else:
        status = "missed"
    evidence = {
        "exact": exact,
        "entity_overlap": round(entity_overlap, 6),
        "char_overlap": round(char_overlap, 6),
        "claim_overlap": round(claim_overlap, 6),
        "evidence_score": round(evidence_score, 6),
        "point_entity_count": len(point_tokens),
        "prediction_entity_count": len(pred_tokens),
    }
    return status, round(evidence_score, 6), matched_aliases, evidence


def _content_token_overlap(point_text: str, normalized_prediction: str) -> float:
    point_tokens = _content_tokens(point_text)
    if not point_tokens:
        return 0.0
    pred_text = normalized_prediction
    matched = [token for token in point_tokens if normalize_text(token) in pred_text]
    return len(matched) / len(point_tokens)


def _content_tokens(text: str) -> List[str]:
    tokens = extract_technical_tokens(text)
    normalized = normalize_text(text)
    for idx in range(max(0, len(normalized) - 1)):
        token = normalized[idx : idx + 2]
        if token not in tokens and not GENERIC_RE.fullmatch(token):
            tokens.append(token)
    return tokens[:80]


def _bigram_overlap(point: str, prediction: str) -> float:
    tokens = {point[idx : idx + 2] for idx in range(max(0, len(point) - 1))}
    pred_tokens = {prediction[idx : idx + 2] for idx in range(max(0, len(prediction) - 1))}
    return len(tokens & pred_tokens) / max(len(tokens), 1) if tokens else 0.0


def _contradiction_evidence(point_text: str, prediction: str) -> List[str]:
    point = normalize_text(point_text)
    pred = normalize_text(prediction)
    shared_entities = set(extract_technical_tokens(point_text)) & set(extract_technical_tokens(prediction))
    if not shared_entities and _bigram_overlap(point, pred) < 0.28:
        return []
    evidence: list[str] = []
    if _explicit_polarity_conflict(point_text, prediction, shared_entities):
        evidence.append("negation_mismatch")
    for group in DIRECTION_GROUPS:
        point_hits = {term for term in group if term in point}
        pred_hits = {term for term in group if term in pred}
        if point_hits and pred_hits and point_hits.isdisjoint(pred_hits):
            evidence.append(f"exclusive_terms:{'/'.join(sorted(point_hits))}->{ '/'.join(sorted(pred_hits)) }")
    point_numbers = set(NUMBER_RE.findall(point_text))
    pred_numbers = set(NUMBER_RE.findall(prediction))
    if point_numbers and pred_numbers and point_numbers.isdisjoint(pred_numbers) and shared_entities:
        evidence.append("number_value_mismatch")
    return evidence[:4]


def _explicit_polarity_conflict(point_text: str, prediction: str, shared_entities: set[str]) -> bool:
    if not shared_entities or _bigram_overlap(normalize_text(point_text), normalize_text(prediction)) < 0.45:
        return False
    high_value_shared = {token for token in shared_entities if token_weight(token) >= 2.0 or len(token) >= 3}
    if not high_value_shared:
        return False
    point = normalize_text(point_text)
    pred = normalize_text(prediction)
    polarity_pairs = [
        ("无需", "需要"),
        ("不需要", "需要"),
        ("不能", "能"),
        ("无法", "可以"),
        ("不是", "是"),
        ("并非", "是"),
    ]
    for negative, positive in polarity_pairs:
        if negative in point and positive in pred and negative not in pred and _polarity_mentions_shared_entity(point, pred, high_value_shared):
            return True
        if positive in point and negative in pred and negative not in point and _polarity_mentions_shared_entity(point, pred, high_value_shared):
            return True
    return False


def _polarity_mentions_shared_entity(point: str, pred: str, shared_entities: set[str]) -> bool:
    for token in shared_entities:
        normalized = normalize_text(token)
        if normalized and normalized in point and normalized in pred:
            return True
    return False
