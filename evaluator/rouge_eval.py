from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from evaluator.jaccard_eval import extract_technical_tokens, normalize_text, token_weight


CLAUSE_SPLIT_RE = re.compile(
    r"[。；;！？!?\n]+|(?:^|\s)(?:[-*•]|\d+[.)、]|[一二三四五六七八九十]+[、.])\s*"
)
SOFT_BOUNDARY_RE = re.compile(r"[，,、]|(?:并且|同时|另外|其中|因此|所以)")
RELATION_RE = re.compile(r"(?:因|由|致|使|若|当|则|需|应|可|会|与|对|把|将|为)")
NEGATION_RE = re.compile(r"(?:不|未|无|非|勿|避免|禁止)")


@dataclass
class ClaimRougeResult:
    score: float
    claims: List[str]
    claim_scores: List[Dict[str, object]]

    def to_json(self) -> Dict[str, object]:
        return {
            "score": self.score,
            "claims": self.claims,
            "claim_scores": self.claim_scores,
            "claim_weights": [
                {"claim": item["claim"], "weight": item["weight"], "features": item.get("features", {})}
                for item in self.claim_scores
            ],
        }


def claim_rouge_l(reference: str, prediction: str, beta: float = 1.5, max_claims: int = 12) -> ClaimRougeResult:
    claims = split_claims(reference, max_claims=max_claims)
    if not claims:
        return ClaimRougeResult(0.0, [], [])
    prediction_claims = split_claims(prediction, max_claims=max_claims * 2) or [prediction]
    prediction_claim_tokens = [_claim_tokens(item) for item in prediction_claims]
    claim_scores: list[dict[str, object]] = []
    weighted_sum = 0.0
    total_weight = 0.0
    for claim in claims:
        ref_tokens = _claim_tokens(claim)
        score = max(
            (_token_rouge_l(ref_tokens, pred_tokens, beta=beta) for pred_tokens in prediction_claim_tokens),
            default=0.0,
        )
        weight, features = _claim_weight(claim)
        weighted_sum += score * weight
        total_weight += weight
        claim_scores.append(
            {
                "claim": claim,
                "score": round(score, 6),
                "weight": round(weight, 6),
                "features": features,
            }
        )
    return ClaimRougeResult(round(weighted_sum / total_weight, 6) if total_weight else 0.0, claims, claim_scores)


def split_claims(reference: str, max_claims: int = 12) -> List[str]:
    claims: list[str] = []
    for part in CLAUSE_SPLIT_RE.split(reference or ""):
        text = _clean_claim(part)
        if not _is_informative_claim(text):
            continue
        for subpart in _split_soft_boundaries(text):
            cleaned = _clean_claim(subpart)
            if _is_informative_claim(cleaned) and cleaned not in claims:
                claims.append(cleaned)
            if len(claims) >= max_claims:
                return claims
    return claims


def _split_soft_boundaries(text: str) -> List[str]:
    if len(normalize_text(text)) <= 38:
        return [text]
    parts = [item for item in SOFT_BOUNDARY_RE.split(text) if item.strip()]
    if len(parts) <= 1:
        return [text]
    merged: list[str] = []
    buffer = ""
    for part in parts:
        candidate = f"{buffer}，{part}" if buffer else part
        if len(normalize_text(candidate)) < 10:
            buffer = candidate
            continue
        merged.append(candidate)
        buffer = ""
    if buffer:
        merged.append(buffer)
    return merged or [text]


def _clean_claim(text: str) -> str:
    return (text or "").strip(" \t\r\n:：-•*0123456789.、)")


def _is_informative_claim(text: str) -> bool:
    normalized = normalize_text(text)
    if len(normalized) < 4:
        return False
    entities = extract_technical_tokens(text)
    if entities:
        return True
    if len(normalized) >= 10 and (RELATION_RE.search(text) or NEGATION_RE.search(text)):
        return True
    return len(normalized) >= 16


def _claim_tokens(text: str) -> List[str]:
    technical = extract_technical_tokens(text)
    normalized = normalize_text(text)
    chars = [normalized[idx : idx + 2] for idx in range(max(0, len(normalized) - 1))]
    if len(normalized) == 1:
        chars = [normalized]
    tokens: list[str] = []
    for token in technical + chars:
        if token and token not in tokens:
            tokens.append(token)
    return tokens


def _token_rouge_l(reference_tokens: List[str], prediction_tokens: List[str], beta: float) -> float:
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    lcs = _sequence_lcs_length(reference_tokens, prediction_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    f_score = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
    coverage = len(set(reference_tokens) & set(prediction_tokens)) / max(len(set(reference_tokens)), 1)
    return max(f_score, coverage)


def _sequence_lcs_length(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for item_a in a:
        cur = [0]
        for idx, item_b in enumerate(b, 1):
            cur.append(prev[idx - 1] + 1 if item_a == item_b else max(prev[idx], cur[-1]))
        prev = cur
    return prev[-1]


def _claim_weight(claim: str) -> tuple[float, Dict[str, object]]:
    normalized = normalize_text(claim)
    entities = extract_technical_tokens(claim)
    structured_weight = sum(token_weight(entity) for entity in entities if token_weight(entity) >= 2.0)
    relation_count = len(RELATION_RE.findall(claim))
    has_negation = bool(NEGATION_RE.search(claim))
    has_number = bool(re.search(r"\d", claim))
    length_bonus = min(len(normalized) / 80.0, 0.35)
    weight = 1.0
    weight += min(structured_weight / 10.0, 0.55)
    weight += min(relation_count * 0.08, 0.25)
    weight += 0.15 if has_negation else 0.0
    weight += 0.12 if has_number else 0.0
    weight += length_bonus
    if not entities and len(normalized) < 12:
        weight *= 0.75
    features = {
        "entity_count": len(entities),
        "structured_weight": round(structured_weight, 6),
        "relation_count": relation_count,
        "has_negation": has_negation,
        "has_number": has_number,
        "normalized_length": len(normalized),
    }
    return max(0.6, min(weight, 2.0)), features
