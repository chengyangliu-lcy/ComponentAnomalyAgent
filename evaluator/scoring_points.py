from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from evaluator.jaccard_eval import normalize_text


NUMBER_UNIT_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:v|V|a|A|w|W|ohm|Ω|k|K|mA|uF|nF|pF|Hz|kHz|MHz|℃|%|ms|us|μs|s)?")
COMPONENT_RE = re.compile(r"\b[A-Z]{1,6}\d{0,4}[A-Z0-9_.+-]*\b|[\u4e00-\u9fff]{0,4}(?:电阻|电容|电感|二极管|三极管|MOS管|芯片|光耦|变压器|传感器|控制器|运放|比较器)")


@dataclass
class ScoringPointResult:
    reference_points: List[str]
    hit_points: List[str]
    missed_points: List[str]
    false_positive_points: List[str]
    coverage: float

    def to_json(self) -> Dict[str, object]:
        return {
            "reference_points": self.reference_points,
            "hit_points": self.hit_points,
            "missed_points": self.missed_points,
            "false_positive_points": self.false_positive_points,
            "coverage": self.coverage,
        }


def extract_scoring_points(reference: str, max_points: int = 12) -> List[str]:
    points: list[str] = []
    for pattern in [COMPONENT_RE, NUMBER_UNIT_RE]:
        for match in pattern.findall(reference or ""):
            cleaned = match.strip()
            if len(cleaned) >= 2 and cleaned not in points:
                points.append(cleaned)
    sentence_parts = re.split(r"[。；;\n]", reference or "")
    cue_words = ["原因", "用于", "作用", "防止", "避免", "建议", "需要", "检查", "调整", "滤波", "补偿"]
    for part in sentence_parts:
        text = part.strip()
        if 8 <= len(text) <= 80 and any(cue in text for cue in cue_words):
            if text not in points:
                points.append(text)
    return points[:max_points]


def judge_scoring_points(reference: str, prediction: str) -> ScoringPointResult:
    points = extract_scoring_points(reference)
    normalized_prediction = normalize_text(prediction)
    hit: list[str] = []
    missed: list[str] = []
    for point in points:
        normalized_point = normalize_text(point)
        if normalized_point and normalized_point in normalized_prediction:
            hit.append(point)
        else:
            tokens = set(normalized_point[idx : idx + 2] for idx in range(max(0, len(normalized_point) - 1)))
            if tokens:
                pred_tokens = set(normalized_prediction[idx : idx + 2] for idx in range(max(0, len(normalized_prediction) - 1)))
                overlap = len(tokens & pred_tokens) / max(len(tokens), 1)
                if overlap >= 0.65:
                    hit.append(point)
                    continue
            missed.append(point)
    prediction_points = extract_scoring_points(prediction)
    normalized_refs = {normalize_text(point) for point in points}
    false_positive = [point for point in prediction_points if normalize_text(point) not in normalized_refs][:8]
    coverage = len(hit) / len(points) if points else 1.0
    return ScoringPointResult(points, hit, missed, false_positive, coverage)

