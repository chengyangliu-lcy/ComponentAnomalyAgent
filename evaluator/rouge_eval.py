from __future__ import annotations

from evaluator.jaccard_eval import normalize_text


def _lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for idx, cb in enumerate(b, 1):
            if ca == cb:
                cur.append(prev[idx - 1] + 1)
            else:
                cur.append(max(prev[idx], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(reference: str, prediction: str) -> float:
    ref = normalize_text(reference)
    pred = normalize_text(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    lcs = _lcs_length(ref, pred)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    beta = 1.2
    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)

