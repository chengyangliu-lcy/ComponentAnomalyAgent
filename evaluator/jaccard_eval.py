from __future__ import annotations

import re
import unicodedata


KEEP_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9.%+\-_/ΩμuUFVAHhmMkKnNpP℃°]+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("μ", "u").replace("Ω", "ohm")
    parts = KEEP_RE.findall(text)
    return "".join(parts).lower()


def char_bigrams(text: str) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) == 1:
        return {normalized}
    return {normalized[idx : idx + 2] for idx in range(len(normalized) - 1)}


def bigram_jaccard(reference: str, prediction: str) -> float:
    ref = char_bigrams(reference)
    pred = char_bigrams(prediction)
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    return len(ref & pred) / len(ref | pred)

