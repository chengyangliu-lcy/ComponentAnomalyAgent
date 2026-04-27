from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional


@dataclass
class SemanticResult:
    score: float
    backend: str
    error: Optional[str] = None


class SemanticSimilarity:
    def __init__(
        self,
        backend: str = "auto",
        bertscore_model: str = "bert-base-chinese",
        bertscore_num_layers: int | None = None,
        sentence_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str | None = None,
    ) -> None:
        self.backend = backend
        self.bertscore_model = bertscore_model
        self.bertscore_num_layers = bertscore_num_layers
        self.sentence_model = sentence_model
        self.device = device
        self._bertscore_scorer = None
        self._bertscore_error: str | None = None
        self._sentence_model = None
        self._sentence_error: str | None = None

    def score(self, reference: str, prediction: str) -> SemanticResult:
        if not reference and not prediction:
            return SemanticResult(1.0, "empty_match")
        if not reference or not prediction:
            return SemanticResult(0.0, "empty_mismatch")
        errors: list[str] = []
        if self.backend in {"auto", "bertscore"}:
            result = self._bertscore(reference, prediction)
            if result.error is None:
                return result
            errors.append(result.error)
            if self.backend == "bertscore":
                return result
        if self.backend in {"auto", "sentence-transformers", "sbert"}:
            result = self._sentence_transformers(reference, prediction)
            if result.error is None:
                return result
            errors.append(result.error)
            if self.backend in {"sentence-transformers", "sbert"}:
                return result
        fallback = SequenceMatcher(None, reference, prediction).ratio()
        return SemanticResult(score=fallback, backend="sequence_matcher", error="; ".join(errors) if errors else None)

    def _bertscore(self, reference: str, prediction: str) -> SemanticResult:
        try:
            from bert_score import BERTScorer

            if self._bertscore_error:
                return SemanticResult(0.0, "bertscore", self._bertscore_error)
            if self._bertscore_scorer is None:
                kwargs = {
                    "model_type": self.bertscore_model,
                    "num_layers": self.bertscore_num_layers,
                    "lang": "zh",
                }
                if self.device:
                    kwargs["device"] = self.device
                self._bertscore_scorer = BERTScorer(**kwargs)
            _, _, f1 = self._bertscore_scorer.score([prediction], [reference])
            return SemanticResult(float(f1[0]), "bertscore")
        except Exception as exc:  # noqa: BLE001
            self._bertscore_error = str(exc)
            return SemanticResult(0.0, "bertscore", str(exc))

    def _sentence_transformers(self, reference: str, prediction: str) -> SemanticResult:
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            if self._sentence_error:
                return SemanticResult(0.0, "sentence-transformers", self._sentence_error)
            if self._sentence_model is None:
                kwargs = {"device": self.device} if self.device else {}
                try:
                    self._sentence_model = SentenceTransformer(
                        self.sentence_model,
                        local_files_only=True,
                        **kwargs,
                    )
                except TypeError:
                    self._sentence_model = SentenceTransformer(self.sentence_model, **kwargs)
            embeddings = self._sentence_model.encode([reference, prediction], normalize_embeddings=True)
            score = float(np.dot(embeddings[0], embeddings[1]))
            return SemanticResult(max(0.0, min(1.0, score)), "sentence-transformers")
        except Exception as exc:  # noqa: BLE001
            self._sentence_error = str(exc)
            return SemanticResult(0.0, "sentence-transformers", str(exc))
