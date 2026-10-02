"""Email spam detection using TF-IDF + Multinomial Naive Bayes.

The vectorizer and classifier are trained offline by
``training/train_spam.py`` and loaded from ``models/spam_vectorizer.pkl`` and
``models/spam_model.pkl``. The raw dataset is never loaded at runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from config import (
    EMAIL_MAX_CHARACTERS,
    SPAM_LABEL_HAM,
    SPAM_LABEL_SPAM,
    SPAM_LOW_CONFIDENCE_BELOW,
    SPAM_MIN_CHARACTERS,
    SPAM_MODEL,
    SPAM_RISK_THRESHOLDS,
    SPAM_VECTORIZER,
)
from utils.helpers import (
    clean_email_text,
    classify_risk,
    confidence_label,
    extract_urls_from_text,
    spam_indicators,
    spam_recommendation,
)

logger = logging.getLogger(__name__)


class SpamModelNotAvailableError(RuntimeError):
    """Raised when the trained spam artifacts are missing or corrupt."""


@dataclass
class SpamResult:
    """Outcome of a single email analysis."""

    text_preview: str
    prediction: str
    confidence: float
    spam_probability: float
    risk_level: str
    indicators: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    recommendation: str = ""
    confidence_note: str = ""
    model_available: bool = True
    error: str | None = None

    @property
    def is_spam(self) -> bool:
        return self.prediction == "SPAM"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_preview": self.text_preview,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "spam_probability": round(self.spam_probability, 4),
            "risk_level": self.risk_level,
            "indicators": self.indicators,
            "urls": self.urls,
            "recommendation": self.recommendation,
            "confidence_note": self.confidence_note,
            "model_available": self.model_available,
            "error": self.error,
        }


class SpamDetector:
    """Loads the vectorizer + Naive Bayes pair once and scores email text."""

    def __init__(
        self, model_path: Path | None = None, vectorizer_path: Path | None = None
    ) -> None:
        self.model_path = Path(model_path or SPAM_MODEL)
        self.vectorizer_path = Path(vectorizer_path or SPAM_VECTORIZER)
        self.model: Any | None = None
        self.vectorizer: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None

    def is_available(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def load(self) -> bool:
        """Load both artifacts. Returns False instead of raising."""
        if self.is_available():
            return True
        missing = [
            path.name
            for path in (self.model_path, self.vectorizer_path)
            if not path.exists()
        ]
        if missing:
            self.load_error = (
                f"Spam model artifact(s) not found: {', '.join(missing)}. "
                "Run: python training/train_spam.py"
            )
            logger.warning(self.load_error)
            return False
        try:
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.model = joblib.load(self.model_path)
        except Exception as exc:  # noqa: BLE001 - corrupt artifact
            self.load_error = f"Spam model could not be loaded: {exc}"
            logger.warning(self.load_error)
            self.model = None
            self.vectorizer = None
            return False

        if isinstance(self.model, dict) and "model" in self.model:
            self.metadata = dict(self.model.get("metadata", {}))
            self.model = self.model["model"]
        self.load_error = None
        logger.info("Spam model loaded.")
        return True

    def predict(self, text: str) -> SpamResult:
        """Classify an email body without ever contacting an external service."""
        raw = text or ""
        cleaned = clean_email_text(raw)

        if not cleaned:
            return SpamResult(
                text_preview="", prediction="UNKNOWN", confidence=0.0,
                spam_probability=0.0, risk_level="LOW",
                recommendation="Paste an email or message to analyse.",
                error="Empty input.",
            )
        if len(cleaned) < SPAM_MIN_CHARACTERS:
            return SpamResult(
                text_preview=cleaned[:120], prediction="UNKNOWN", confidence=0.0,
                spam_probability=0.0, risk_level="LOW",
                recommendation="The message is too short to analyse reliably.",
                error=f"Only {len(cleaned)} usable characters.",
            )
        if len(cleaned) > EMAIL_MAX_CHARACTERS:
            cleaned = cleaned[:EMAIL_MAX_CHARACTERS]

        indicators = spam_indicators(cleaned)
        urls = extract_urls_from_text(cleaned)

        if not self.load():
            return SpamResult(
                text_preview=cleaned[:200], prediction="UNKNOWN", confidence=0.0,
                spam_probability=0.0, risk_level="UNKNOWN",
                indicators=indicators, urls=urls,
                recommendation=(
                    "The spam model is unavailable, so only the structural "
                    "indicators above could be produced."
                ),
                model_available=False,
                error=self.load_error,
            )

        try:
            vectorized = self.vectorizer.transform([cleaned])
            probabilities = self.model.predict_proba(vectorized)[0]
        except Exception as exc:  # noqa: BLE001 - unexpected model failure
            logger.exception("Spam prediction failed")
            return SpamResult(
                text_preview=cleaned[:200], prediction="UNKNOWN", confidence=0.0,
                spam_probability=0.0, risk_level="UNKNOWN",
                indicators=indicators, urls=urls,
                recommendation="The model raised an error while scoring this message.",
                error=f"Model prediction error: {exc}",
            )

        classes = [str(label) for label in self.model.classes_]
        if SPAM_LABEL_SPAM in classes:
            spam_idx = classes.index(SPAM_LABEL_SPAM)
        else:  # pragma: no cover - defensive
            spam_idx = len(classes) - 1
        spam_probability = float(probabilities[spam_idx])
        predicted = classes[int(max(range(len(probabilities)), key=probabilities.__getitem__))]

        prediction = "SPAM" if predicted == SPAM_LABEL_SPAM else "HAM"
        risk = classify_risk(spam_probability, SPAM_RISK_THRESHOLDS, "LOW")

        return SpamResult(
            text_preview=cleaned[:200],
            prediction=prediction,
            confidence=spam_probability if prediction == "SPAM"
            else 1.0 - spam_probability,
            spam_probability=spam_probability,
            risk_level=risk,
            indicators=indicators,
            urls=urls,
            recommendation=spam_recommendation(prediction, risk),
            confidence_note=confidence_label(
                spam_probability, SPAM_LOW_CONFIDENCE_BELOW
            ),
        )

    def summary(self) -> dict[str, Any]:
        """Human-readable status block for the UI and tests."""
        return {
            "available": self.is_available(),
            "model_path": self.model_path.name,
            "vectorizer_path": self.vectorizer_path.name,
            "error": self.load_error,
            "trained_at": self.metadata.get("trained_at"),
            "metrics": self.metadata.get("metrics"),
        }


__all__ = [
    "SpamDetector",
    "SpamResult",
    "SpamModelNotAvailableError",
    "SPAM_LABEL_HAM",
]