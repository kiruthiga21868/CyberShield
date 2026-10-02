"""Phishing URL detection using a pre-trained Random Forest classifier.

The model is trained offline by ``training/train_phishing.py`` and loaded from
``models/phishing_model.pkl``. Nothing is trained at application start-up.

The feature vector is produced by :func:`utils.url_features.extract_url_features`,
the exact function used at training time, which is what allows a URL that was
never seen during training to be scored on the same schema.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config import (
    PHISHING_CLASS_NAMES,
    PHISHING_DEFAULT_RISK,
    PHISHING_LABEL_LEGITIMATE,
    PHISHING_LABEL_PHISHING,
    PHISHING_LOW_CONFIDENCE_BELOW,
    PHISHING_MODEL,
)
from modules.risk_rules import aggregate_risk, evaluate_rules
from utils.helpers import confidence_label, phishing_recommendation
from utils.url_features import (
    ALL_FEATURE_NAMES,
    InvalidURLError,
    describe_indicators,
    extract_url_features,
)

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained phishing model file is missing or corrupt."""


@dataclass
class PhishingResult:
    """Outcome of a single URL analysis.

    ``prediction`` and ``confidence`` reflect the **Random Forest only**, while
    ``risk_level`` reflects the aggregated ML + rule decision. Keeping both
    explicit prevents the UI from presenting a rule-based verdict as if it
    came from the model.
    """

    url: str
    prediction: str
    confidence: float
    phishing_probability: float
    risk_level: str
    indicators: list[str] = field(default_factory=list)
    recommendation: str = ""
    confidence_note: str = ""
    model_available: bool = True
    error: str | None = None
    risk_basis: str = ""
    rule_score: float = 0.0

    @property
    def is_phishing(self) -> bool:
        return self.prediction == "PHISHING"

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "phishing_probability": round(self.phishing_probability, 4),
            "risk_level": self.risk_level,
            "indicators": self.indicators,
            "recommendation": self.recommendation,
            "confidence_note": self.confidence_note,
            "model_available": self.model_available,
            "error": self.error,
            "risk_basis": self.risk_basis,
            "rule_score": round(self.rule_score, 4),
        }


class PhishingDetector:
    """Loads the Random Forest pipeline once and scores URLs on demand.

    Usage::

        detector = PhishingDetector()
        result = detector.predict("http://paypal-secure-login.tk/verify")
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = Path(model_path or PHISHING_MODEL)
        self.pipeline: Any | None = None
        self.feature_names: tuple[str, ...] = ALL_FEATURE_NAMES
        self.metadata: dict[str, Any] = {}
        self.load_error: str | None = None

    # -- loading -----------------------------------------------------------
    def is_available(self) -> bool:
        return self.pipeline is not None

    def load(self) -> bool:
        """Load the trained pipeline. Returns False instead of raising.

        A missing model must degrade the application, never crash it.
        """
        if self.pipeline is not None:
            return True
        if not self.model_path.exists():
            self.load_error = (
                f"Phishing model not found at {self.model_path.name}. "
                "Run: python training/train_phishing.py"
            )
            logger.warning(self.load_error)
            return False
        try:
            artifact = joblib.load(self.model_path)
        except Exception as exc:  # noqa: BLE001 - corrupt artifact
            self.load_error = f"Phishing model could not be loaded: {exc}"
            logger.warning(self.load_error)
            return False

        try:
            self.pipeline = artifact["pipeline"]
            self.feature_names = tuple(artifact["feature_names"])
            self.metadata = dict(artifact.get("metadata", {}))
        except Exception as exc:  # noqa: BLE001 - wrong artifact layout
            self.load_error = f"Phishing model artifact is malformed: {exc}"
            logger.warning(self.load_error)
            return False

        # Guard against silent schema drift between training and inference.
        if self.feature_names != ALL_FEATURE_NAMES:
            self.load_error = (
                "Feature schema mismatch between the saved model and "
                "utils.url_features.ALL_FEATURE_NAMES. Retrain the model."
            )
            logger.error(self.load_error)
            self.pipeline = None
            return False

        self.load_error = None
        logger.info("Phishing model loaded (%d features).", len(self.feature_names))
        return True

    # -- inference ---------------------------------------------------------
    def predict(self, url: str) -> PhishingResult:
        """Score a single URL and return a fully explained result."""
        cleaned = (url or "").strip()
        if not cleaned:
            return PhishingResult(
                url=cleaned, prediction="UNKNOWN", confidence=0.0,
                phishing_probability=0.0, risk_level=PHISHING_DEFAULT_RISK,
                recommendation="Enter a URL to analyse.",
                error="Empty input.",
            )

        try:
            features = extract_url_features(cleaned)
        except InvalidURLError as exc:
            return PhishingResult(
                url=cleaned, prediction="UNKNOWN", confidence=0.0,
                phishing_probability=0.0, risk_level=PHISHING_DEFAULT_RISK,
                recommendation="This does not look like a valid URL.",
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return PhishingResult(
                url=cleaned, prediction="UNKNOWN", confidence=0.0,
                phishing_probability=0.0, risk_level=PHISHING_DEFAULT_RISK,
                recommendation="The URL could not be analysed.",
                error=f"Unexpected error while extracting URL features: {exc}",
            )

        # Rule evaluation works with or without the model, so a degraded
        # deployment still produces a useful, explainable assessment.
        assessment = evaluate_rules(cleaned, features)

        if not self.load():
            aggregated = aggregate_risk(0.0, assessment)
            return PhishingResult(
                url=cleaned, prediction="UNKNOWN", confidence=0.0,
                phishing_probability=0.0,
                risk_level=aggregated.risk_level,
                indicators=aggregated.indicators,
                recommendation=(
                    "The local phishing model is unavailable, so this result comes "
                    "from the structural rule layer only. Treat the URL as "
                    "unverified."
                ),
                model_available=False,
                error=self.load_error,
                risk_basis=aggregated.basis,
                rule_score=aggregated.rule_score,
            )

        try:
            frame = pd.DataFrame([features], columns=list(self.feature_names))
            probabilities = self.pipeline.predict_proba(frame)[0]
        except Exception as exc:  # noqa: BLE001 - unexpected model failure
            logger.exception("Phishing prediction failed")
            aggregated = aggregate_risk(0.0, assessment)
            return PhishingResult(
                url=cleaned, prediction="UNKNOWN", confidence=0.0,
                phishing_probability=0.0,
                risk_level=aggregated.risk_level,
                indicators=aggregated.indicators,
                recommendation=(
                    "The model raised an error while scoring this URL; only the "
                    "structural rule layer was applied."
                ),
                error=f"Model prediction error: {exc}",
                risk_basis=aggregated.basis,
                rule_score=aggregated.rule_score,
            )

        classes = list(self.pipeline.classes_)
        if PHISHING_LABEL_PHISHING in classes:
            phishing_idx = classes.index(PHISHING_LABEL_PHISHING)
        else:  # pragma: no cover - defensive
            phishing_idx = len(classes) - 1
        phishing_probability = float(probabilities[phishing_idx])
        predicted_label = classes[int(np.argmax(probabilities))]

        prediction = (
            "PHISHING"
            if predicted_label == PHISHING_LABEL_PHISHING
            else "LEGITIMATE"
        )

        # Rule layer and ML probability are combined explicitly so the final
        # risk level is explainable and cannot be a bare model assertion.
        assessment = evaluate_rules(cleaned, features)
        aggregated = aggregate_risk(phishing_probability, assessment)

        return PhishingResult(
            url=cleaned,
            prediction=prediction,
            confidence=phishing_probability if prediction == "PHISHING"
            else 1.0 - phishing_probability,
            phishing_probability=phishing_probability,
            risk_level=aggregated.risk_level,
            indicators=aggregated.indicators or describe_indicators(cleaned, features),
            recommendation=phishing_recommendation(aggregated.risk_level),
            confidence_note=confidence_label(
                phishing_probability, PHISHING_LOW_CONFIDENCE_BELOW
            ),
            risk_basis=aggregated.basis,
            rule_score=aggregated.rule_score,
        )

    def predict_many(self, urls: list[str]) -> list[PhishingResult]:
        """Score several URLs, isolating failures to individual entries."""
        return [self.predict(url) for url in urls]

    def describe_label(self, label: int) -> str:
        return PHISHING_CLASS_NAMES.get(int(label), "Unknown")

    def summary(self) -> dict[str, Any]:
        """Human-readable status block for the UI and tests."""
        return {
            "available": self.is_available(),
            "model_path": str(self.model_path),
            "feature_count": len(self.feature_names),
            "error": self.load_error,
            "trained_at": self.metadata.get("trained_at"),
            "metrics": self.metadata.get("metrics"),
        }