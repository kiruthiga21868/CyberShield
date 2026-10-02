"""Shared pytest fixtures for the CyberShield AI test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import PHISHING_MODEL, SPAM_MODEL, SPAM_VECTORIZER  # noqa: E402


@pytest.fixture(scope="session")
def phishing_detector():
    """A loaded phishing detector, skipped when the model is absent."""
    from modules.phishing_detector import PhishingDetector

    detector = PhishingDetector()
    if not PHISHING_MODEL.exists():
        pytest.skip(f"Phishing model missing. Run: python training/train_phishing.py")
    assert detector.load(), detector.load_error
    return detector


@pytest.fixture(scope="session")
def spam_detector():
    """A loaded spam detector, skipped when artifacts are absent."""
    from modules.email_detector import SpamDetector

    detector = SpamDetector()
    if not (SPAM_MODEL.exists() and SPAM_VECTORIZER.exists()):
        pytest.skip(f"Spam model missing. Run: python training/train_spam.py")
    assert detector.load(), detector.load_error
    return detector


@pytest.fixture(autouse=True)
def _clear_external_service_cache():
    """Reset module-level caches so service tests cannot leak into each other."""
    from services.virustotal_service import clear_cache

    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def missing_model_detector(tmp_path):
    """A detector pointed at a path that does not exist."""
    from modules.phishing_detector import PhishingDetector

    return PhishingDetector(model_path=tmp_path / "nope.pkl")


@pytest.fixture
def missing_model_spam_detector(tmp_path):
    """A spam detector pointed at paths that do not exist."""
    from modules.email_detector import SpamDetector

    return SpamDetector(
        model_path=tmp_path / "nope.pkl",
        vectorizer_path=tmp_path / "nope_vec.pkl",
    )