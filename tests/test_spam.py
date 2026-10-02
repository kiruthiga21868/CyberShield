"""Tests for the email spam detector and text helpers."""
from __future__ import annotations

import pytest

from utils.helpers import (
    clean_email_text,
    extract_urls_from_text,
    spam_indicators,
)


HAM_MESSAGES = [
    "Hey Bob, are we still on for lunch tomorrow? I booked the usual place at 1pm.",
    "Please find attached the meeting notes from yesterday. Let me know if you spot "
    "anything missing before Friday.",
    "Reminder: the project deadline moved to next Wednesday. Nothing else to report.",
]

SPAM_MESSAGES = [
    "WINNER!! Claim your FREE cash prize now. Text CLAIM to 87121 to collect "
    "£1000. T&C apply, 18+ only.",
    "URGENT: your account has been suspended. Verify your account password "
    "immediately at http://secure-verify-account.tk/login or lose access forever.",
    "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to "
    "receive entry question. Std txt rate 08452810075 over 18s.",
]


# ------------------------------------------------------------------ helpers
def test_clean_email_text_strips_reply_chain():
    raw = "Sounds good, thanks!\n\n> previous message\n> older text"
    cleaned = clean_email_text(raw)
    assert "previous message" not in cleaned
    assert "Sounds good" in cleaned


def test_clean_email_text_handles_empty():
    assert clean_email_text("") == ""
    assert clean_email_text(None) == ""


@pytest.mark.parametrize(
    "text,expected",
    [
        ("go to http://evil.tk/login now", ["http://evil.tk/login"]),
        ("visit www.example.com/page", ["www.example.com/page"]),
        ("no links here", []),
        ("", []),
    ],
)
def test_extract_urls(text, expected):
    assert extract_urls_from_text(text) == expected


def test_extract_urls_deduplicates():
    assert extract_urls_from_text("http://a.tk/ http://a.tk/ ") == ["http://a.tk/"]


def test_spam_indicators_detect_prize_language():
    assert any("Prize" in indicator for indicator in spam_indicators(SPAM_MESSAGES[0]))


def test_spam_indicators_empty_text():
    assert spam_indicators("") == []


# --------------------------------------------------------------- inference
@pytest.mark.parametrize("message", SPAM_MESSAGES)
def test_spam_messages_are_detected(spam_detector, message):
    result = spam_detector.predict(message)
    assert result.prediction == "SPAM", f"{message[:40]} -> {result.prediction}"
    assert result.spam_probability > 0.5
    assert result.recommendation


@pytest.mark.parametrize("message", HAM_MESSAGES)
def test_ham_messages_are_detected(spam_detector, message):
    result = spam_detector.predict(message)
    assert result.prediction == "HAM", f"{message[:40]} -> {result.prediction}"
    assert result.indicators is not None


def test_empty_input_is_handled(spam_detector):
    result = spam_detector.predict("")
    assert result.prediction == "UNKNOWN"
    assert result.error == "Empty input."


def test_whitespace_only_input_is_handled(spam_detector):
    assert spam_detector.predict("    \n  ").prediction == "UNKNOWN"


def test_too_short_input_is_handled(spam_detector):
    result = spam_detector.predict("hi")
    assert result.prediction == "UNKNOWN"
    assert "too short" in result.recommendation


def test_urls_are_returned_for_optional_url_scan(spam_detector):
    result = spam_detector.predict(
        "Verify at http://secure-verify-account.tk/login or call 0800 1234567"
    )
    assert "http://secure-verify-account.tk/login" in result.urls


def test_result_is_serialisable(spam_detector):
    payload = spam_detector.predict(HAM_MESSAGES[0]).to_dict()
    assert set(payload) >= {"prediction", "confidence", "risk_level", "indicators"}
    assert 0.0 <= payload["confidence"] <= 1.0


def test_long_input_is_truncated_not_crashed(spam_detector):
    result = spam_detector.predict("spam free prize win " * 20000)
    assert result.prediction in {"SPAM", "HAM"}


# ------------------------------------------------------------ degraded path
def test_missing_model_degrades_gracefully(missing_model_spam_detector):
    assert missing_model_spam_detector.load() is False
    result = missing_model_spam_detector.predict(SPAM_MESSAGES[0])
    assert result.model_available is False
    assert result.error
    # Rule-based indicators are still produced.
    assert result.indicators


def test_corrupt_vectorizer_is_handled(monkeypatch, tmp_path):
    import joblib

    from modules.email_detector import SpamDetector

    (tmp_path / "spam_model.pkl").write_bytes(b"not a pickle")
    (tmp_path / "spam_vectorizer.pkl").write_bytes(b"not a pickle")
    monkeypatch.setattr(joblib, "load", lambda _: (_ for _ in ()).throw(
        OSError("corrupt")
    ))
    detector = SpamDetector(
        model_path=tmp_path / "spam_model.pkl",
        vectorizer_path=tmp_path / "spam_vectorizer.pkl",
    )
    assert detector.load() is False
    assert detector.predict("hello there, how are you?").model_available is False


def test_model_error_does_not_crash(spam_detector):
    class Boom:
        classes_ = ["ham", "spam"]

        def predict_proba(self, _matrix):
            raise RuntimeError("model exploded")

    original_model = spam_detector.model
    original_vectorizer = spam_detector.vectorizer
    try:
        spam_detector.model = Boom()
        spam_detector.vectorizer = object()
        result = spam_detector.predict("some reasonably long message text here")
        assert result.prediction == "UNKNOWN"
        assert result.error
    finally:
        spam_detector.model = original_model
        spam_detector.vectorizer = original_vectorizer