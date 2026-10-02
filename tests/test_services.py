"""Tests for the optional external services.

All network behaviour is mocked: the suite never makes a real HTTP request,
and it verifies that a missing key, a network failure, an auth rejection and a
rate limit all degrade instead of raising.
"""
from __future__ import annotations

import pytest
import requests

from config import VIRUSTOTAL_UNAVAILABLE_MESSAGE
from services.groq_service import GroqService, offline_answer
from services.virustotal_service import (
    VirusTotalService,
    encode_url_id,
    merge_virustotal,
)

SAMPLE_URL = "http://paypal-secure-login.tk/verify?account=99"


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


def malicious_payload(malicious: int = 12, total: int = 70) -> dict:
    return {
        "data": {
            "attributes": {
                "last_analysis_stats": {
                    "malicious": malicious,
                    "suspicious": 2,
                    "harmless": total - malicious - 3,
                    "undetected": 1,
                },
                "categories": {"Forcepoint": "phishing", "vendor": "search engines"},
            }
        }
    }


# --------------------------------------------------------- encode_url_id
def test_encode_url_id_is_unpadded_base64url():
    encoded = encode_url_id(SAMPLE_URL)
    assert "=" not in encoded
    assert "+" not in encoded and "/" not in encoded


# ------------------------------------------------------- missing API key
def test_service_is_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("VIRUSTOTAL_API_KEY", raising=False)
    service = VirusTotalService(api_key=None)
    assert service.is_available() is False
    result = service.lookup_url(SAMPLE_URL)
    assert result.available is False
    assert result.status == "UNAVAILABLE"
    assert result.message == VIRUSTOTAL_UNAVAILABLE_MESSAGE
    assert result.risk_level == "UNKNOWN"


def test_missing_key_never_raises(monkeypatch):
    monkeypatch.delenv("VIRUSTOTAL_API_KEY", raising=False)
    assert VirusTotalService().lookup_url(SAMPLE_URL).error is not None


# --------------------------------------------------------- happy path
def test_successful_lookup_is_parsed(monkeypatch):
    monkeypatch.setattr(requests.Session, "get",
                        lambda self, *a, **k: FakeResponse(200, malicious_payload()))
    result = VirusTotalService(api_key="test-key").lookup_url(SAMPLE_URL)
    assert result.available is True
    assert result.status == "MALICIOUS"
    assert result.malicious_engines == 12
    assert result.total_engines == 70
    assert result.risk_level == "HIGH"
    assert "12 of 70" in result.message
    assert result.categories


def test_single_flag_is_only_suspicious(monkeypatch):
    monkeypatch.setattr(requests.Session, "get", lambda self, *a, **k:
                        FakeResponse(200, malicious_payload(1, 70)))
    result = VirusTotalService(api_key="k").lookup_url(SAMPLE_URL)
    assert result.status == "SUSPICIOUS"
    assert result.risk_level == "MEDIUM"


def test_zero_flags_but_suspicious_is_low_risk(monkeypatch):
    payload = malicious_payload(0, 70)
    payload["data"]["attributes"]["last_analysis_stats"]["suspicious"] = 5
    monkeypatch.setattr(requests.Session, "get",
                        lambda self, *a, **k: FakeResponse(200, payload))
    result = VirusTotalService(api_key="k").lookup_url(SAMPLE_URL)
    assert result.status == "LOW_RISK"
    assert result.risk_level == "LOW"


def test_clean_url_is_low_risk(monkeypatch):
    monkeypatch.setattr(requests.Session, "get", lambda self, *a, **k:
                        FakeResponse(200, malicious_payload(0, 70)))
    result = VirusTotalService(api_key="k").lookup_url("https://example.com")
    assert result.status in {"CLEAN", "LOW_RISK"}
    assert result.risk_level == "LOW"


# ------------------------------------------------------------- failures
def test_network_failure_is_handled(monkeypatch):
    def explode(self, *a, **k):
        raise requests.ConnectionError("dns failure")

    monkeypatch.setattr(requests.Session, "get", explode)
    result = VirusTotalService(api_key="k").lookup_url(SAMPLE_URL)
    assert result.available is False
    assert "could not be reached" in result.message


def test_timeout_is_handled(monkeypatch):
    def timeout(self, *a, **k):
        raise requests.Timeout()

    monkeypatch.setattr(requests.Session, "get", timeout)
    result = VirusTotalService(api_key="k").lookup_url(SAMPLE_URL)
    assert "timed out" in result.message


def test_rate_limit_is_handled(monkeypatch):
    monkeypatch.setattr(requests.Session, "get",
                        lambda self, *a, **k: FakeResponse(429))
    result = VirusTotalService(api_key="k").lookup_url(SAMPLE_URL)
    assert "rate limit" in result.message.lower()


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_is_handled(monkeypatch, status):
    monkeypatch.setattr(requests.Session, "get",
                        lambda self, *a, **k: FakeResponse(status))
    result = VirusTotalService(api_key="bad").lookup_url(SAMPLE_URL)
    assert "rejected" in result.message


def test_unexpected_status_is_handled(monkeypatch):
    monkeypatch.setattr(requests.Session, "get",
                        lambda self, *a, **k: FakeResponse(503))
    result = VirusTotalService(api_key="k").lookup_url(SAMPLE_URL)
    assert "unexpected status" in result.message


def test_malformed_json_is_handled(monkeypatch):
    class Broken:
        status_code = 200

        def json(self):
            raise ValueError("not json")

    monkeypatch.setattr(requests.Session, "get", lambda self, *a, **k: Broken())
    result = VirusTotalService(api_key="k").lookup_url(SAMPLE_URL)
    assert "unreadable" in result.message


def test_api_key_never_appears_in_results(monkeypatch):
    monkeypatch.setattr(requests.Session, "get", lambda self, *a, **k:
                        FakeResponse(200, malicious_payload()))
    secret = "super-secret-key-abc123"
    result = VirusTotalService(api_key=secret).lookup_url(SAMPLE_URL)
    assert secret not in str(result.to_dict())


def test_invalid_url_is_rejected_before_any_request(monkeypatch):
    called = []
    monkeypatch.setattr(requests.Session, "get",
                        lambda self, *a, **k: called.append(1) or FakeResponse(200))
    result = VirusTotalService(api_key="k").lookup_url("http://")
    assert result.available is False
    assert called == []


# ------------------------------------------------------------- merging
def test_merge_takes_the_more_severe_verdict():
    from services.virustotal_service import VirusTotalResult

    bad = VirusTotalResult(available=True, status="MALICIOUS", message="",
                           malicious_engines=10, total_engines=70)
    assert merge_virustotal("LOW", bad)[0] == "HIGH"


def test_merge_keeps_local_when_more_severe():
    from services.virustotal_service import VirusTotalResult

    clean = VirusTotalResult(available=True, status="CLEAN", message="")
    assert merge_virustotal("HIGH", clean)[0] == "HIGH"


def test_merge_preserves_local_when_unavailable():
    from services.virustotal_service import VirusTotalResult

    down = VirusTotalResult(available=False, status="UNAVAILABLE",
                            message=VIRUSTOTAL_UNAVAILABLE_MESSAGE)
    risk, explanation = merge_virustotal("MEDIUM", down)
    assert risk == "MEDIUM"
    assert VIRUSTOTAL_UNAVAILABLE_MESSAGE in explanation


# --------------------------------------------------------------- chatbot
def test_groq_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    service = GroqService(api_key=None)
    assert service.is_available() is False
    reply = service.ask("What is phishing?")
    assert reply.is_ai is False
    assert "social-engineering" in reply.text


def test_groq_mode_label_reflects_configuration(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert GroqService(api_key=None).mode_label == "Offline Cybersecurity Assistant"
    assert "AI Chatbot" in GroqService(api_key="k").mode_label


def test_groq_info_exposes_no_secret(monkeypatch):
    service = GroqService(api_key="secret-key-value")
    assert "secret-key-value" not in str(service.info())


def test_offline_answer_covers_main_topics():
    for question, keyword in [
        ("what is phishing", "social-engineering"),
        ("how do I make a strong password", "password manager"),
        ("I clicked a bad link, what now", "two-factor"),
        ("is this QR code safe", "quishing"),
        ("how do I report spam", "Reporting"),
    ]:
        assert keyword.lower() in offline_answer(question).lower(), question


def test_offline_answer_handles_unknown_and_empty():
    assert offline_answer("") != ""
    assert offline_answer("zzzz qqqq") != ""


def test_offline_answer_for_link_question_is_actionable():
    answer = offline_answer("is this link http://foo.tk safe?").lower()
    assert "url scanner" in answer


def test_groq_failure_falls_back_to_offline(monkeypatch):
    service = GroqService(api_key="k")

    class Boom:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**_):
                    raise RuntimeError("API failure")

    service._client = Boom()
    reply = service.ask("what is spam?")
    assert reply.is_ai is False
    assert reply.error and "offline" in reply.error
    assert reply.text


def test_groq_empty_response_falls_back(monkeypatch):
    service = GroqService(api_key="k")

    class Empty:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**_):
                    return type("R", (), {"choices": [
                        type("C", (), {"message": type("M", (), {"content": "  "})()})()
                    ]})()

    service._client = Empty()
    assert service.ask("what is phishing?").is_ai is False


def test_groq_success_path(monkeypatch):
    service = GroqService(api_key="k")

    class Ok:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    assert kwargs["messages"][0]["role"] == "system"
                    return type("R", (), {"choices": [
                        type("C", (), {"message": type("M", (), {
                            "content": "Phishing tricks you into revealing data."})()})()
                    ]})()

    service._client = Ok()
    reply = service.ask("what is phishing?")
    assert reply.is_ai is True
    assert "Phishing" in reply.text