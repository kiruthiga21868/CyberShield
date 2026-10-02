"""Tests for the phishing URL detector and the rule-based risk layer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modules.risk_rules import aggregate_risk, evaluate_rules
from utils.url_features import (
    ALL_FEATURE_NAMES,
    InvalidURLError,
    extract_url_features,
    normalize_url,
    shannon_entropy,
)


# ---------------------------------------------------------------- extractor
def test_feature_schema_is_stable():
    features = extract_url_features("https://example.com/path?a=1")
    assert list(features.keys()) == list(ALL_FEATURE_NAMES)
    assert len(features) == 22


def test_extract_is_deterministic():
    first = extract_url_features("https://example.com/a/b?c=1")
    second = extract_url_features("https://example.com/a/b?c=1")
    assert first == second


def test_tld_is_resolved_for_bare_domains():
    """The original dataset extractor returned NaN here; the repair must not."""
    features = extract_url_features("google.com")
    assert features["tld"] == "com"


def test_https_detection():
    assert extract_url_features("https://example.com")["has_https"] == 1
    assert extract_url_features("http://example.com")["has_https"] == 0


def test_ip_literal_detection_ignores_port():
    """A phishing URL on a non-standard port is still an IP host."""
    features = extract_url_features("http://185.1.2.3:44512/bin.sh")
    assert features["host_is_ip"] == 1
    assert features["nonstandard_port"] == 1
    assert features["tld"] == "<none>"


def test_port_80_and_443_are_standard():
    assert extract_url_features("http://1.2.3.4:80/x")["nonstandard_port"] == 0
    assert extract_url_features("https://1.2.3.4:443/x")["nonstandard_port"] == 0


def test_domain_is_not_an_ip():
    assert extract_url_features("https://1password.com")["host_is_ip"] == 0


def test_path_segments_and_params():
    features = extract_url_features("https://example.com/a/b/c?x=1&y=2&z=3")
    assert features["num_subdirs"] == 3
    assert features["num_params"] == 3


def test_percent_encoding_is_decoded_for_keyword_analysis():
    features = extract_url_features("https://example.com/%6Cogin%2Fverify")
    assert features["suspicious_words"] >= 2


def test_normalize_adds_default_scheme():
    assert normalize_url("example.com").startswith("https://example.com")


@pytest.mark.parametrize(
    "bad",
    ["", "   ", "http://", "://///", "javascript:alert(1)", "not a url",
     "https://exa mple.com", "a" * 5000],
)
def test_invalid_urls_raise(bad):
    with pytest.raises(InvalidURLError):
        extract_url_features(bad)


def test_entropy_of_constant_string_is_zero():
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("") == 0.0


def test_entropy_increases_with_variety():
    assert shannon_entropy("abcdefgh") > shannon_entropy("aaaaaaa")


# ------------------------------------------------------------- rule layer
def test_rule_layer_flags_clear_phishing():
    assessment = evaluate_rules("http://paypal-secure-login.tk/verify?account=99")
    codes = {hit.code for hit in assessment.hits}
    assert "abused_tld" in codes
    # Three or more keywords escalate to the HIGH-severity rule.
    assert "keyword_stuffing" in codes
    assert "no_https" in codes
    assert assessment.rule_score >= 0.55


def test_rule_layer_reports_single_keyword_as_medium():
    assessment = evaluate_rules("https://example.com/account/verify")
    hit = next(h for h in assessment.hits if h.code == "suspicious_words")
    assert hit.severity == "MEDIUM"


def test_rule_layer_is_quiet_for_ordinary_sites():
    assessment = evaluate_rules("https://www.microsoft.com/en-us")
    high = [hit for hit in assessment.hits if hit.severity == "HIGH"]
    assert high == []


def test_rule_layer_detects_executable_payload():
    assessment = evaluate_rules("https://cdn.example.com/update/Setup.exe")
    assert "executable_payload" in {hit.code for hit in assessment.hits}


def test_rule_layer_detects_punycode():
    assert "punycode" in {
        hit.code for hit in evaluate_rules("https://xn--pypal-4ve.com/login").hits
    }


def test_private_ip_is_not_high_severity():
    assessment = evaluate_rules("http://192.168.0.1/admin")
    hit = next(h for h in assessment.hits if h.code == "ip_host")
    assert hit.severity == "MEDIUM"


def test_public_ip_is_high_severity():
    assessment = evaluate_rules("http://185.1.2.3/x")
    hit = next(h for h in assessment.hits if h.code == "ip_host")
    assert hit.severity == "HIGH"


def test_rule_layer_handles_invalid_url():
    assessment = evaluate_rules("http://")
    assert assessment.hits and assessment.hits[0].code == "invalid_url"


# ------------------------------------------------------------ aggregation
def test_ml_alone_is_capped_at_medium():
    """This is the guard against the training-data bias."""
    assessment = evaluate_rules("https://www.microsoft.com/en-us")
    aggregated = aggregate_risk(1.0, assessment)
    assert aggregated.risk_level == "MEDIUM"


def test_rules_can_escalate_to_high():
    assessment = evaluate_rules("http://paypal-secure-login.tk/verify?account=99")
    assert aggregate_risk(0.1, assessment).risk_level == "HIGH"


def test_low_signal_everything_is_low():
    assessment = evaluate_rules("https://example.com")
    assert aggregate_risk(0.02, assessment).risk_level == "LOW"


def test_aggregation_reports_basis_text():
    aggregated = aggregate_risk(0.9, evaluate_rules("https://example.com"))
    assert "capped" in aggregated.basis


# ------------------------------------------------------------- inference
def test_legitimate_url_with_path_is_not_high(phishing_detector):
    """Regression guard for the over-flagging failure mode."""
    for url in ("https://github.com/torvalds/linux",
                "https://en.wikipedia.org/wiki/Phishing",
                "https://docs.python.org/3/library/os.html"):
        result = phishing_detector.predict(url)
        assert result.risk_level == "MEDIUM", url
        assert result.risk_level != "HIGH"


def test_known_phishing_url_is_high(phishing_detector):
    result = phishing_detector.predict(
        "http://paypal-secure-login.tk/verify?account=99"
    )
    assert result.risk_level == "HIGH"
    assert result.recommendation


def test_result_has_no_raw_secret_and_explains_itself(phishing_detector):
    result = phishing_detector.predict("https://example.com")
    assert result.to_dict()["prediction"] in {"LEGITIMATE", "PHISHING"}
    assert result.risk_basis
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize("bad", ["", "   ", "http://", "javascript:alert(1)"])
def test_invalid_input_returns_unknown_not_crash(phishing_detector, bad):
    result = phishing_detector.predict(bad)
    assert result.prediction == "UNKNOWN"
    assert result.error


def test_empty_input_handled(phishing_detector):
    assert phishing_detector.predict("").prediction == "UNKNOWN"


def test_predict_many_isolates_failures(phishing_detector):
    results = phishing_detector.predict_many(
        ["https://example.com", "", "http://paypal-secure-login.tk/v"]
    )
    assert len(results) == 3
    assert results[0].prediction != "UNKNOWN"
    assert results[1].prediction == "UNKNOWN"
    assert results[2].risk_level == "HIGH"


# ------------------------------------------------- degraded / missing model
def test_missing_model_degrades_to_rules(missing_model_detector):
    assert missing_model_detector.load() is False
    result = missing_model_detector.predict("http://paypal-secure-login.tk/verify")
    assert result.model_available is False
    assert result.error
    # The rule layer must still produce a usable assessment.
    assert result.risk_level == "HIGH"
    assert result.indicators


def test_model_schema_is_validated(monkeypatch, phishing_detector):
    """A saved model with a different feature order must be rejected."""
    import joblib

    from modules.phishing_detector import PhishingDetector

    bad = {
        "pipeline": object(),
        "feature_names": ["wrong", "schema"],
        "metadata": {},
    }
    monkeypatch.setattr(joblib, "load", lambda _: bad)
    detector = PhishingDetector()
    assert detector.load() is False
    assert "mismatch" in (detector.load_error or "")


def test_corrupt_model_artifact_is_handled(monkeypatch):
    import joblib

    from modules.phishing_detector import PhishingDetector

    def explode(_):
        raise OSError("corrupt pickle")

    monkeypatch.setattr(joblib, "load", explode)
    detector = PhishingDetector()
    assert detector.load() is False
    result = detector.predict("https://example.com")
    assert result.model_available is False
    assert result.prediction == "UNKNOWN"


def test_model_error_does_not_crash(phishing_detector):
    class Boom:
        classes_ = [0, 1]

        def predict_proba(self, _frame):
            raise RuntimeError("model exploded")

    original = phishing_detector.pipeline
    try:
        phishing_detector.pipeline = Boom()
        result = phishing_detector.predict("https://example.com")
        assert result.error and "Model prediction error" in result.error
        assert result.prediction == "UNKNOWN"
    finally:
        phishing_detector.pipeline = original