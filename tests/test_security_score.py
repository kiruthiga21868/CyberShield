"""Tests for the rule-based security score.

The central property under test is that an un-analysed component scores zero
and is reported as "Not analyzed", so a missing check can never inflate the
total.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modules.security_score import NOT_ANALYZED, build_report


@dataclass
class FakePassword:
    score: int = 90
    strength: str = "STRONG"
    is_valid_input: bool = True


@dataclass
class FakeScan:
    risk_level: str = "LOW"
    prediction: str = "LEGITIMATE"
    found: bool = True
    payloads: list = field(default_factory=list)


def qr_with_risk(risk: str) -> FakeScan:
    payload = type("P", (), {"kind": "url", "analysis": {"risk_level": risk}})()
    return FakeScan(risk_level=risk, payloads=[payload])


# ------------------------------------------------------------------- empty
def test_empty_session_scores_zero_and_is_incomplete():
    report = build_report()
    assert report.total == 0
    assert report.band == "NEEDS_ATTENTION"
    assert report.is_complete is False
    assert report.analyzed_count == 0
    assert all(not c.analyzed for c in report.components)
    assert all(c.status == NOT_ANALYZED for c in report.components)


def test_unanalysed_components_never_award_points():
    report = build_report(password_report=FakePassword(score=100))
    others = {c.key: c for c in report.components if c.key != "password"}
    assert all(c.points == 0 for c in others.values())
    assert report.total == 25  # only the password component earned points


# ------------------------------------------------------------------ bounds
def test_all_perfect_analyses_reach_full_score():
    report = build_report(
        password_report=FakePassword(score=100),
        url_result=FakeScan(risk_level="LOW"),
        email_result=FakeScan(risk_level="LOW", prediction="HAM"),
        qr_result=qr_with_risk("LOW"),
    )
    assert report.total == 100
    assert report.band == "GOOD"
    assert report.is_complete is True


def test_all_bad_analyses_score_zero():
    report = build_report(
        password_report=FakePassword(score=0, strength="WEAK"),
        url_result=FakeScan(risk_level="HIGH", prediction="PHISHING"),
        email_result=FakeScan(risk_level="HIGH", prediction="SPAM"),
        qr_result=qr_with_risk("HIGH"),
    )
    assert report.total == 0
    assert report.band == "NEEDS_ATTENTION"


def test_total_never_exceeds_max():
    for risk in ("LOW", "MEDIUM", "HIGH", "UNKNOWN"):
        report = build_report(
            password_report=FakePassword(score=100),
            url_result=FakeScan(risk_level=risk),
            email_result=FakeScan(risk_level=risk),
            qr_result=qr_with_risk(risk),
        )
        assert 0 <= report.total <= report.max_total == 100


# ------------------------------------------------------------------- bands
def test_band_thresholds_are_applied():
    def total_for(risk: str) -> int:
        return build_report(url_result=FakeScan(risk_level=risk)).total

    assert total_for("LOW") == 25          # 25 -> NEEDS_ATTENTION
    assert total_for("MEDIUM") == 12       # 12 -> NEEDS_ATTENTION
    assert total_for("HIGH") == 0


def test_band_boundaries():
    from modules.security_score import _band_for

    assert _band_for(100)[0] == "GOOD"
    assert _band_for(80)[0] == "GOOD"
    assert _band_for(79)[0] == "MODERATE"
    assert _band_for(60)[0] == "MODERATE"
    assert _band_for(59)[0] == "NEEDS_ATTENTION"
    assert _band_for(0)[0] == "NEEDS_ATTENTION"


def test_band_labels_are_human_readable():
    report = build_report(
        password_report=FakePassword(score=100),
        url_result=FakeScan(risk_level="LOW"),
        email_result=FakeScan(risk_level="LOW"),
        qr_result=qr_with_risk("LOW"),
    )
    assert report.band_label == "Good"
    assert build_report().band_label == "Needs Attention"


# ------------------------------------------------------------------ partial
def test_coverage_note_names_missing_checks():
    report = build_report(password_report=FakePassword())
    assert report.coverage_note.startswith("1 of 4")
    assert "URL Security" in report.coverage_note
    assert "partial evidence" in report.coverage_note


def test_complete_report_has_no_partial_warning():
    report = build_report(
        password_report=FakePassword(),
        url_result=FakeScan(),
        email_result=FakeScan(),
        qr_result=qr_with_risk("LOW"),
    )
    assert "All four checks" in report.coverage_note


# ---------------------------------------------------------------------- QR
def test_qr_with_no_url_is_not_scored():
    """A decoded plain-text QR cannot earn URL-safety points."""
    payload = type("P", (), {"kind": "text", "analysis": None})()
    result = FakeScan(payloads=[payload])
    report = build_report(qr_result=result)
    component = next(c for c in report.components if c.key == "qr")
    assert component.analyzed is False
    assert component.points == 0


def test_qr_uses_the_worst_payload_risk():
    low = type("P", (), {"kind": "url", "analysis": {"risk_level": "LOW"}})()
    high = type("P", (), {"kind": "url", "analysis": {"risk_level": "HIGH"}})()
    report = build_report(qr_result=FakeScan(payloads=[low, high]))
    component = next(c for c in report.components if c.key == "qr")
    assert component.verdict == "HIGH"
    assert component.points == 0


# ------------------------------------------------------------ serialisation
def test_report_is_serialisable():
    report = build_report(
        password_report=FakePassword(),
        url_result=FakeScan(),
        email_result=FakeScan(),
        qr_result=qr_with_risk("MEDIUM"),
    )
    payload = report.to_dict()
    assert payload["total"] == report.total
    assert len(payload["components"]) == 4
    assert all("status" in c for c in payload["components"])


def test_invalid_password_input_is_not_scored():
    report = build_report(
        password_report=FakePassword(score=100, is_valid_input=False)
    )
    component = next(c for c in report.components if c.key == "password")
    assert component.analyzed is False
    assert component.points == 0