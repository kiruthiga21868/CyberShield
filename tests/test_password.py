"""Tests for the local, rule-based password analyzer.

These tests also enforce the privacy guarantees: a password must never leave
the machine and must never be retained in a report.
"""
from __future__ import annotations

import pytest

from config import PASSWORD_MAX_LENGTH
from modules.password_analyzer import analyze_password, estimate_entropy

WEAK = ["a", "123456", "password", "password123", "qwerty12345", "Summer2024",
        "aaaaaaaaaaaaaaaa"]
MEDIUM = ["P@ssw0rd", "BlueTiger99", "correct-horse"]
STRONG = ["Xk9#mP2$vL8@nQ4w", "Tr0ub4dor&3xyz!", "7#Hj2!pQ@wZ4^tR6&"]


# ------------------------------------------------------------------ privacy
def test_report_never_contains_the_password():
    secret = "SuperSecret-Passw0rd-2024"
    report = analyze_password(secret)
    serialised = str(report.to_dict())
    assert secret not in serialised
    for value in report.checks + report.warnings + report.recommendations:
        assert secret not in value


def test_report_dict_has_no_raw_password_field():
    report = analyze_password("Hunter2Hunter2")
    assert set(report.to_dict()) == {
        "strength", "score", "entropy_bits", "length", "checks", "warnings",
        "recommendations", "is_valid_input", "error",
    }


# ----------------------------------------------------------------- ordering
@pytest.mark.parametrize("password", WEAK)
def test_weak_passwords_score_weak(password):
    report = analyze_password(password)
    assert report.strength == "WEAK", f"{password} -> {report.strength}"
    assert report.score < 55


@pytest.mark.parametrize("password", STRONG)
def test_strong_passwords_score_strong(password):
    report = analyze_password(password)
    assert report.strength == "STRONG", f"{password} -> {report.strength}"
    assert report.score >= 80


def test_medium_passwords_are_between():
    report = analyze_password("P@ssw0rd")
    assert 0 <= report.score <= 100
    assert report.strength in {"WEAK", "MEDIUM", "STRONG"}


def test_score_is_ordered_by_complexity():
    scores = [analyze_password(p).score for p in
              ("password", "Password1", "Password1!", "Xk9#mP2$vL8@nQ4w")]
    assert scores == sorted(scores)


def test_perfect_score_is_reachable():
    report = analyze_password("correct horse battery staple 7#Hq")
    assert report.score == 100


def test_score_is_always_within_bounds():
    for password in WEAK + MEDIUM + STRONG:
        assert 0 <= analyze_password(password).score <= 100


# ------------------------------------------------------------------ checks
def test_empty_password_is_rejected():
    report = analyze_password("")
    assert report.strength == "UNKNOWN"
    assert report.is_valid_input is False
    assert report.error == "Empty password."


def test_none_password_is_rejected():
    assert analyze_password(None).is_valid_input is False


def test_overlong_password_is_rejected():
    report = analyze_password("a1!" * (PASSWORD_MAX_LENGTH // 2))
    assert report.is_valid_input is False
    assert "too long" in report.error.lower()


def test_known_weak_list_is_detected():
    report = analyze_password("password123")
    assert any("commonly used" in warning for warning in report.warnings)


def test_sequence_is_detected():
    report = analyze_password("abcdefgh12345678")
    assert any("sequence" in warning.lower() for warning in report.warnings)


def test_repeated_characters_are_detected():
    report = analyze_password("aaaa")
    assert any("repeated" in warning.lower() for warning in report.warnings)


def test_unicode_password_is_handled():
    report = analyze_password("Pässwörd123!ünicode")
    assert report.strength in {"MEDIUM", "STRONG"}
    assert report.length > 0


def test_entropy_scales_with_length():
    assert estimate_entropy("aB3!") < estimate_entropy("aB3!aB3!aB3!aB3!")


def test_recommendations_always_present():
    for password in WEAK + STRONG:
        assert analyze_password(password).recommendations