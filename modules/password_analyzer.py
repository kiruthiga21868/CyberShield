"""Rule-based password strength analyzer.

Design constraints (deliberate and enforced by tests):

* the password is analysed **entirely on the local machine**;
* it is **never** sent to Groq, VirusTotal or any network service;
* it is **never** written to a log, a file or session state;
* only aggregate, non-reversible statistics leave this module.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from config import (
    COMMON_PASSWORDS,
    PASSWORD_BREACH_PATTERN_POINTS,
    PASSWORD_CHARCLASS_POINTS,
    PASSWORD_DIVERSITY_POINTS,
    PASSWORD_GOOD_LENGTH,
    PASSWORD_LENGTH_POINTS,
    PASSWORD_MAX_LENGTH,
    PASSWORD_MIN_LENGTH,
    PASSWORD_SEQUENCE_POINTS,
    PASSWORD_STRENGTH_THRESHOLDS,
    WEAK_PATTERNS,
)

_SEQUENCES = (
    "abcdefghijklmnopqrstuvwxyz",
    "0123456789",
    "qwertyuiop",
    "asdfghjkl",
    "zxcvbnm",
)
_REPEAT_RUN = re.compile(r"(.)\1{2,}")
_KEYBOARD_RUN = re.compile(r"(?:0123|1234|2345|3456|4567|5678|6789|abcd|bcde|cdef|"
                           r"defg|efgh|fghi|ghi|qwer|wert|erty|rty|tyui|yuio|uiop"
                           r"|asdf|sdfg|dfgh|fghj|ghjk|zxcv|xcvb|cvbn|vbnm)")
_YEAR = re.compile(r"(19|20)\d{2}")


@dataclass
class PasswordReport:
    """Aggregate password assessment. Never contains the password itself."""

    strength: str
    score: int
    entropy_bits: float
    length: int
    checks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    is_valid_input: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strength": self.strength,
            "score": self.score,
            "entropy_bits": round(self.entropy_bits, 1),
            "length": self.length,
            "checks": self.checks,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "is_valid_input": self.is_valid_input,
            "error": self.error,
        }


def _charset_size(password: str) -> int:
    """Size of the character repertoire actually used by the password."""
    size = 0
    if any(char.islower() and char.isascii() for char in password):
        size += 26
    if any(char.isupper() and char.isascii() for char in password):
        size += 26
    if any(char.isdigit() and char.isascii() for char in password):
        size += 10
    if any(not char.isalnum() for char in password):
        size += 32
    if any(ord(char) > 127 for char in password):
        size += 100
    return max(size, 1)


def estimate_entropy(password: str) -> float:
    """Estimate Shannon entropy in bits assuming a uniform random draw.

    This is an approximation: it does not model human password-selection
    biases, so it is reported as an informational figure only.
    """
    if not password:
        return 0.0
    return len(password) * math.log2(_charset_size(password))


def _has_sequential_run(password: str) -> bool:
    lowered = password.lower()
    for sequence in _SEQUENCES:
        for start in range(len(sequence) - 2):
            chunk = sequence[start:start + 3]
            if chunk in lowered or chunk[::-1] in lowered:
                return True
    return bool(_KEYBOARD_RUN.search(lowered))


def _masked_preview(password: str) -> str:
    """A non-reversible hint, used only inside recommendations text."""
    if not password:
        return ""
    classes = 0
    classes += any(c.islower() for c in password)
    classes += any(c.isupper() for c in password)
    classes += any(c.isdigit() for c in password)
    classes += any(not c.isalnum() for c in password)
    return "*" * min(len(password), 16)


def analyze_password(password: str) -> PasswordReport:
    """Score a password locally and return an explainable report.

    The password is only read, never retained beyond this call.
    """
    if password is None or not isinstance(password, str) or password == "":
        return PasswordReport(
            strength="UNKNOWN", score=0, entropy_bits=0.0, length=0,
            checks=[], warnings=["No password was provided."],
            recommendations=["Enter a password to analyse it."],
            is_valid_input=False, error="Empty password.",
        )

    if len(password) > PASSWORD_MAX_LENGTH:
        return PasswordReport(
            strength="UNKNOWN", score=0, entropy_bits=0.0,
            length=len(password), checks=[],
            warnings=[f"Password exceeds the {PASSWORD_MAX_LENGTH}-character limit."],
            recommendations=["Use a password of at most "
                             f"{PASSWORD_MAX_LENGTH} characters."],
            is_valid_input=False, error="Password too long.",
        )

    normalized = unicodedata.normalize("NFKC", password)
    lowered = normalized.lower()
    length = len(normalized)

    has_lower = any(c.islower() for c in normalized)
    has_upper = any(c.isupper() for c in normalized)
    has_digit = any(c.isdigit() for c in normalized)
    has_symbol = any(not c.isalnum() for c in normalized)
    class_count = sum([has_lower, has_upper, has_digit, has_symbol])

    checks: list[str] = []
    warnings: list[str] = []
    recommendations: list[str] = []
    score = 0

    # -- length ------------------------------------------------------------
    if length >= PASSWORD_GOOD_LENGTH:
        score += PASSWORD_LENGTH_POINTS
        checks.append(f"Length {length} characters (meets the {PASSWORD_GOOD_LENGTH}+ target)")
    elif length >= PASSWORD_MIN_LENGTH:
        partial = PASSWORD_LENGTH_POINTS * 0.6
        score += partial
        checks.append(f"Length {length} characters (minimum is {PASSWORD_MIN_LENGTH})")
        recommendations.append(
            f"Extend to at least {PASSWORD_GOOD_LENGTH} characters; length matters "
            "more than symbol count."
        )
    else:
        checks.append(f"Length {length} characters (below the {PASSWORD_MIN_LENGTH} minimum)")
        warnings.append("Password is too short to resist offline cracking.")
        recommendations.append(f"Use at least {PASSWORD_MIN_LENGTH} characters.")

    # -- character classes -------------------------------------------------
    earned = 0
    if has_lower:
        earned += 1
    if has_upper:
        earned += 1
    if has_digit:
        earned += 1
    if has_symbol:
        earned += 1
    score += PASSWORD_CHARCLASS_POINTS * (earned / 4)
    present = [
        name for name, flag in
        (("lowercase", has_lower), ("uppercase", has_upper),
         ("digits", has_digit), ("symbols", has_symbol)) if flag
    ]
    if class_count == 4:
        checks.append("Contains all four character types")
    elif class_count >= 2:
        checks.append(f"Contains {class_count} of 4 character types ({', '.join(present)})")
    else:
        warnings.append("Uses only one character type, which is easy to guess.")
        recommendations.append("Mix uppercase, lowercase, digits and symbols.")
    if class_count == 1:
        score -= 10

    # -- diversity and predictable sequences -------------------------------
    diversity_issues: list[str] = []
    sequence_issues: list[str] = []

    if _REPEAT_RUN.search(normalized):
        diversity_issues.append("Contains 3 or more repeated characters in a row")
    unique_ratio = len(set(normalized)) / length
    if length >= 8 and unique_ratio < 0.45:
        diversity_issues.append(
            f"Only {len(set(normalized))} distinct characters out of {length}"
        )

    if _has_sequential_run(normalized):
        sequence_issues.append("Contains a keyboard or alphabetic sequence")
    if _YEAR.search(normalized):
        sequence_issues.append("Contains what looks like a year")

    if not diversity_issues:
        score += PASSWORD_DIVERSITY_POINTS
        checks.append("Good character variety with no obvious repetition")
    else:
        ratio_penalty = (PASSWORD_DIVERSITY_POINTS
                         * min(1.0, len(diversity_issues) / 2))
        score += PASSWORD_DIVERSITY_POINTS - ratio_penalty
        warnings.extend(diversity_issues)
        recommendations.append("Avoid repeating the same character several times.")

    if not sequence_issues:
        score += PASSWORD_SEQUENCE_POINTS
        checks.append("No predictable keyboard or alphabetic sequences")
    else:
        penalty = PASSWORD_SEQUENCE_POINTS * min(1.0, len(sequence_issues) / 2)
        score += PASSWORD_SEQUENCE_POINTS - penalty
        warnings.extend(sequence_issues)
        recommendations.append("Avoid sequences like 'abc', '123' or 'qwerty'.")

    # -- known breached patterns ------------------------------------------
    breach_issues: list[str] = []
    if lowered in COMMON_PASSWORDS:
        breach_issues.append("Appears in a list of commonly used passwords")
    else:
        for pattern in WEAK_PATTERNS:
            if pattern in lowered:
                breach_issues.append(f"Contains the well-known weak pattern '{pattern}'")
                break
    if lowered.isdigit():
        breach_issues.append("Consists only of digits")

    if not breach_issues:
        score += PASSWORD_BREACH_PATTERN_POINTS
        checks.append("No well-known breached-password pattern detected")
    else:
        warnings.extend(breach_issues)
        recommendations.append(
            "Do not use dictionary words or names of common services; assume any "
            "pattern on this list is already in an attacker's wordlist."
        )

    score = int(max(0, min(100, round(score))))

    if length >= PASSWORD_GOOD_LENGTH and class_count == 4 and not breach_issues:
        recommendations.append("Good length and character variety.")
    recommendations.append("Never reuse this password across websites.")
    recommendations.append("Enable multi-factor authentication where available.")
    recommendations.append(
        "Use a reputable password manager so a unique password can be kept per site."
    )
    if not breach_issues:
        recommendations.append(
            "This password is not in our local weak-password list, but only an "
            "online breach-database check can confirm it has never leaked."
        )

    strength = "WEAK"
    for threshold, label in PASSWORD_STRENGTH_THRESHOLDS:
        if score >= threshold:
            strength = label
            break

    return PasswordReport(
        strength=strength,
        score=score,
        entropy_bits=estimate_entropy(normalized),
        length=length,
        checks=checks,
        warnings=warnings,
        recommendations=recommendations,
    )


def strength_from_score(score: int) -> str:
    """Map a 0-100 score onto WEAK / MEDIUM / STRONG."""
    for threshold, label in PASSWORD_STRENGTH_THRESHOLDS:
        if score >= threshold:
            return label
    return "WEAK"