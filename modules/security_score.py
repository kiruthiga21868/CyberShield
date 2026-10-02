"""Rule-based, transparent overall security score.

This module is deliberately **not** machine learned. Every point is earned by
a documented rule so the score can be explained to a user and defended during
a review. All weights and band thresholds live in ``config.py``.

A component the user has not analysed contributes **zero** points and is
labelled "Not analyzed". The score is therefore never inflated by missing
data, and the report always shows how many of the 100 points were actually
assessable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import SCORE_BANDS, SCORE_MAX, SCORE_UNANALYZED_POINTS, SCORE_WEIGHTS

COMPONENT_LABELS: dict[str, str] = {
    "password": "Password Security",
    "url": "URL Security",
    "email": "Email Security",
    "qr": "QR Security",
}

NOT_ANALYZED = "Not analyzed"


@dataclass
class ComponentScore:
    """Score contribution of a single analysis."""

    key: str
    label: str
    analyzed: bool
    points: int
    max_points: int
    detail: str = ""
    verdict: str = ""

    @property
    def status(self) -> str:
        return self.verdict if self.analyzed else NOT_ANALYZED

    @property
    def percent(self) -> float:
        if not self.analyzed or self.max_points == 0:
            return 0.0
        return round(self.points / self.max_points * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "analyzed": self.analyzed,
            "points": self.points,
            "max_points": self.max_points,
            "detail": self.detail,
            "verdict": self.verdict,
            "status": self.status,
            "percent": self.percent,
        }


@dataclass
class SecurityReport:
    """The full scored report."""

    components: list[ComponentScore] = field(default_factory=list)
    total: int = 0
    max_total: int = SCORE_MAX
    band: str = ""
    band_label: str = ""
    analyzed_count: int = 0
    total_count: int = len(SCORE_WEIGHTS)

    @property
    def is_complete(self) -> bool:
        return self.analyzed_count == self.total_count

    @property
    def coverage_note(self) -> str:
        if self.is_complete:
            return "All four checks have been analysed."
        missing = [c.label for c in self.components if not c.analyzed]
        return (
            f"{self.analyzed_count} of {self.total_count} checks analysed. "
            f"{', '.join(missing)} still contribute 0 points, so this score is "
            "based on partial evidence."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "max_total": self.max_total,
            "band": self.band,
            "band_label": self.band_label,
            "analyzed_count": self.analyzed_count,
            "total_count": self.total_count,
            "is_complete": self.is_complete,
            "coverage_note": self.coverage_note,
            "components": [component.to_dict() for component in self.components],
        }


def _band_for(score: int) -> tuple[str, str]:
    """Map a raw score onto a configured risk band."""
    for threshold, band in SCORE_BANDS:
        if score >= threshold:
            return band, {
                "GOOD": "Good",
                "MODERATE": "Moderate",
                "NEEDS_ATTENTION": "Needs Attention",
            }.get(band, band.title())
    return SCORE_BANDS[-1][1], SCORE_BANDS[-1][1].title()


def _risk_points(risk: str, weight: int) -> int | None:
    """Convert a LOW/MEDIUM/HIGH risk verdict into points, or None if unknown."""
    return {"LOW": weight, "MEDIUM": weight // 2, "HIGH": 0}.get(risk)


def _password_points(report: Any) -> tuple[int, str, str]:
    """Award points from the password analyzer's 0-100 score."""
    if report is None or not getattr(report, "is_valid_input", False):
        return SCORE_UNANALYZED_POINTS, "", NOT_ANALYZED
    ratio = max(0.0, min(1.0, report.score / 100.0))
    points = int(round(SCORE_WEIGHTS["password"] * ratio))
    return points, f"Strength {report.strength} ({report.score}/100)", report.strength


def _scanner_points(result: Any, key: str) -> tuple[int, str, str]:
    """Award points from a risk level produced by any scanner."""
    if result is None:
        return SCORE_UNANALYZED_POINTS, "", NOT_ANALYZED
    risk = getattr(result, "risk_level", "UNKNOWN")
    points = _risk_points(risk, SCORE_WEIGHTS[key])
    if points is None:
        return SCORE_UNANALYZED_POINTS, "", NOT_ANALYZED
    prediction = getattr(result, "prediction", "")
    detail = f"Risk {risk}" + (f" (model: {prediction})" if prediction else "")
    return points, detail, risk


def _email_points(result: Any) -> tuple[int, str, str]:
    """Award points from a spam scan's risk level."""
    return _scanner_points(result, "email")


def _url_points(result: Any) -> tuple[int, str, str]:
    """Award points from a phishing scan's risk level."""
    return _scanner_points(result, "url")


def _qr_points(result: Any) -> tuple[int, str, str]:
    """Award points from a QR scan's worst payload risk."""
    if result is None or not getattr(result, "found", False):
        return SCORE_UNANALYZED_POINTS, "", NOT_ANALYZED
    payloads = getattr(result, "payloads", []) or []
    risks = []
    for payload in payloads:
        analysis = getattr(payload, "analysis", None) or {}
        risk = analysis.get("risk_level")
        if risk:
            risks.append(risk)
    if not risks:
        # A QR was decoded but contained no URL, so nothing was scored.
        kinds = {getattr(p, "kind", "text") for p in payloads}
        return (SCORE_UNANALYZED_POINTS,
                f"QR decoded ({', '.join(sorted(kinds))}) - contained no link to score",
                NOT_ANALYZED)
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    worst = max(risks, key=lambda r: order.get(r, -1))
    points = _risk_points(worst, SCORE_WEIGHTS["qr"])
    if points is None:
        return SCORE_UNANALYZED_POINTS, "", NOT_ANALYZED
    return points, f"Worst payload risk {worst}", worst


_BUILDERS = {
    "password": _password_points,
    "url": _url_points,
    "email": _email_points,
    "qr": _qr_points,
}


def build_report(
    password_report: Any = None,
    url_result: Any = None,
    email_result: Any = None,
    qr_result: Any = None,
) -> SecurityReport:
    """Aggregate the session's analyses into a scored report.

    Any argument left as ``None`` is treated as "not analysed" and scores zero.
    """
    inputs = {
        "password": password_report,
        "url": url_result,
        "email": email_result,
        "qr": qr_result,
    }

    components: list[ComponentScore] = []
    for key, weight in SCORE_WEIGHTS.items():
        supplied = inputs.get(key)
        points, detail, verdict = _BUILDERS[key](supplied)
        components.append(ComponentScore(
            key=key,
            label=COMPONENT_LABELS[key],
            analyzed=verdict != NOT_ANALYZED,
            points=int(max(0, min(weight, points))),
            max_points=weight,
            detail=detail,
            verdict=verdict,
        ))

    total = sum(component.points for component in components)
    band, band_label = _band_for(total)
    analyzed = sum(1 for component in components if component.analyzed)

    return SecurityReport(
        components=components,
        total=total,
        max_total=SCORE_MAX,
        band=band,
        band_label=band_label,
        analyzed_count=analyzed,
        total_count=len(SCORE_WEIGHTS),
    )


__all__ = ["build_report", "SecurityReport", "ComponentScore", "NOT_ANALYZED"]