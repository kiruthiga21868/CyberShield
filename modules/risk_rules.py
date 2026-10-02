"""Transparent rule layer for URL risk assessment.

The Random Forest in :mod:`modules.phishing_detector` is trained on a corpus
whose legitimate class contains only bare apex domains, so it reports a very
high phishing probability for any URL containing a path segment. Rather than
pretending otherwise, the final assessment is an explicit aggregation of two
independent signals:

1. the **ML probability** produced by the Random Forest, and
2. a **rule layer** of high-precision, individually explainable indicators.

Both are always reported side by side. The rule layer can escalate the result
to HIGH on its own; the ML signal alone can never exceed
``config.ML_ONLY_MAX_RISK`` (MEDIUM). Every threshold lives in ``config.py``.

Nothing in this module performs network access or opens a URL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlparse

from config import (
    DANGEROUS_EXTENSIONS,
    HIGH_RISK_TLDS,
    ML_MEDIUM_THRESHOLD,
    ML_ONLY_MAX_RISK,
    RULE_HIGH_COMBO,
    RULE_MEDIUM_COMBO,
    PHISHING_RULE_WEIGHTS,
)
from utils.url_features import InvalidURLError, extract_url_features, normalize_url


@dataclass
class RuleHit:
    """A single explainable rule observation."""

    code: str
    label: str
    severity: str  # HIGH | MEDIUM | INFO
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass
class RuleAssessment:
    """Aggregated rule-layer outcome."""

    hits: list[RuleHit] = field(default_factory=list)
    rule_score: float = 0.0

    @property
    def severities(self) -> list[str]:
        return [hit.severity for hit in self.hits]

    def has_severity(self, severity: str) -> bool:
        return any(hit.severity == severity for hit in self.hits)

    def labels(self, min_severity: str = "MEDIUM") -> list[str]:
        """Human-readable strings for indicators at or above ``min_severity``."""
        order = {"HIGH": 0, "MEDIUM": 1, "INFO": 2}
        cutoff = order.get(min_severity, 1)
        return [
            f"[{hit.severity}] {hit.label}" + (f" - {hit.detail}" if hit.detail else "")
            for hit in self.hits
            if order.get(hit.severity, 2) <= cutoff
        ]


_EMAIL_IN_HOST = re.compile(r"^[^/]*@", re.IGNORECASE)


def _is_private_ip(host: str) -> bool:
    """True when ``host`` is a loopback, private or link-local address."""
    import ipaddress

    candidate = host.strip("[]").rsplit(":", 1)[0] if host.count(":") == 1 else host
    try:
        address = ipaddress.ip_address(candidate.strip("[]"))
    except ValueError:
        return False
    return (
        address.is_private or address.is_loopback or address.is_link_local
        or address.is_reserved or address.is_unspecified
    )


def _path_of(url: str) -> str:
    try:
        return unquote(urlparse(normalize_url(url)).path or "").lower()
    except InvalidURLError:
        return ""


def evaluate_rules(url: str, features: dict[str, Any] | None = None) -> RuleAssessment:
    """Run every structural rule against ``url`` and return the hits."""
    try:
        feats = features or extract_url_features(url)
    except InvalidURLError as exc:
        return RuleAssessment(hits=[RuleHit(
            "invalid_url", "URL could not be parsed", "MEDIUM", str(exc)
        )])

    hits: list[RuleHit] = []
    cleaned = normalize_url(url)
    parsed = urlparse(cleaned)
    host = (parsed.netloc or "").rsplit("@", 1)[-1].lower()
    path = _path_of(url)

    # -- HIGH severity: individually strong phishing evidence ---------------
    if feats["host_is_ip"]:
        private = _is_private_ip(host)
        hits.append(RuleHit(
            "ip_host",
            "Host is a private/internal IP address" if private
            else "Host is a raw IP address",
            "MEDIUM" if private else "HIGH",
            f"{host} - a link to a bare IP address is unusual in legitimate "
            + ("internal services and is often used to bypass reputation checks"
               if private else "communications"),
        ))
    if feats["punycode"]:
        hits.append(RuleHit(
            "punycode", "Punycode homograph encoding", "HIGH",
            "xn-- encoding is used to imitate well-known domains",
        ))
    if _EMAIL_IN_HOST.match(parsed.netloc or "") or feats["at_symbol"]:
        hits.append(RuleHit(
            "at_in_url", "'@' present in the URL", "HIGH",
            "text before '@' is discarded by browsers and can disguise the real host",
        ))
    if feats["suspicious_words"] >= 3:
        hits.append(RuleHit(
            "keyword_stuffing", "Multiple phishing keywords", "HIGH",
            f"{feats['suspicious_words']} occurrences of login/verify/account-style words",
        ))
    if feats["nonstandard_port"]:
        hits.append(RuleHit(
            "nonstandard_port", "Non-standard network port", "HIGH",
            f"service is served on an unusual port rather than 80/443",
        ))
    if feats["suspicious_tld"]:
        hits.append(RuleHit(
            "abused_tld", "Heavily abused top-level domain", "HIGH",
            f"'.{feats['tld']}' is frequently used in disposable phishing domains",
        ))
    if path.endswith(tuple(DANGEROUS_EXTENSIONS)):
        extension = path.rsplit(".", 1)[-1]
        hits.append(RuleHit(
            "executable_payload", "URL points at an executable file", "HIGH",
            f"link target has extension '.{extension}'",
        ))
    if feats["at_symbol"] and feats["num_hyphens"] >= 4:
        hits.append(RuleHit(
            "deceptive_host", "Host mixes keywords and hyphens", "HIGH",
            "typical of look-alike brand domains",
        ))
    if feats["has_https"] == 0 and feats["suspicious_words"] >= 2 and feats["entropy"] >= 4.2:
        hits.append(RuleHit(
            "unencrypted_keyword_host", "Unencrypted keyword-heavy host", "HIGH",
            "no HTTPS combined with credential-style keywords and high entropy",
        ))

    # -- MEDIUM severity: worth attention on its own ------------------------
    if feats["has_https"] == 0:
        hits.append(RuleHit(
            "no_https", "No HTTPS", "MEDIUM", "traffic is not encrypted",
        ))
    if 0 < feats["suspicious_words"] < 3:
        hits.append(RuleHit(
            "suspicious_words", "Contains phishing-style keywords", "MEDIUM",
            f"{feats['suspicious_words']} keyword occurrence(s): "
            "login, verify, account, secure, ...",
        ))
    if feats["num_hyphens"] >= 3:
        hits.append(RuleHit(
            "many_hyphens", "Many hyphens in the URL", "MEDIUM",
            f"{feats['num_hyphens']} hyphen characters",
        ))
    if feats["url_length"] > 75:
        hits.append(RuleHit(
            "long_url", "Unusually long URL", "MEDIUM",
            f"{feats['url_length']} characters",
        ))
    if feats["url_length"] < 12:
        hits.append(RuleHit(
            "short_url", "Very short URL", "MEDIUM",
            f"only {feats['url_length']} characters - often a redirector",
        ))
    if feats["num_params"] >= 3:
        hits.append(RuleHit(
            "many_params", "Many query parameters", "MEDIUM",
            f"{feats['num_params']} parameters, a common data-exfiltration form",
        ))
    if feats["entropy"] >= 4.6:
        hits.append(RuleHit(
            "high_entropy", "High character entropy", "MEDIUM",
            f"{feats['entropy']:.2f} bits/char suggests a randomised hostname",
        ))
    if feats["digits_count"] >= 20:
        hits.append(RuleHit(
            "many_digits", "High digit count", "MEDIUM",
            f"{feats['digits_count']} digits, possible encoded payload",
        ))
    if feats["num_dots"] >= 5:
        hits.append(RuleHit(
            "many_subdomains", "Deeply nested subdomains", "MEDIUM",
            f"{feats['num_dots']} dots, often used to bury the real domain",
        ))
    if feats["double_slash_in_path"]:
        hits.append(RuleHit(
            "double_slash_path", "Repeated slashes in the path", "MEDIUM",
            "can be used to confuse URL parsers",
        ))

    # -- INFO severity: neutral or mildly reassuring ------------------------
    if feats["has_https"] == 1 and not feats["suspicious_words"]:
        hits.append(RuleHit("https", "Uses HTTPS", "INFO", "transport is encrypted"))
    if feats["tld"] in HIGH_RISK_TLDS:
        hits.append(RuleHit("tld_noted", "TLD is on the abuse watchlist", "INFO",
                            f"'.{feats['tld']}'"))

    score = min(
        1.0,
        sum(PHISHING_RULE_WEIGHTS.get(hit.severity, 0.0) for hit in hits),
    )
    return RuleAssessment(hits=hits, rule_score=round(score, 4))


@dataclass
class AggregatedRisk:
    """Final risk decision plus the reasoning that produced it."""

    risk_level: str
    basis: str
    ml_probability: float
    rule_score: float
    rule_hits: list[RuleHit] = field(default_factory=list)

    @property
    def indicators(self) -> list[str]:
        order = {"HIGH": 0, "MEDIUM": 1, "INFO": 2}
        return [
            f"[{hit.severity}] {hit.label}"
            + (f" - {hit.detail}" if hit.detail else "")
            for hit in self.rule_hits
            if order.get(hit.severity, 2) <= 1
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "basis": self.basis,
            "ml_probability": round(self.ml_probability, 4),
            "rule_score": round(self.rule_score, 4),
            "indicators": self.indicators,
            "rule_hits": [hit.to_dict() for hit in self.rule_hits],
        }


def aggregate_risk(
    ml_probability: float, assessment: RuleAssessment
) -> AggregatedRisk:
    """Combine the ML probability and rule score into a single risk level.

    Decision table (the ML signal alone can never reach HIGH)::

        rule_score >= RULE_HIGH_COMBO                 -> HIGH
        rule_score >= RULE_MEDIUM_COMBO               -> MEDIUM
        ml_probability >= ML_MEDIUM_THRESHOLD         -> MEDIUM (capped)
        otherwise                                     -> LOW

    The cap on the ML-only path is deliberate and is the reason a legitimate
    URL that merely contains a path is not reported as HIGH risk.
    """
    ml_probability = float(ml_probability or 0.0)
    rule_score = float(assessment.rule_score or 0.0)

    if rule_score >= RULE_HIGH_COMBO:
        return AggregatedRisk(
            "HIGH",
            f"Structural rules scored {rule_score:.2f} "
            f"(threshold {RULE_HIGH_COMBO:.2f}).",
            ml_probability, rule_score, assessment.hits,
        )

    if rule_score >= RULE_MEDIUM_COMBO:
        return AggregatedRisk(
            "MEDIUM",
            f"Structural rules scored {rule_score:.2f}; no rule reached the "
            "HIGH threshold.",
            ml_probability, rule_score, assessment.hits,
        )

    if ml_probability >= ML_MEDIUM_THRESHOLD:
        return AggregatedRisk(
            ML_ONLY_MAX_RISK,
            f"Random Forest reported a phishing probability of "
            f"{ml_probability:.0%}, but no structural indicator was found, so "
            f"the result is capped at {ML_ONLY_MAX_RISK}. See the training-data "
            "limitation note in the About page.",
            ml_probability, rule_score, assessment.hits,
        )

    return AggregatedRisk(
        "LOW",
        f"Random Forest probability {ml_probability:.0%} is below "
        f"{ML_MEDIUM_THRESHOLD:.0%} and no structural indicator was found.",
        ml_probability, rule_score, assessment.hits,
    )


__all__ = [
    "RuleHit",
    "RuleAssessment",
    "AggregatedRisk",
    "evaluate_rules",
    "aggregate_risk",
]