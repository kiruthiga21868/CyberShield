"""Small shared helpers: risk classification, recommendations, text cleaning."""
from __future__ import annotations

import re
from typing import Any, Iterable

from config import (
    PHISHING_DEFAULT_RISK,
    PHISHING_LOW_CONFIDENCE_BELOW,
    PHISHING_RISK_THRESHOLDS,
    SPAM_LOW_CONFIDENCE_BELOW,
    SPAM_RISK_THRESHOLDS,
)


def classify_risk(
    probability: float,
    thresholds: dict[str, float] | None = None,
    default: str | None = None,
) -> str:
    """Map a probability onto a LOW / MEDIUM / HIGH risk band.

    Thresholds are explicit configuration rather than magic numbers so the
    decision boundary is inspectable during a project review.
    """
    if probability is None:
        return default or PHISHING_DEFAULT_RISK
    table = thresholds if thresholds is not None else PHISHING_RISK_THRESHOLDS
    fallback = default if default is not None else PHISHING_DEFAULT_RISK
    for level in ("HIGH", "MEDIUM"):
        cutoff = table.get(level)
        if cutoff is not None and probability >= cutoff:
            return level
    return fallback


def confidence_label(probability: float, threshold: float) -> str:
    """Describe how much weight the UI should place on a prediction."""
    if probability is None:
        return "Unknown"
    if probability >= 0.85:
        return "Very high confidence"
    if probability >= threshold:
        return "Moderate confidence"
    return "Low confidence - treat this result cautiously"


def phishing_recommendation(risk_level: str) -> str:
    """Actionable guidance, worded so no result is presented as a guarantee."""
    if risk_level == "HIGH":
        return (
            "Do not open this URL or enter credentials, card details or personal "
            "information. Verify the organisation through a channel you already "
            "trust (type the address yourself or use a saved bookmark)."
        )
    if risk_level == "MEDIUM":
        return (
            "This URL shows some suspicious characteristics but the verdict is not "
            "conclusive. Verify the sender and destination before proceeding, and do "
            "not submit sensitive information on first contact."
        )
    return (
        "No strong phishing indicators were detected. This is not a guarantee of "
        "safety - stay alert to unexpected requests for credentials or payment "
        "details."
    )


def spam_recommendation(prediction: str, risk: str) -> str:
    """Guidance for a spam-classified message."""
    if prediction == "SPAM":
        return (
            "Do not click links, reply, or call phone numbers contained in this "
            "message. Mark it as spam and delete it. If it impersonates an "
            "organisation you use, contact that organisation through its official "
            "website instead."
        )
    if risk == "MEDIUM":
        return (
            "This message looks largely legitimate but shares traits with spam. "
            "Verify any links before opening them and be cautious about requests "
            "for personal or financial information."
        )
    return (
            "No strong spam indicators were detected. Still verify unexpected "
            "requests for money, credentials or personal data."
    )


_WHITESPACE = re.compile(r"\s+")
_URL_IN_TEXT = re.compile(
    r"""(?i)\b((?:https?://|www\.)[^\s<>"'\)\]]+)""",
)
_EMAIL_IN_TEXT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def clean_email_text(text: str) -> str:
    """Normalise an email body without destroying useful signal.

    Lower-casing is intentionally left to the vectorizer so the trained model
    and inference share identical behaviour.
    """
    if not text:
        return ""
    text = text.replace("\ufffd", " ")
    # Collapse an email reply chain, which adds length but no new signal.
    text = re.split(r"^\s*>{1,}", text, flags=re.MULTILINE)[0]
    return _WHITESPACE.sub(" ", text).strip()


def extract_urls_from_text(text: str) -> list[str]:
    """Return every URL-like token found in free text.

    Used to offer a separate URL scan from the email scanner. Nothing is ever
    opened or fetched automatically.
    """
    if not text:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for match in _URL_IN_TEXT.findall(text):
        candidate = match.rstrip(".,;:!?)\"'")
        key = candidate.lower()
        if key not in seen:
            seen.add(key)
            found.append(candidate)
    return found


def extract_emails_from_text(text: str) -> list[str]:
    """Return email-address-like tokens found in free text."""
    if not text:
        return []
    return list(dict.fromkeys(_EMAIL_IN_TEXT.findall(text)))


def spam_indicators(text: str) -> list[str]:
    """Rule based, explainable spam signals for the UI."""
    if not text:
        return []
    indicators: list[str] = []
    lowered = text.lower()
    urls = extract_urls_from_text(text)
    if urls:
        indicators.append(f"Contains {len(urls)} link(s)")
    if len(urls) >= 5:
        indicators.append("Unusually high number of links")
    if extract_emails_from_text(text):
        indicators.append("Contains an email address (reply-to manipulation)")
    money = [w for w in ("£", "$", "€", "usd", "eur", "gbp", "rs.", "inr")
             if w in lowered]
    if money:
        indicators.append(f"Mentions currency amounts ({', '.join(sorted(set(money))[:3])})")
    if re.search(r"\b(win|winner|won|prize|lottery|jackpot)\b", lowered):
        indicators.append("Prize or lottery language")
    if re.search(r"\b(free|discount|offer|deal|bonus|coupon)\b", lowered):
        indicators.append("Promotional or urgency language")
    if re.search(r"\b(click|click here|tap now|open now)\b", lowered):
        indicators.append("Direct call-to-action to click")
    if re.search(r"\b(urgent|immediately|suspend|verify your account|act now)\b", lowered):
        indicators.append("Urgency or account-suspension pressure")
    if re.search(r"\b(viagra|cialis|casino|lottery|crypto|bitcoin|bet)\b", lowered):
        indicators.append("Regulated or high-risk product category")
    if re.search(r"\b(\+?\d[\d\s\-]{8,})\b", text):
        indicators.append("Contains long phone-number-like sequences (premium rate)")
    if text.count("!") >= 3:
        indicators.append("Excessive exclamation marks")
    if re.search(r"[A-Z]{6,}", text):
        indicators.append("Long runs of capital letters (shouting)")
    return indicators


def format_percentage(value: float | None) -> str:
    """Render a probability as a percentage string."""
    if value is None:
        return "N/A"
    return f"{max(0.0, min(1.0, float(value))) * 100:.1f}%"


def truncate(text: str, limit: int = 300) -> str:
    """Shorten text for display without losing context."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def first_not_empty(items: Iterable[Any], default: str = "") -> str:
    """Return the first truthy item from ``items`` as a string."""
    for item in items:
        if item:
            return str(item)
    return default