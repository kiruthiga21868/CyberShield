"""Reusable URL feature extraction for the phishing detector.

This module is the single source of truth for the phishing feature schema.
Training (``training/train_phishing.py``) and inference
(``modules/phishing_detector.py``) both call :func:`extract_url_features`,
which guarantees an identical feature contract at train and inference time.

Why the features are re-derived from the raw URL
------------------------------------------------
The supplied ``phishing_features.csv`` ships with pre-extracted columns, but
its extractor was inconsistent: for all 820 legitimate rows (which are bare,
scheme-less apex domains such as ``google.com``) it stored ``tld = NaN`` and
``has_https = 0``, even though ``google.com`` plainly has TLD ``com``. Those
two columns were extraction artifacts rather than real signal, and they are
the reason the classes looked perfectly separable.

Re-deriving every column from the raw ``url`` text with one consistent
implementation removes the artifact and makes the schema honest. No label is
modified and no row is added, removed or reweighted at this stage.
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse, unquote

from config import HIGH_RISK_TLDS, SUSPICIOUS_KEYWORDS

# Canonical feature order. Persisted with the model and re-validated on load.
FEATURE_NAMES: tuple[str, ...] = (
    "url_length",
    "num_dots",
    "num_hyphens",
    "has_https",
    "has_ip",
    "host_length",
    "num_subdirs",
    "num_params",
    "suspicious_words",
    "special_char_count",
    "digits_count",
    "entropy",
    "at_symbol",
    "double_slash_in_path",
    "punycode",
    "host_is_ip",
    "nonstandard_port",
    "suspicious_tld",
    "tld_entropy",
    "path_entropy",
    "query_entropy",
)

# Non-numeric columns that are fitted on the training corpus only.
CATEGORICAL_FEATURES: tuple[str, ...] = ("tld",)

ALL_FEATURE_NAMES: tuple[str, ...] = FEATURE_NAMES + CATEGORICAL_FEATURES

# Characters treated as "special" for the special_char_count feature. The URL
# structural delimiters are excluded so the count reflects obfuscation attempts
# (e.g. the @ and percent-encoding seen in phishing URLs) rather than syntax.
_STRUCTURAL_CHARS = set("/:.-_?=&%#")


class InvalidURLError(ValueError):
    """Raised when a string cannot be interpreted as a URL at all."""


def shannon_entropy(text: str) -> float:
    """Shannon entropy (bits per character) of ``text``.

    Returns ``0.0`` for empty input. High entropy indicates randomised or
    obfuscated hostnames, a common trait of phishing infrastructure.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length) for count in counts.values()
    )


def normalize_url(url: str) -> str:
    """Trim and prepend a scheme when the user omitted one.

    Users type ``example.com`` far more often than ``https://example.com``.
    Adding the default scheme does not change the host and keeps the feature
    values comparable with the training corpus.
    """
    candidate = (url or "").strip()
    # Strip control characters and surrounding whitespace/quotes that are
    # commonly pasted from email clients or spreadsheets.
    candidate = candidate.strip().strip("<>\"'` \t\r\n")
    if not candidate:
        raise InvalidURLError("URL is empty.")
    if len(candidate) > 4096:
        raise InvalidURLError("URL exceeds the maximum supported length.")
    if "://" not in candidate:
        candidate = f"https://{candidate.lstrip('/')}"
    return candidate


_HOST_PATTERN = re.compile(r"^(?!-)[\w.\-]{1,253}$", re.UNICODE)


def _validate_host(host: str) -> None:
    """Reject hosts that cannot represent a real destination.

    Guards against inputs such as ``://///`` (empty authority) and
    ``javascript:alert(1)`` (script pseudo-URL), which would otherwise reach
    the model as meaningless feature vectors.
    """
    if not host:
        raise InvalidURLError("URL does not contain a host name.")
    if _is_ip_literal(host):
        return
    if any(char.isspace() for char in host):
        raise InvalidURLError("URL host contains whitespace or an invalid character.")
    if not _HOST_PATTERN.match(host):
        raise InvalidURLError(f"'{truncate_host(host)}' is not a valid host name.")


def truncate_host(host: str, limit: int = 40) -> str:
    """Shorten a host for safe display inside an error message."""
    host = host or ""
    return host if len(host) <= limit else host[:limit] + "..."


def validate_and_normalize_url(url: str) -> str:
    """Normalise a URL and confirm it has a usable host.

    Used by callers that need a trustworthy URL string without computing the
    whole feature vector (for example the VirusTotal service).
    """
    cleaned = normalize_url(url)
    host, _ = _strip_port(urlparse(cleaned).netloc.rsplit("@", 1)[-1].lower())
    _validate_host(host)
    return cleaned


def _is_ip_literal(host: str) -> bool:
    """Return True when ``host`` is a bare IPv4/IPv6 literal."""
    if not host:
        return False
    candidate = _strip_port(host)[0]
    if candidate.startswith("[") and "]" in candidate:
        try:
            ipaddress.IPv6Address(candidate.strip("[]"))
            return True
        except ValueError:
            return False
    if candidate.replace(".", "").isdigit():
        try:
            ipaddress.IPv4Address(candidate)
            return True
        except ValueError:
            return False
    return False


def _strip_port(host: str) -> tuple[str, int | None]:
    """Split ``host`` into (hostname, port), tolerating IPv6 bracket syntax.

    Phishing URLs frequently serve content on non-standard ports such as
    ``http://185.1.2.3:44512/bin.sh``, so the port must be separated before
    any IP-literal check is performed.
    """
    host = (host or "").strip()
    if host.startswith("["):  # [::1]:8080
        closing = host.find("]")
        if closing == -1:
            return host, None
        hostname = host[: closing + 1]
        remainder = host[closing + 1:]
        if remainder.startswith(":"):
            port_text = remainder[1:]
            return hostname, int(port_text) if port_text.isdigit() else None
        return hostname, None
    if ":" in host:
        hostname, _, port_text = host.rpartition(":")
        if port_text.isdigit():
            return hostname, int(port_text)
        return host, None
    return host, None


def _tld_of(host: str) -> str:
    """Best-effort public-suffix approximation (last label of the host).

    A full public-suffix-list dependency is deliberately avoided; the last
    label is a stable, explainable signal and matches how the feature is
    described in the dataset documentation.
    """
    host = host.strip("[]").lower()
    if not host or _is_ip_literal(host):
        return ""
    labels = [label for label in host.split(".") if label]
    if len(labels) < 2:
        return ""
    return labels[-1]


def _split_parts(parsed) -> tuple[str, str, str, int | None]:
    """Return (host, path, query, port) from a parsed URL, lower-casing host."""
    netloc = parsed.netloc or ""
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]  # drop userinfo
    host, port = _strip_port(netloc.lower())
    return host, parsed.path or "", parsed.query or "", port


def extract_url_features(url: str) -> dict[str, Any]:
    """Extract the phishing feature vector from a single URL string.

    The returned dictionary always contains exactly :data:`ALL_FEATURE_NAMES`
    keys, in that order, so the frame handed to the model always matches the
    schema seen during training.

    Raises:
        InvalidURLError: if the input is empty or has no extractable host.
    """
    cleaned = normalize_url(url)
    parsed = urlparse(cleaned)
    host, path, query, port = _split_parts(parsed)

    _validate_host(host)

    if not host and not path:
        raise InvalidURLError("URL does not contain a valid host.")

    # Surface-level counts are taken from the URL exactly as written, because
    # that is what a user actually sees and inspects. The decoded text is used
    # for obfuscation analysis (entropy and keyword counts), which is what
    # defeats percent-encoding tricks such as "%2F" or "%3d".
    decoded = unquote(cleaned)

    segments = [segment for segment in path.split("/") if segment]
    query_pairs = [pair for pair in query.split("&") if pair]

    lower_host = host.lower()
    lower_decoded = decoded.lower()

    suspicious_words = sum(lower_decoded.count(word) for word in SUSPICIOUS_KEYWORDS)

    digits = sum(char.isdigit() for char in cleaned)
    special = sum(
        (not char.isalnum()) and char not in _STRUCTURAL_CHARS for char in cleaned
    )

    tld = _tld_of(host)
    host_ip = _is_ip_literal(host)
    nonstandard_port = int(port is not None and port not in (80, 443))

    features: dict[str, Any] = {
        "url_length": len(cleaned),
        "num_dots": cleaned.count("."),
        "num_hyphens": cleaned.count("-"),
        "has_https": int(cleaned.lower().startswith("https://")),
        "has_ip": int(host_ip),
        "host_length": len(host),
        "num_subdirs": len(segments),
        "num_params": len(query_pairs),
        "suspicious_words": suspicious_words,
        "special_char_count": special,
        "digits_count": digits,
        "entropy": round(shannon_entropy(cleaned), 6),
        "at_symbol": int("@" in cleaned),
        "double_slash_in_path": int("//" in path),
        "punycode": int("xn--" in lower_host),
        "host_is_ip": int(host_ip),
        "nonstandard_port": nonstandard_port,
        "suspicious_tld": int(tld in HIGH_RISK_TLDS),
        "tld_entropy": round(shannon_entropy(tld), 6),
        "path_entropy": round(shannon_entropy("/".join(segments)), 6),
        "query_entropy": round(shannon_entropy("&".join(query_pairs)), 6),
        "tld": tld if tld else "<none>",
    }
    return {name: features[name] for name in ALL_FEATURE_NAMES}


def extract_feature_frame(urls: list[str]) -> tuple["Any", list[str], list[str]]:
    """Vectorised convenience wrapper returning ``(frame, valid_urls, failures)``.

    Rows that fail validation are skipped rather than raising, so a single
    malformed URL can never abort a batch run. ``valid_urls`` is returned so
    callers can keep external labels aligned with the extracted rows.
    """
    import pandas as pd

    records: list[dict[str, Any]] = []
    valid: list[str] = []
    failures: list[str] = []
    for candidate in urls:
        try:
            records.append(extract_url_features(candidate))
            valid.append(candidate)
        except InvalidURLError:
            failures.append(candidate)
    frame = pd.DataFrame(records, columns=list(ALL_FEATURE_NAMES))
    return frame, valid, failures


def describe_indicators(url: str, features: dict[str, Any] | None = None) -> list[str]:
    """Return human-readable risk indicators for a URL.

    Purely rule based and transparent; this never overrides the model.
    """
    try:
        feats = features or extract_url_features(url)
    except InvalidURLError:
        return []

    indicators: list[str] = []
    if feats["has_https"]:
        indicators.append("Uses HTTPS (encrypted transport)")
    else:
        indicators.append("No HTTPS - connection is not encrypted")

    if feats["host_is_ip"]:
        indicators.append("Host is a raw IP address instead of a domain name")
    if feats["nonstandard_port"]:
        indicators.append("Uses a non-standard network port")
    if feats["punycode"]:
        indicators.append("Host uses punycode (xn--) encoding")
    if feats["at_symbol"]:
        indicators.append("Contains an '@' symbol, which can disguise the real host")
    if feats["num_hyphens"] >= 5:
        indicators.append(f"Host contains {feats['num_hyphens']} hyphens (keyword stuffing)")
    if feats["suspicious_words"]:
        indicators.append(
            f"Contains {feats['suspicious_words']} suspicious keyword(s) "
            "such as login/verify/account"
        )
    if feats["suspicious_tld"]:
        indicators.append(f"Top-level domain '{feats['tld']}' is heavily abused")
    if feats["url_length"] > 75:
        indicators.append(f"Unusually long URL ({feats['url_length']} characters)")
    if feats["url_length"] < 12:
        indicators.append(f"Very short URL ({feats['url_length']} characters)")
    if feats["num_subdirs"] >= 5:
        indicators.append(f"Deep path with {feats['num_subdirs']} segments")
    if feats["num_params"] >= 3:
        indicators.append(f"{feats['num_params']} query parameters (possible data exfiltration form)")
    if feats["entropy"] >= 4.5:
        indicators.append(f"High character entropy ({feats['entropy']:.2f}) - randomised hostname")
    if feats["special_char_count"] >= 5:
        indicators.append(f"{feats['special_char_count']} unusual/special characters")
    if feats["digits_count"] >= 20:
        indicators.append(f"High digit count ({feats['digits_count']}) - possible encoded payload")
    return indicators