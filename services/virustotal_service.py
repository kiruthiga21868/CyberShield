"""Optional VirusTotal threat-intelligence enrichment.

Design rules
------------
* Entirely optional. Without ``VIRUSTOTAL_API_KEY`` every call returns an
  ``available=False`` result carrying the standard "currently unavailable"
  message. The caller keeps working.
* Never replaces the local model. VirusTotal is one signal that is *merged*
  with the local assessment by the caller.
* Never raises. Network errors, rate limits, malformed payloads and auth
  failures are all converted into structured results.
* The API key is read from the environment, never logged and never included
  in any returned value. Requests only ever carry the user-supplied URL.
* Results are cached briefly so repeated scans of the same URL do not consume
  the free-tier quota.
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from config import (
    VIRUSTOTAL_CACHE_MAX_ENTRIES,
    VIRUSTOTAL_CACHE_TTL_SECONDS,
    VIRUSTOTAL_ENDPOINT,
    VIRUSTOTAL_MIN_INTERVAL_SECONDS,
    VIRUSTOTAL_TIMEOUT,
    VIRUSTOTAL_UNAVAILABLE_MESSAGE,
    get_virustotal_api_key,
)
from utils.url_features import InvalidURLError, validate_and_normalize_url

logger = logging.getLogger(__name__)

_CACHE: dict[str, tuple[float, "VirusTotalResult"]] = {}
_CACHE_LOCK = threading.Lock()
_LAST_CALL = 0.0
_RATE_LOCK = threading.Lock()

# Reputation bands derived from the malicious / total engine counts.
# Both an absolute flag count and a ratio are used: a ratio alone misses a
# heavily-flagged site on a large engine set, and a count alone over-reacts
# on a small one.
MALICIOUS_MIN_FLAGS = 5
MALICIOUS_MIN_RATIO = 0.10
SUSPICIOUS_MIN_FLAGS = 1
SUSPICIOUS_MIN_RATIO = 0.01
LOW_RISK_MIN_SUSPICIOUS = 3


@dataclass
class VirusTotalResult:
    """Structured outcome of a VirusTotal lookup."""

    available: bool
    status: str  # MALICIOUS | SUSPICIOUS | LOW_RISK | CLEAN | UNKNOWN | UNAVAILABLE
    message: str
    malicious_engines: int = 0
    suspicious_engines: int = 0
    total_engines: int = 0
    reputation: float = 0.0
    categories: dict[str, str] = field(default_factory=dict)
    from_cache: bool = False
    error: str | None = None

    @property
    def risk_level(self) -> str:
        """Map the VirusTotal verdict onto the app's risk vocabulary."""
        return {
            "MALICIOUS": "HIGH",
            "SUSPICIOUS": "MEDIUM",
            "LOW_RISK": "LOW",
            "CLEAN": "LOW",
        }.get(self.status, "UNKNOWN")

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "message": self.message,
            "malicious_engines": self.malicious_engines,
            "suspicious_engines": self.suspicious_engines,
            "total_engines": self.total_engines,
            "reputation": round(self.reputation, 4),
            "categories": self.categories,
            "from_cache": self.from_cache,
            "risk_level": self.risk_level,
            "error": self.error,
        }


def encode_url_id(url: str) -> str:
    """VirusTotal expects a base64url-encoded URL id without padding."""
    encoded = base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=")


def _classify(malicious: int, total: int) -> tuple[str, float]:
    """Assign a reputation band from engine counts.

    ``total`` is the denominator used for the ratio; callers pass at least 1.
    """
    ratio = malicious / total if total else 0.0
    if malicious >= MALICIOUS_MIN_FLAGS or ratio >= MALICIOUS_MIN_RATIO:
        return "MALICIOUS", ratio
    if malicious >= SUSPICIOUS_MIN_FLAGS or ratio >= SUSPICIOUS_MIN_RATIO:
        return "SUSPICIOUS", ratio
    return "CLEAN", ratio


def _cache_get(key: str) -> VirusTotalResult | None:
    with _CACHE_LOCK:
        entry = _CACHE.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.time() - stored_at > VIRUSTOTAL_CACHE_TTL_SECONDS:
            _CACHE.pop(key, None)
            return None
        return VirusTotalResult(**{**value.__dict__, "from_cache": True})


def _cache_put(key: str, value: VirusTotalResult) -> None:
    with _CACHE_LOCK:
        if len(_CACHE) >= VIRUSTOTAL_CACHE_MAX_ENTRIES:
            oldest = min(_CACHE, key=lambda k: _CACHE[k][0])
            _CACHE.pop(oldest, None)
        _CACHE[key] = (time.time(), value)


def clear_cache() -> None:
    """Empty the response cache.

    Exposed mainly so tests can run in isolation; the application relies on the
    cache persisting for the lifetime of the process.
    """
    with _CACHE_LOCK:
        _CACHE.clear()


def _throttle() -> None:
    """Block until the minimum interval between calls has elapsed."""
    global _LAST_CALL
    with _RATE_LOCK:
        wait = VIRUSTOTAL_MIN_INTERVAL_SECONDS - (time.time() - _LAST_CALL)
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL = time.time()


class VirusTotalService:
    """Looks up URL reputation, degrading gracefully when unconfigured."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self.session = requests.Session()

    @property
    def api_key(self) -> str | None:
        return self._api_key if self._api_key is not None else get_virustotal_api_key()

    def is_available(self) -> bool:
        """True only when a key is configured. No network call is made."""
        return bool(self.api_key)

    def _unavailable(self, reason: str = VIRUSTOTAL_UNAVAILABLE_MESSAGE) -> VirusTotalResult:
        return VirusTotalResult(
            available=False, status="UNAVAILABLE", message=reason, error=reason
        )

    def lookup_url(self, url: str) -> VirusTotalResult:
        """Return the VirusTotal reputation for ``url``. Never raises."""
        key = self.api_key
        if not key:
            logger.info("VirusTotal lookup skipped: no API key configured.")
            return self._unavailable()

        try:
            normalized = validate_and_normalize_url(url)
        except InvalidURLError as exc:
            return self._unavailable(f"Invalid URL for VirusTotal lookup: {exc}")

        cache_key = f"{key[:6]}::{normalized}"
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

        headers = {"x-apikey": key, "accept": "application/json"}
        params = {"url": normalized}
        url_id = encode_url_id(normalized)

        _throttle()
        try:
            # Report by URL id first (single GET); fall back to the POST
            # endpoint because a first-time submission returns 404.
            response = self.session.get(
                f"{VIRUSTOTAL_ENDPOINT}/{url_id}", headers=headers,
                timeout=VIRUSTOTAL_TIMEOUT,
            )
            if response.status_code == 404:
                response = self.session.post(
                    VIRUSTOTAL_ENDPOINT, headers=headers, params=params,
                    timeout=VIRUSTOTAL_TIMEOUT,
                )
        except requests.Timeout:
            logger.warning("VirusTotal request timed out after %ss.",
                           VIRUSTOTAL_TIMEOUT)
            return self._unavailable("VirusTotal request timed out.")
        except requests.RequestException as exc:
            # The message is logged but the raw exception is not echoed to the
            # UI, keeping internal network details away from the user.
            logger.warning("VirusTotal request failed: %s", type(exc).__name__)
            return self._unavailable("VirusTotal could not be reached.")

        if response.status_code == 429:
            logger.warning("VirusTotal rate limit reached.")
            return self._unavailable(
                "VirusTotal rate limit reached. Try again shortly."
            )
        if response.status_code in (401, 403):
            logger.warning("VirusTotal rejected the API key (HTTP %s).",
                           response.status_code)
            return self._unavailable("VirusTotal rejected the configured API key.")
        if response.status_code != 200:
            logger.warning("VirusTotal returned unexpected HTTP %s.",
                           response.status_code)
            return self._unavailable(
                f"VirusTotal returned an unexpected status ({response.status_code})."
            )

        try:
            payload = response.json()
            result = self._parse(payload)
        except Exception as exc:  # noqa: BLE001 - malformed response
            logger.warning("VirusTotal response could not be parsed: %s",
                           type(exc).__name__)
            return self._unavailable("VirusTotal returned an unreadable response.")

        _cache_put(cache_key, result)
        return result

    def _parse(self, payload: dict[str, Any]) -> VirusTotalResult:
        """Extract only the fields the UI needs from a VirusTotal response."""
        attributes = (payload.get("data") or {}).get("attributes") or {}
        stats = attributes.get("last_analysis_stats") or {}
        malicious = int(stats.get("malicious", 0) or 0)
        suspicious = int(stats.get("suspicious", 0) or 0)
        # Every numeric key in last_analysis_stats is an engine bucket
        # (malicious, suspicious, harmless, undetected, timeout), so the total
        # engine count is their sum. Omitting 'undetected' would understate the
        # denominator and inflate the malicious ratio.
        total = sum(
            int(value or 0)
            for value in stats.values()
            if isinstance(value, (int, float))
        )
        status, ratio = _classify(malicious, max(total, 1))

        categories: dict[str, str] = {}
        raw_categories = attributes.get("categories") or {}
        if isinstance(raw_categories, dict):
            categories = {
                str(key): str(value)
                for key, value in list(raw_categories.items())[:12]
                if isinstance(value, str)
            }

        if status == "CLEAN" and suspicious >= LOW_RISK_MIN_SUSPICIOUS:
            status = "LOW_RISK"

        messages = {
            "MALICIOUS": f"{malicious} of {total} security vendors flagged this URL.",
            "SUSPICIOUS": f"{malicious} of {total} security vendors flagged this URL.",
            "LOW_RISK": f"No vendor flagged this URL outright, but {suspicious} "
                        f"raised a lower-confidence concern out of {total}.",
            "CLEAN": f"No security vendor flagged this URL ({total} engines scanned).",
        }
        return VirusTotalResult(
            available=True,
            status=status,
            message=messages[status],
            malicious_engines=malicious,
            suspicious_engines=suspicious,
            total_engines=total,
            reputation=ratio,
            categories=categories,
        )


def merge_virustotal(
    local_risk: str, vt_result: VirusTotalResult
) -> tuple[str, str]:
    """Combine a local risk level with VirusTotal reputation.

    The more severe of the two verdicts wins, which keeps a clean local result
    from masking a known-bad URL. Returns ``(risk_level, explanation)``.
    """
    if not vt_result.available:
        return local_risk, (
            f"{vt_result.message} The local Random Forest assessment "
            f"({local_risk}) was used unchanged."
        )

    vt_risk = vt_result.risk_level
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "UNKNOWN": 0}
    merged = (
        local_risk
        if order.get(local_risk, 0) >= order.get(vt_risk, 0)
        else vt_risk
    )
    explanation = (
        f"Local model: {local_risk}. VirusTotal: {vt_result.status} "
        f"({vt_result.malicious_engines}/{vt_result.total_engines} engines). "
        f"Combined assessment: {merged}."
    )
    return merged, explanation


__all__ = [
    "VirusTotalService",
    "VirusTotalResult",
    "encode_url_id",
    "merge_virustotal",
]