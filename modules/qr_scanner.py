"""QR code scanning and safety analysis.

Uses OpenCV's built-in ``QRCodeDetector``, which ships inside the
``opencv-python-headless`` wheel. This avoids ``pyzbar``, which needs a
separate native library (the zbar DLL) and is a frequent source of install
failures on Windows and on Render.

The scanner never opens or requests a decoded URL. It only returns text for
the caller to inspect, and optionally routes URL payloads through the phishing
detector.
"""
from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

_URL_SCHEME = re.compile(r"(?i)^(https?|ftp)://\S+$")
_WWW = re.compile(r"(?i)^www\.\S+$")

# Non-URL QR payload types worth naming explicitly for the user.
# Each entry is (payload prefix, payload kind, human description).
_QR_PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("WIFI:", "wifi", "Wi-Fi network configuration"),
    ("tel:", "tel", "Telephone number"),
    ("mailto:", "email", "Email address"),
    ("smsto:", "sms", "SMS message"),
    ("geo:", "geo", "Geographic location"),
    ("begin:vcard", "vcard", "Contact card (vCard)"),
    ("otpauth://", "otp", "One-time password (2FA) setup"),
)


@dataclass
class QRPayload:
    """A single decoded QR payload and its interpretation."""

    content: str
    kind: str  # "url" | "text" | "wifi" | "tel" | "email" | "geo" | "vcard" | "otp"
    description: str
    analysis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "kind": self.kind,
            "description": self.description,
            "analysis": self.analysis,
        }


@dataclass
class QRResult:
    """Outcome of a QR scanning attempt."""

    found: bool
    payloads: list[QRPayload] = field(default_factory=list)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def urls(self) -> list[str]:
        return [p.content for p in self.payloads if p.kind == "url"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "found": self.found,
            "payloads": [p.to_dict() for p in self.payloads],
            "urls": self.urls,
            "error": self.error,
            "warnings": self.warnings,
        }


def classify_payload(content: str) -> tuple[str, str]:
    """Decide whether a decoded payload is a URL and describe it."""
    candidate = (content or "").strip()
    if not candidate:
        return "text", "Empty payload"

    if _URL_SCHEME.match(candidate) or _WWW.match(candidate):
        return "url", "Web address"

    for prefix, kind, description in _QR_PREFIXES:
        if candidate.upper().startswith(prefix.upper()):
            return kind, description

    if "@" in candidate and "." in candidate and " " not in candidate:
        return "email", "Email address"

    if re.fullmatch(r"[+]?[\d\s().-]{6,}", candidate):
        return "tel", "Telephone number"

    return "text", "Plain text"


def _to_bgr(source: Any) -> np.ndarray:
    """Decode uploaded bytes or an existing image object into a BGR array."""
    import cv2

    if isinstance(source, np.ndarray):
        array = source
    else:
        if hasattr(source, "read"):
            source.seek(0)
            raw = source.read()
        elif isinstance(source, (bytes, bytearray)):
            raw = bytes(source)
        elif isinstance(source, str):
            raise ValueError(
                "A file path cannot be scanned safely. Upload the image instead."
            )
        else:
            raw = source

        if not isinstance(raw, (bytes, bytearray)):
            raise ValueError(f"Unsupported image input type: {type(source).__name__}")
        if not raw:
            raise ValueError("The uploaded file is empty.")
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"The image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
            )

        try:
            buffer = np.frombuffer(raw, dtype=np.uint8)
            array = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"The image could not be decoded: {exc}") from exc
        if array is None:
            raise ValueError(
                "The uploaded file is not a readable image "
                "(PNG, JPEG, WEBP and BMP are supported)."
            )

    if array.size == 0:
        raise ValueError("The image contains no pixel data.")
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

    if array.shape[0] * array.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("The image has too many pixels to process safely.")

    return np.ascontiguousarray(array)


def _decode_all(array: np.ndarray) -> list[str]:
    """Run single and multi-code detection, tolerating API version differences."""
    import cv2

    detector = cv2.QRCodeDetector()
    found: list[str] = []

    try:
        data, _, _ = detector.detectAndDecode(array)
        if data:
            found.append(data)
    except Exception as exc:  # noqa: BLE001 - version differences
        logger.debug("detectAndDecode failed: %s", exc)

    try:
        result = detector.detectAndDecodeMulti(array)
        if isinstance(result, tuple) and result:
            ok = result[0]
            decoded = result[1] if len(result) > 1 else ()
            if ok and decoded:
                if isinstance(decoded, (tuple, list)):
                    found.extend(str(item) for item in decoded if item)
                else:
                    found.append(str(decoded))
    except Exception as exc:  # noqa: BLE001 - version differences
        logger.debug("detectAndDecodeMulti failed: %s", exc)

    # Preserve order while removing duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


class QRScanner:
    """Decodes QR codes from an uploaded image and classifies the payload."""

    def __init__(self, phishing_detector: Any | None = None) -> None:
        self.phishing_detector = phishing_detector

    def scan(self, source: Any, analyze_urls: bool = True) -> QRResult:
        """Decode an image and, optionally, analyse any URL payload."""
        try:
            array = _to_bgr(source)
        except ValueError as exc:
            return QRResult(found=False, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - unexpected decoding failure
            logger.exception("QR image decoding failed")
            return QRResult(found=False, error=f"Unexpected decoding error: {exc}")

        try:
            decoded = _decode_all(array)
        except Exception as exc:  # noqa: BLE001
            logger.exception("QR detection failed")
            return QRResult(found=False, error=f"QR detection error: {exc}")

        if not decoded:
            return QRResult(
                found=False,
                error="No QR code was detected in this image.",
                warnings=[
                    "The image may be blurry, cropped, too small, or the QR code "
                    "may be low-contrast against its background."
                ],
            )

        warnings: list[str] = []
        if len(decoded) > 1:
            warnings.append(f"{len(decoded)} QR codes were found; all were decoded.")

        payloads: list[QRPayload] = []
        for content in decoded:
            kind, description = classify_payload(content)
            payload = QRPayload(content=content, kind=kind, description=description)
            if kind == "url" and analyze_urls:
                payload.analysis = self._analyze_url(content)
            payloads.append(payload)

        return QRResult(found=True, payloads=payloads, warnings=warnings)

    def _analyze_url(self, url: str) -> dict[str, Any]:
        """Run the phishing detector over a decoded URL payload."""
        if self.phishing_detector is None:
            return {
                "available": False,
                "note": "The phishing detector is not configured for this session.",
            }
        try:
            result = self.phishing_detector.predict(url)
            return {
                "available": True,
                "prediction": result.prediction,
                "confidence": round(result.confidence, 4),
                "phishing_probability": round(result.phishing_probability, 4),
                "risk_level": result.risk_level,
                "indicators": result.indicators,
                "recommendation": result.recommendation,
                "risk_basis": result.risk_basis,
                "error": result.error,
            }
        except Exception as exc:  # noqa: BLE001 - never break the QR flow
            logger.exception("QR URL analysis failed")
            return {
                "available": False,
                "note": f"The URL could not be analysed: {exc}",
            }


def make_qr_image_for_tests(content: str, scale: int = 8) -> bytes:
    """Render a QR code PNG in memory. Test helper; never used at runtime."""
    import qrcode

    image = qrcode.make(content).convert("RGB")
    image = image.resize(
        (image.width * scale, image.height * scale), 0
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


__all__ = [
    "QRScanner",
    "QRResult",
    "QRPayload",
    "classify_payload",
    "make_qr_image_for_tests",
]