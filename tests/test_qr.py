"""Tests for the QR code scanner.

Synthetic QR images are generated with the ``qrcode`` package so the suite is
self-contained and needs no binary fixtures in the repository.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qrcode", reason="the qrcode package generates test images")

from modules.qr_scanner import QRScanner, classify_payload, make_qr_image_for_tests

LEGIT_URL = "https://www.microsoft.com/en-us"
PHISHING_URL = "http://paypal-secure-login.tk/verify?account=99"


@pytest.fixture(scope="module")
def scanner():
    return QRScanner()


# -------------------------------------------------------------- classifier
@pytest.mark.parametrize(
    "payload,kind",
    [
        ("https://example.com", "url"),
        ("http://example.com", "url"),
        ("www.example.com", "url"),
        ("WIFI:T:WPA;S:Net;P:pw;;", "wifi"),
        ("tel:+15551234567", "tel"),
        ("mailto:a@b.com", "email"),
        ("geo:51.5,-0.12", "geo"),
        ("BEGIN:VCARD\nVERSION:3.0", "vcard"),
        ("Just some plain text", "text"),
        ("", "text"),
    ],
)
def test_classify_payload(payload, kind):
    assert classify_payload(payload)[0] == kind


# ------------------------------------------------------------------ decode
def test_url_qr_is_decoded_and_analysed(scanner):
    result = scanner.scan(make_qr_image_for_tests(LEGIT_URL))
    assert result.found is True
    assert result.payloads[0].kind == "url"
    assert result.payloads[0].content == LEGIT_URL
    assert result.urls == [LEGIT_URL]


def test_plain_text_qr_is_decoded_without_url_analysis(scanner):
    result = scanner.scan(make_qr_image_for_tests("Meeting at 3pm, room 4"))
    assert result.found is True
    assert result.payloads[0].kind == "text"
    assert result.payloads[0].analysis is None
    assert result.urls == []


def test_wifi_qr_is_classified(scanner):
    result = scanner.scan(make_qr_image_for_tests("WIFI:T:WPA;S:Net;P:pw;;"))
    assert result.payloads[0].kind == "wifi"


def test_url_analysis_can_be_disabled(scanner):
    result = scanner.scan(make_qr_image_for_tests(LEGIT_URL), analyze_urls=False)
    assert result.payloads[0].kind == "url"
    assert result.payloads[0].analysis is None


def test_phishing_qr_is_high_risk(scanner, phishing_detector):
    scanner_with_detector = QRScanner(phishing_detector=phishing_detector)
    result = scanner_with_detector.scan(make_qr_image_for_tests(PHISHING_URL))
    analysis = result.payloads[0].analysis
    assert analysis["risk_level"] == "HIGH"
    assert analysis["prediction"] == "PHISHING"


def test_legitimate_qr_url_is_not_high_risk(scanner, phishing_detector):
    scanner_with_detector = QRScanner(phishing_detector=phishing_detector)
    result = scanner_with_detector.scan(make_qr_image_for_tests(LEGIT_URL))
    assert result.payloads[0].analysis["risk_level"] != "HIGH"


# --------------------------------------------------------------- failures
def test_corrupt_file_is_reported(scanner):
    result = scanner.scan(b"definitely not an image")
    assert result.found is False
    assert "not a readable image" in result.error


def test_empty_upload_is_reported(scanner):
    result = scanner.scan(b"")
    assert result.found is False
    assert "empty" in result.error.lower()


def test_truncated_image_is_reported(scanner):
    data = make_qr_image_for_tests(LEGIT_URL)
    result = scanner.scan(data[: len(data) // 3])
    assert result.found is False
    assert result.error


def test_image_without_qr_is_reported(scanner):
    blank = np.zeros((240, 240, 3), dtype=np.uint8)
    result = scanner.scan(blank)
    assert result.found is False
    assert "No QR code" in result.error
    assert result.warnings


def test_file_path_input_is_refused(scanner, tmp_path):
    path = tmp_path / "qr.png"
    path.write_bytes(make_qr_image_for_tests(LEGIT_URL))
    result = scanner.scan(str(path))
    assert result.found is False
    assert "Upload the image" in result.error


def test_grayscale_and_rgba_images_are_handled(scanner):
    """Different channel layouts must not crash the decoder."""
    image = make_qr_image_for_tests(LEGIT_URL)
    from PIL import Image
    import io

    rgba = Image.open(io.BytesIO(image)).convert("RGBA")
    buffer = io.BytesIO()
    rgba.save(buffer, format="PNG")
    result = scanner.scan(buffer.getvalue())
    assert result.found is True
    assert result.payloads[0].content == LEGIT_URL


def test_scanner_never_returns_an_opened_url(scanner):
    """The decoded URL must be data, never a fetched resource."""
    result = scanner.scan(make_qr_image_for_tests(LEGIT_URL))
    assert isinstance(result.payloads[0].content, str)
    assert result.payloads[0].content.startswith("https://")


def test_result_is_serialisable(scanner):
    payload = scanner.scan(make_qr_image_for_tests("plain text")).to_dict()
    assert payload["found"] is True
    assert "payloads" in payload and "urls" in payload