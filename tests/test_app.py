"""End-to-end UI smoke tests for ``app.py`` using Streamlit's AppTest.

These render every page and drive the real widgets, so a template error or a
broken helper is caught here rather than by a user opening the app.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

# AppTest resolves relative paths against the calling file, so build an
# absolute path from the project root.
APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = [
    "Home",
    "URL Scanner",
    "Email Spam Detector",
    "Password Analyzer",
    "QR Scanner",
    "Security Score",
    "AI Chatbot",
    "About & Model Metrics",
]


def _new_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=180)


def _run(test) -> None:
    test.run()
    assert not test.exception, f"Unhandled exception: {test.exception}"


def _body(test) -> str:
    return "\n".join(m.value for m in test.markdown)


def test_app_loads_home():
    at = _new_app()
    _run(at)
    # set_page_config is not exposed as a title element, so assert on content.
    assert "CyberShield AI" in _body(at)
    assert at.radio[0].options == PAGES


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = _new_app()
    _run(at)
    at.sidebar.radio[0].set_value(page)
    _run(at)
    assert len(_body(at)) > 200, f"{page} rendered almost nothing"


def test_url_scanner_reports_risk():
    at = _new_app()
    _run(at)
    at.sidebar.radio[0].set_value("URL Scanner")
    _run(at)

    at.text_input[0].set_value("http://paypal-secure-login.tk/verify/account")
    at.button[0].click().run()
    assert not at.exception, at.exception

    body = _body(at)
    assert "HIGH" in body
    assert "Final Risk Assessment" in body
    # The model signal and the rule layer must both be shown, never just one.
    assert "Model signal" in body
    assert "Rule layer" in body
    # The training-data caveat must accompany every URL scan.
    assert "820 bare apex domains" in body


def test_password_page_scores_input():
    at = _new_app()
    _run(at)
    at.sidebar.radio[0].set_value("Password Analyzer")
    _run(at)

    at.text_input[0].set_value("Tr0ub4dor&3-c0rrect-horse")
    at.button[0].click().run()
    assert not at.exception, at.exception

    body = _body(at)
    assert "Password Strength" in body
    assert "not stored, logged or sent to any service" in body


def test_email_page_classifies_text():
    at = _new_app()
    _run(at)
    at.sidebar.radio[0].set_value("Email Spam Detector")
    _run(at)

    at.text_area[0].set_value(
        "URGENT! You have won a free prize. Claim your reward now at "
        "http://winner-prizes.example/claim immediately!!!"
    )
    at.button[0].click().run()
    assert not at.exception, at.exception

    assert "Spam Assessment" in _body(at)


def test_security_score_starts_unanalysed():
    at = _new_app()
    _run(at)
    at.sidebar.radio[0].set_value("Security Score")
    _run(at)

    body = _body(at)
    assert "Not analyzed" in body
    assert "0 of 4 checks analysed" in body


def test_about_page_shows_real_metrics():
    at = _new_app()
    _run(at)
    at.sidebar.radio[0].set_value("About & Model Metrics")
    _run(at)

    body = _body(at)
    # Headline metrics are read from models/metrics.json and formatted to 4dp.
    assert "0.9227" in body
    assert "0.9576" in body
    assert "Training-data limitation" in body


def test_chatbot_answers_offline_without_api_key():
    at = _new_app()
    _run(at)
    at.sidebar.radio[0].set_value("AI Chatbot")
    _run(at)
    # With no key the assistant must say so instead of failing silently.
    assert "GROQ_API_KEY" in _body(at) or "offline knowledge base" in _body(at)


def test_reset_button_clears_results():
    at = _new_app()
    _run(at)
    at.sidebar.radio[0].set_value("URL Scanner")
    _run(at)
    at.text_input[0].set_value("http://secure-login-update.xyz/verify")
    at.button[0].click().run()
    assert "Final Risk Assessment" in _body(at)

    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert "Final Risk Assessment" not in _body(at)