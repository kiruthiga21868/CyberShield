"""CyberShield AI - AI-Powered Cybersecurity Assistant.

A Streamlit application providing five analyses behind one dashboard:

* URL phishing detection   - Random Forest + transparent structural rules
* Email spam detection     - TF-IDF + Multinomial Naive Bayes
* Password strength        - local-only rule analysis (never leaves the device)
* QR code safety           - OpenCV decoding with URL risk routing
* Security score           - rule-based rollup of everything analysed

Groq and VirusTotal are optional. When their API keys are absent the features
degrade to an offline knowledge base and a clearly-labelled "unavailable"
notice; the application never fails to start because a key is missing.

Run locally with::

    streamlit run app.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Allow `streamlit run app.py` to import the top-level project packages.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from assets import theme  # noqa: E402
from config import (  # noqa: E402
    APP_NAME,
    APP_TAGLINE,
    APP_VERSION,
    METRICS_PATH,
    PHISHING_DATA_CAVEAT,
)
from modules.email_detector import SpamDetector  # noqa: E402
from modules.password_analyzer import analyze_password  # noqa: E402
from modules.phishing_detector import PhishingDetector  # noqa: E402
from modules.qr_scanner import QRScanner  # noqa: E402
from modules.security_score import NOT_ANALYZED, build_report  # noqa: E402
from services.groq_service import GroqService  # noqa: E402
from services.virustotal_service import VirusTotalService, merge_virustotal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

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

SESSION_RESULTS = ("password_report", "url_result", "email_result", "qr_result")


# --------------------------------------------------------------------------
# Resource management - models are loaded once and cached across reruns
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading local models...")
def _load_detectors() -> tuple[PhishingDetector, SpamDetector]:
    """Load both model artifacts. Missing artifacts degrade, never crash."""
    phishing = PhishingDetector()
    spam = SpamDetector()
    phishing.load()
    spam.load()
    if not phishing.is_available():
        logger.warning("Phishing model unavailable: %s", phishing.load_error)
    if not spam.is_available():
        logger.warning("Spam model unavailable: %s", spam.load_error)
    return phishing, spam


@st.cache_resource
def _load_groq() -> GroqService:
    return GroqService()


@st.cache_resource
def _load_virustotal() -> VirusTotalService:
    return VirusTotalService()


def _scanner() -> QRScanner:
    phishing, _ = _load_detectors()
    return QRScanner(phishing_detector=phishing)


@st.cache_data
def _load_metrics() -> dict[str, Any]:
    """Read generated metrics, tolerating a missing or malformed file."""
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("metrics.json could not be read: %s", exc)
        return {}


def init_session() -> None:
    """Create the session keys the dashboard reads, without clobbering results."""
    for key in SESSION_RESULTS:
        st.session_state.setdefault(key, None)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("vt_result", None)


def reset_session() -> None:
    for key in SESSION_RESULTS:
        st.session_state[key] = None
    st.session_state["vt_result"] = None
    st.session_state["chat_history"] = []


# --------------------------------------------------------------------------
# Shared rendering helpers
# --------------------------------------------------------------------------
def show_disclaimer() -> None:
    theme.notice(
        "<b>No automated scanner can guarantee safety.</b> Every result below is "
        "one signal among several. Verify important destinations through a channel "
        "you already trust before acting.",
        "warn",
    )


def show_model_banner() -> None:
    """Warn once per page when a local model is missing."""
    phishing, spam = _load_detectors()
    if not phishing.is_available():
        theme.notice(
            f"<b>Phishing model unavailable.</b> {phishing.load_error or ''} "
            "URL results below use the structural rule layer only.",
            "bad",
        )
    if not spam.is_available():
        theme.notice(
            f"<b>Spam model unavailable.</b> {spam.load_error or ''} "
            "Only structural spam indicators will be shown.",
            "bad",
        )


def render_indicator_block(indicators: list[str], empty: str) -> None:
    theme.indicator_list(indicators, empty)


def render_password_report(report: Any) -> None:
    """Render a password analyzer report (no password text is ever shown)."""
    if report is None or not getattr(report, "is_valid_input", False):
        theme.notice(
            report.error if report and report.error
            else "Enter a password to see a local-only strength analysis.",
            "info",
        )
        return

    theme.result_panel("Password Strength", report.strength, {
        "Score": f"{report.score} / 100",
        "Estimated entropy": f"{report.entropy_bits:.0f} bits",
        "Length": f"{report.length} characters",
    })

    if report.checks:
        st.markdown("**Checks passed**")
        for item in report.checks:
            st.markdown(f"- {item}")
    if report.warnings:
        st.markdown("**Warnings**")
        for item in report.warnings:
            st.markdown(f"- {item}")
    if report.recommendations:
        st.markdown("**Recommendations**")
        for item in report.recommendations:
            st.markdown(f"- {item}")

    theme.notice(
        "Analysed entirely on this device. The password was not stored, logged "
        "or sent to any service.",
        "ok",
    )


def render_url_result(result: Any, show_virustotal: bool = True) -> None:
    """Render a phishing scan, separating the ML signal from the rule layer."""
    if result is None:
        return
    if result.error and not result.model_available:
        theme.notice(f"<b>{result.error}</b>", "warn")

    theme.result_panel("Final Risk Assessment", result.risk_level, {
        "Model prediction": result.prediction,
        "Model confidence": f"{result.confidence:.1%}",
        "Phishing probability (Random Forest)": f"{result.phishing_probability:.1%}",
        "Structural rule score": f"{result.rule_score:.2f}",
        "Why": result.risk_basis,
        "Model confidence note": result.confidence_note,
        "Recommendation": result.recommendation,
    })

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Model signal")
        st.progress(min(max(result.phishing_probability, 0.0), 1.0))
        st.markdown(
            f'<div class="cs-metric-label">Random Forest phishing probability: '
            f'{result.phishing_probability:.2%}</div>',
            unsafe_allow_html=True,
        )
        if not result.model_available:
            theme.notice("Model unavailable - this is the rule layer only.", "bad")
    with right:
        st.markdown("#### Rule layer")
        st.progress(min(max(result.rule_score, 0.0), 1.0))
        st.markdown(
            f'<div class="cs-metric-label">Structural rule score: '
            f'{result.rule_score:.2f}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Detected indicators")
    render_indicator_block(result.indicators, "No structural indicators were found.")

    if show_virustotal:
        _render_virustotal(result)


def _render_virustotal(result: Any) -> None:
    """Optional VirusTotal enrichment, clearly separated from the local model."""
    service = _load_virustotal()
    st.markdown("#### VirusTotal reputation (optional)")

    if not service.is_available():
        theme.notice(
            "VirusTotal is not configured. Add a VIRUSTOTAL_API_KEY environment "
            "variable to enrich URL results with third-party reputation. The "
            "local assessment above is unaffected.",
            "info",
        )
        return

    if st.button("Check VirusTotal reputation", key="vt_lookup"):
        with st.spinner("Contacting VirusTotal..."):
            vt_result = service.lookup_url(result.url)
        st.session_state["vt_result"] = vt_result
        merged, explanation = merge_virustotal(result.risk_level, vt_result)
        st.session_state["merged_risk"] = merged
        st.session_state["merged_explanation"] = explanation

    vt_result = st.session_state.get("vt_result")
    if vt_result is None:
        return
    if not vt_result.available:
        theme.notice(vt_result.message, "warn")
        return

    theme.result_panel("VirusTotal", vt_result.status, {
        "Verdict": vt_result.message,
        "Malicious engines": vt_result.malicious_engines,
        "Suspicious engines": vt_result.suspicious_engines,
        "Total engines": vt_result.total_engines,
        "Categories": ", ".join(f"{k}: {v}" for k, v in vt_result.categories.items()) or "none",
        "Cached": "yes" if vt_result.from_cache else "no",
    })
    merged = st.session_state.get("merged_risk")
    if merged:
        theme.result_panel("Combined Assessment", merged, {
            "Explanation": st.session_state.get("merged_explanation", "")
        })


def render_spam_result(result: Any) -> None:
    if result is None:
        return
    if result.error:
        theme.notice(f"<b>{result.error}</b>", "warn")
    theme.result_panel("Spam Assessment", result.prediction, {
        "Spam probability": f"{result.spam_probability:.1%}",
        "Model confidence": f"{result.confidence:.1%}",
        "Risk level": result.risk_level,
        "Confidence note": result.confidence_note,
        "Recommendation": result.recommendation,
    })
    st.markdown("#### Structural indicators")
    render_indicator_block(result.indicators, "No structural spam indicators found.")
    if result.urls:
        st.markdown("#### Links found in the message")
        for url in result.urls:
            st.code(url, language=None)


def render_qr_result(result: Any) -> None:
    if result is None:
        return
    if result.error:
        theme.notice(f"<b>{result.error}</b>", "warn")
    for warning in result.warnings:
        theme.notice(warning, "info")
    if not result.found:
        return

    for index, payload in enumerate(result.payloads, start=1):
        theme.result_panel(f"Payload {index}", payload.kind.upper(), {
            "Type": payload.description,
            "Content": payload.content,
        })
        if payload.analysis is not None:
            analysis = payload.analysis
            if not analysis.get("available"):
                theme.notice(analysis.get("note", "URL not analysed."), "info")
            else:
                theme.result_panel("URL Risk", analysis.get("risk_level", "UNKNOWN"), {
                    "Model prediction": analysis.get("prediction"),
                    "Confidence": f"{analysis.get('confidence', 0):.1%}",
                    "Why": analysis.get("risk_basis"),
                    "Recommendation": analysis.get("recommendation"),
                })
                st.markdown("**Indicators**")
                render_indicator_block(analysis.get("indicators", []), "None.")


def render_security_report(report: Any) -> None:
    if report is None:
        return
    theme.result_panel("Overall Security Score", report.band_label, {
        "Score": f"{report.total} / {report.max_total}",
        "Coverage": f"{report.analyzed_count} of {report.total_count} checks analysed",
        "Note": report.coverage_note,
    })
    st.progress(report.total / report.max_total if report.max_total else 0.0)

    st.markdown("#### Component breakdown")
    for component in report.components:
        if component.status == NOT_ANALYZED:
            label = (
                '<span class="cs-badge" style="background:#6e7681">'
                f"{NOT_ANALYZED}</span>"
            )
        else:
            label = theme.badge(component.status)
        st.markdown(f"**{component.label}** {label}", unsafe_allow_html=True)
        st.progress(component.percent / 100)
        detail = component.detail or "No analysis was run for this component."
        st.markdown(
            f'<div class="cs-metric-label">{component.points} / '
            f'{component.max_points} points &mdash; {detail}</div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_home() -> None:
    theme.header(APP_NAME, APP_TAGLINE)
    phishing, spam = _load_detectors()
    groq = _load_groq()
    virustotal = _load_virustotal()

    st.markdown(
        "CyberShield AI combines locally trained machine-learning models with "
        "transparent, rule-based checks to help you evaluate suspicious links, "
        "emails, passwords and QR codes. Everything except the two optional "
        "external services runs on your own machine."
    )
    show_disclaimer()

    theme.card("ML + rules, not ML alone",
              "URL risk combines a Random Forest probability with a visible "
              "rule layer. The model signal alone can never report HIGH risk, "
              "and both signals are always shown side by side.")

    columns = st.columns(4)
    columns[0].markdown(
        f'<div class="cs-metric-label">Phishing model</div>'
        f'<div style="font-size:1.1rem;font-weight:700">'
        f'{"Ready" if phishing.is_available() else "Unavailable"}</div>',
        unsafe_allow_html=True,
    )
    columns[1].markdown(
        f'<div class="cs-metric-label">Spam model</div>'
        f'<div style="font-size:1.1rem;font-weight:700">'
        f'{"Ready" if spam.is_available() else "Unavailable"}</div>',
        unsafe_allow_html=True,
    )
    columns[2].markdown(
        f'<div class="cs-metric-label">AI assistant</div>'
        f'<div style="font-size:1.1rem;font-weight:700">'
        f'{"Groq" if groq.is_available() else "Offline knowledge base"}</div>',
        unsafe_allow_html=True,
    )
    columns[3].markdown(
        f'<div class="cs-metric-label">VirusTotal</div>'
        f'<div style="font-size:1.1rem;font-weight:700">'
        f'{"Enabled" if virustotal.is_available() else "Not configured"}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("What you can do")
    st.markdown(
        "- **URL Scanner** - paste a link for a phishing probability, the "
        "structural rules that fired, and a combined risk level.\n"
        "- **Email Spam Detector** - classify a message and surface the "
        "links it contains.\n"
        "- **Password Analyzer** - score a password locally; it never leaves "
        "your device.\n"
        "- **QR Scanner** - decode a QR image and check any URL it contains "
        "before you visit it.\n"
        "- **Security Score** - roll up everything you have analysed into one "
        "transparent 0-100 score.\n"
        "- **AI Chatbot** - ask defensive cybersecurity questions."
    )

    st.subheader("Current session")
    report = build_report(
        st.session_state.get("password_report"),
        st.session_state.get("url_result"),
        st.session_state.get("email_result"),
        st.session_state.get("qr_result"),
    )
    theme.metric_row([
        ("Password", "Analysed" if st.session_state.get("password_report") else "Not analysed"),
        ("URL", "Analysed" if st.session_state.get("url_result") else "Not analysed"),
        ("Email", "Analysed" if st.session_state.get("email_result") else "Not analysed"),
        ("QR", "Analysed" if st.session_state.get("qr_result") else "Not analysed"),
    ])
    theme.result_panel("Session Security Score", report.band_label, {
        "Score": f"{report.total} / {report.max_total}",
        "Coverage": report.coverage_note,
    })


def page_url_scanner() -> None:
    theme.header("URL Scanner", "Check a link before you click it")
    show_disclaimer()
    show_model_banner()

    url = st.text_input("URL to analyse", placeholder="https://example.com/login")
    if st.button("Analyse URL", type="primary") and url.strip():
        phishing, _ = _load_detectors()
        st.session_state["url_result"] = phishing.predict(url.strip())
        st.session_state["vt_result"] = None

    result = st.session_state.get("url_result")
    if result is not None:
        render_url_result(result)

    st.markdown("---")
    theme.notice(PHISHING_DATA_CAVEAT, "warn")


def page_email_detector() -> None:
    theme.header("Email Spam Detector", "Classify a message and extract its links")
    show_disclaimer()
    show_model_banner()

    text = st.text_area(
        "Email or message body", height=220,
        placeholder="Paste the message body here...",
    )
    if st.button("Analyse message", type="primary") and text.strip():
        _, spam = _load_detectors()
        st.session_state["email_result"] = spam.predict(text)

    render_spam_result(st.session_state.get("email_result"))


def page_password_analyzer() -> None:
    theme.header("Password Analyzer", "Scored entirely on your device")
    theme.notice(
        "This analysis runs locally. The password is not stored, logged, or "
        "transmitted anywhere, and the app never sends it to the AI chatbot.",
        "ok",
    )

    password = st.text_input("Password to analyse", type="password")
    if st.button("Analyse password", type="primary") and password:
        st.session_state["password_report"] = analyze_password(password)

    render_password_report(st.session_state.get("password_report"))


def page_qr_scanner() -> None:
    theme.header("QR Scanner", "Decode a QR code and check its destination")
    theme.notice(
        "Decoding is done locally with OpenCV. The app never opens or visits a "
        "decoded URL - it only analyses the text.",
        "ok",
    )

    upload = st.file_uploader(
        "Upload a QR image", type=["png", "jpg", "jpeg", "webp", "bmp"],
    )
    if upload is not None and st.button("Scan QR code", type="primary"):
        with st.spinner("Decoding..."):
            st.session_state["qr_result"] = _scanner().scan(upload)

    result = st.session_state.get("qr_result")
    if result is not None:
        render_qr_result(result)


def page_security_score() -> None:
    theme.header("Security Score", "A transparent rollup of everything you analysed")
    theme.card("How this works",
              "Each of the four checks is worth up to 25 points. A check you "
              "have not run scores 0 and is labelled 'Not analyzed', so the "
              "score can never be inflated by missing data. Every point comes "
              "from a documented rule.")

    report = build_report(
        st.session_state.get("password_report"),
        st.session_state.get("url_result"),
        st.session_state.get("email_result"),
        st.session_state.get("qr_result"),
    )
    render_security_report(report)

    if st.button("Reset session results"):
        reset_session()
        st.rerun()


def page_chatbot() -> None:
    groq = _load_groq()
    theme.header("Security Assistant", groq.mode_label)
    theme.notice(
        "The assistant covers defensive cybersecurity education. It will not "
        "guarantee a file or site is safe, and it never asks for passwords, "
        "card numbers, one-time codes or API keys.",
        "info",
    )

    if not groq.is_available():
        theme.notice(
            "No GROQ_API_KEY is configured, so the assistant is answering from "
            "its built-in offline knowledge base. Add the key for live AI answers.",
            "warn",
        )
    else:
        theme.notice(
            "Groq live AI answers are enabled. If a request fails, the assistant "
            "falls back to its built-in offline knowledge base.",
            "ok",
        )

    for entry in st.session_state["chat_history"]:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    question = st.chat_input("Ask about phishing, passwords, spam or QR codes")
    if question:
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = groq.ask(
                    question,
                    history=[
                        e for e in st.session_state["chat_history"][:-1]
                    ],
                )
            st.markdown(reply.text)
        if reply.error:
            theme.notice(reply.error, "warn")
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": reply.text}
        )


def _metric_block(metrics: dict[str, Any], spam: bool = False) -> None:
    """Render headline metrics for one model block from ``metrics.json``."""
    scores = metrics.get("metrics") or {}
    if not scores:
        st.info(
            "Metrics are not available. Run the matching training script "
            "(`python training/train_phishing.py` or "
            "`python training/train_spam.py`) to generate them."
        )
        return

    positive_label = "spam" if spam else "1"
    positive = (scores.get("per_class") or {}).get(positive_label) or {}
    low, high = st.columns(2)
    with low:
        theme.metric_row([
            ("Accuracy", f"{scores.get('accuracy', 0):.4f}"),
            ("Macro F1", f"{scores.get('macro_f1', 0):.4f}"),
        ])
        st.markdown(
            f'<div class="cs-metric-label">Weighted F1 '
            f'{scores.get("weighted_f1", 0):.4f}</div>',
            unsafe_allow_html=True,
        )
    with high:
        name = "Spam" if spam else "Phishing"
        theme.metric_row([
            (f"{name} precision", f"{positive.get('precision', 0):.4f}"),
            (f"{name} recall", f"{positive.get('recall', 0):.4f}"),
            (f"{name} F1", f"{positive.get('f1', 0):.4f}"),
        ])

    matrix = scores.get("confusion_matrix")
    if matrix:
        with st.expander("Confusion matrix (rows = actual, columns = predicted)"):
            labels = scores.get("labels") or []
            st.markdown(
                "| | " + " | ".join(f"pred {label}" for label in labels) + " |",
            )
            st.markdown("|" + "---|" * (len(labels) + 1))
            for index, row in enumerate(matrix):
                actual = labels[index] if index < len(labels) else index
                st.markdown(
                    f"| actual {actual} | " + " | ".join(str(v) for v in row) + " |"
                )

    st.markdown(
        f'<div class="cs-metric-label">Training rows: '
        f'{metrics.get("n_train_rows", "?")} &nbsp;|&nbsp; '
        f'Evaluation rows: {metrics.get("n_test_rows", "?")} &nbsp;|&nbsp; '
        f'Trained: {str(metrics.get("trained_at", "unknown"))[:19]}'
        "</div>",
        unsafe_allow_html=True,
    )


def _limitations_block(metrics: dict[str, Any]) -> None:
    limitations = metrics.get("limitations") or []
    if not limitations:
        return
    st.markdown("**Recorded limitations**")
    for item in limitations:
        st.markdown(f"- {item}")


def page_about() -> None:
    theme.header("About & Model Metrics", "What this tool can and cannot do")
    metrics = _load_metrics()
    phishing, spam = _load_detectors()

    st.markdown(
        f"**{APP_NAME}** v{APP_VERSION} - {APP_TAGLINE}. Both ML models are "
        "trained offline and run locally; Groq and VirusTotal are optional and "
        "degrade gracefully when unconfigured."
    )

    theme.notice(PHISHING_DATA_CAVEAT, "warn")

    st.subheader("Phishing URL model")
    st.markdown(
        "- **Algorithm**: "
        f"{metrics.get('phishing', {}).get('algorithm', 'Random Forest')} "
        f"({metrics.get('phishing', {}).get('n_estimators', '?')} trees, "
        f"class_weight={metrics.get('phishing', {}).get('class_weight', 'balanced')})\n"
        "- **Features**: 22 URL features from "
        "`utils.url_features.extract_url_features` - the same function used at "
        "training time, so the saved model can never drift from the schema\n"
        "- **Risk model**: model probability aggregated with the structural "
        "rule layer; the ML signal alone is capped at MEDIUM risk"
    )
    _metric_block(metrics.get("phishing") or {}, spam=False)
    _limitations_block(metrics.get("phishing") or {})

    st.subheader("Email spam model")
    st.markdown(
        "- **Algorithm**: "
        f"{metrics.get('spam', {}).get('algorithm', 'Multinomial Naive Bayes')} "
        f"over {metrics.get('spam', {}).get('vectorizer', 'TF-IDF')} features "
        f"(vocabulary {metrics.get('spam', {}).get('vocabulary_size', '?')})\n"
        "- **Training data**: labelled messages with duplicates removed"
    )
    _metric_block(metrics.get("spam") or {}, spam=True)
    _limitations_block(metrics.get("spam") or {})

    st.subheader("Model status")
    theme.metric_row([
        ("Phishing model", "Ready" if phishing.is_available() else "Unavailable"),
        ("Spam model", "Ready" if spam.is_available() else "Unavailable"),
    ])
    if not phishing.is_available() or not spam.is_available():
        theme.notice(
            "At least one local model is not loaded, so some pages report "
            "structural indicators only. Retrain with "
            "`python training/train_phishing.py` and "
            "`python training/train_spam.py`.",
            "bad",
        )

    st.subheader("Privacy")
    st.markdown(
        "- Passwords are analysed locally and never transmitted or stored.\n"
        "- Phishing and spam inference runs entirely on your machine.\n"
        "- Only VirusTotal lookups (when enabled) send a URL to a third party.\n"
        "- API keys are read from environment variables, never stored in code."
    )

    st.subheader("Limitations")
    st.markdown(
        "- No automated scanner can guarantee that a link, file or message is "
        "safe; every result is advisory.\n"
        "- The phishing training data lacks legitimate URLs containing path "
        "segments, so the model may over-flag them. This is exactly why the "
        "rule layer and the MEDIUM cap exist.\n"
        "- Always verify important destinations through a channel you already "
        "trust, and report suspected phishing to your provider or IT team."
    )


PAGE_FUNCTIONS = {
    "Home": page_home,
    "URL Scanner": page_url_scanner,
    "Email Spam Detector": page_email_detector,
    "Password Analyzer": page_password_analyzer,
    "QR Scanner": page_qr_scanner,
    "Security Score": page_security_score,
    "AI Chatbot": page_chatbot,
    "About & Model Metrics": page_about,
}


def main() -> None:
    st.set_page_config(
        page_title=APP_NAME, layout="wide",
        initial_sidebar_state="expanded",
    )
    theme.inject_css()
    init_session()

    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption(APP_TAGLINE)
        choice = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.markdown("---")
        if st.button("Reset session", use_container_width=True):
            reset_session()
            st.rerun()

    PAGE_FUNCTIONS[choice]()


if __name__ == "__main__":
    main()