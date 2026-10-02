"""Streamlit presentation helpers: CSS theme, badges, cards and banners.

Kept separate from ``app.py`` so the page code stays readable and the styling
can be adjusted without touching the analysis logic.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

# Dark security-dashboard palette.
_BACKGROUND = "#0d1117"
_SURFACE = "#161b22"
_SURFACE_ALT = "#1c2333"
_BORDER = "#30363d"
_TEXT = "#e6edf3"
_MUTED = "#8b949e"
_ACCENT = "#2f81f7"

RISK_COLOURS: dict[str, str] = {
    "HIGH": "#f85149",
    "MEDIUM": "#d29922",
    "LOW": "#3fb950",
    "UNKNOWN": "#6e7681",
    "CLEAN": "#3fb950",
    "SUSPICIOUS": "#d29922",
    "MALICIOUS": "#f85149",
    "GOOD": "#3fb950",
    "MODERATE": "#d29922",
    "NEEDS_ATTENTION": "#f85149",
    "STRONG": "#3fb950",
    "MEDIUM_STRENGTH": "#d29922",
    "WEAK": "#f85149",
}

PASSWORD_COLOURS = {
    "STRONG": RISK_COLOURS["STRONG"],
    "MEDIUM": RISK_COLOURS["MEDIUM"],
    "WEAK": RISK_COLOURS["WEAK"],
    "UNKNOWN": RISK_COLOURS["UNKNOWN"],
}

_CSS = f"""
<style>
  .stApp {{ background: {_BACKGROUND}; color: {_TEXT}; }}
  html, body, [class*="css"] {{ font-family: "Segoe UI", Roboto, sans-serif; }}

  .cs-header {{
      padding: 1.4rem 1.6rem; border-radius: 12px; margin-bottom: 1rem;
      background: linear-gradient(120deg, #12263f 0%, #1a2a44 55%, #16233a 100%);
      border: 1px solid {_BORDER};
  }}
  .cs-header h1 {{ margin: 0; font-size: 2.1rem; letter-spacing: .5px; }}
  .cs-header p {{ margin: .3rem 0 0; color: {_MUTED}; font-size: 1.02rem; }}

  .cs-card {{
      background: {_SURFACE}; border: 1px solid {_BORDER}; border-radius: 10px;
      padding: 1rem 1.1rem; margin-bottom: .8rem;
  }}
  .cs-card h4 {{ margin: 0 0 .35rem; font-size: 1.02rem; }}
  .cs-card p {{ margin: 0; color: {_MUTED}; font-size: .87rem; line-height: 1.45; }}

  .cs-badge {{
      display: inline-block; padding: .2rem .7rem; border-radius: 999px;
      font-weight: 700; font-size: .78rem; letter-spacing: .6px;
      color: #0d1117; text-transform: uppercase;
  }}
  .cs-metric-label {{ color: {_MUTED}; font-size: .78rem; text-transform: uppercase;
                      letter-spacing: .6px; }}

  .cs-ok    {{ background: rgba(63,185,80,.12); border-left: 4px solid {RISK_COLOURS['LOW']}; }}
  .cs-warn  {{ background: rgba(210,153,34,.12); border-left: 4px solid {RISK_COLOURS['MEDIUM']}; }}
  .cs-bad   {{ background: rgba(248,81,73,.12); border-left: 4px solid {RISK_COLOURS['HIGH']}; }}
  .cs-info  {{ background: rgba(47,129,247,.10); border-left: 4px solid {_ACCENT}; }}
  .cs-panel {{ background: {_SURFACE}; border: 1px solid {_BORDER};
               border-radius: 10px; padding: .8rem 1rem; }}

  div[data-testid="stMetricValue"] {{ color: {_TEXT}; }}
  .stButton>button {{
      border-radius: 8px; border: 1px solid {_ACCENT}; background: {_ACCENT};
      color: #fff; font-weight: 600;
  }}
  .stButton>button:hover {{ background: #1f6feb; border-color: #1f6feb; }}
  hr {{ border-color: {_BORDER}; }}
</style>
"""


def inject_css() -> None:
    """Apply the dark dashboard theme. Safe to call on every rerun."""
    st.markdown(_CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str = "") -> None:
    """Render the application banner."""
    st.markdown(
        f'<div class="cs-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    """Render a small informational card."""
    st.markdown(
        f'<div class="cs-card"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def badge(label: str) -> str:
    """HTML badge coloured by the risk/strength vocabulary."""
    colour = (
        RISK_COLOURS.get(label.upper())
        or PASSWORD_COLOURS.get(label.upper())
        or _MUTED
    )
    return (
        f'<span class="cs-badge" style="background:{colour}">{label.upper()}</span>'
    )


def notice(message: str, kind: str = "info") -> None:
    """Coloured callout: 'ok', 'warn', 'bad' or 'info'."""
    st.markdown(
        f'<div class="cs-{kind} cs-panel">{message}</div>',
        unsafe_allow_html=True,
    )


def metric_row(items: list[tuple[str, str]]) -> None:
    """Render a compact row of label/value pairs."""
    columns = st.columns(len(items))
    for column, (label, value) in zip(columns, items):
        with column:
            st.markdown(
                f'<div class="cs-metric-label">{label}</div>'
                f'<div style="font-size:1.35rem;font-weight:700">{value}</div>',
                unsafe_allow_html=True,
            )


def result_panel(title: str, badge_text: str, rows: dict[str, Any]) -> None:
    """Render a titled panel with a badge and label/value rows."""
    st.markdown(f"### {title} {badge(badge_text)}", unsafe_allow_html=True)
    if not rows:
        return
    st.markdown(
        '<div class="cs-panel">' + "".join(
            f'<div style="margin:.15rem 0"><b>{key}:</b> {value}</div>'
            for key, value in rows.items()
            if value not in (None, "", [])
        ) + "</div>",
        unsafe_allow_html=True,
    )


def indicator_list(indicators: list[str], empty: str = "No indicators detected.") -> None:
    """Render detected indicators as a bulleted list."""
    if not indicators:
        st.markdown(f'<div class="cs-metric-label">{empty}</div>',
                    unsafe_allow_html=True)
        return
    for indicator in indicators:
        st.markdown(f"- {indicator}")