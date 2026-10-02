"""Central configuration for CyberShield AI.

All tunable thresholds, paths and model metadata live here so that risk
thresholds are transparent and adjustable rather than buried in logic.

No secret values are stored in this file. API keys are read from environment
variables (optionally seeded from a local .env file).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Paths - resolved relative to the project root so the app works on Windows,
# Linux and Render regardless of where the repository is cloned.
# --------------------------------------------------------------------------
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
ASSETS_DIR: Final[Path] = PROJECT_ROOT / "assets"

PHISHING_DATASET: Final[Path] = DATA_DIR / "phishing_features.csv"
SPAM_DATASET: Final[Path] = DATA_DIR / "spam-emails.csv"

PHISHING_MODEL: Final[Path] = MODELS_DIR / "phishing_model.pkl"
SPAM_MODEL: Final[Path] = MODELS_DIR / "spam_model.pkl"
SPAM_VECTORIZER: Final[Path] = MODELS_DIR / "spam_vectorizer.pkl"
METRICS_PATH: Final[Path] = MODELS_DIR / "metrics.json"

# Load a local .env when present. Missing .env is NOT an error.
load_dotenv(PROJECT_ROOT / ".env", override=False)


# --------------------------------------------------------------------------
# Optional external services. Absent keys simply disable the feature.
# --------------------------------------------------------------------------
def get_groq_api_key() -> str | None:
    key = (os.getenv("GROQ_API_KEY") or "").strip()
    return key or None


def get_virustotal_api_key() -> str | None:
    key = (os.getenv("VIRUSTOTAL_API_KEY") or "").strip()
    return key or None


GROQ_MODEL: Final[str] = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TIMEOUT: Final[int] = int(os.getenv("GROQ_TIMEOUT", "30"))
GROQ_MAX_TOKENS: Final[int] = int(os.getenv("GROQ_MAX_TOKENS", "800"))
GROQ_TEMPERATURE: Final[float] = float(os.getenv("GROQ_TEMPERATURE", "0.3"))

VIRUSTOTAL_ENDPOINT: Final[str] = "https://www.virustotal.com/api/v3/urls"
VIRUSTOTAL_TIMEOUT: Final[int] = int(os.getenv("VIRUSTOTAL_TIMEOUT", "10"))

# Public (keyless) rate limiting for the free VirusTotal tier.
VIRUSTOTAL_MIN_INTERVAL_SECONDS: Final[float] = float(
    os.getenv("VIRUSTOTAL_MIN_INTERVAL_SECONDS", "15")
)
VIRUSTOTAL_CACHE_TTL_SECONDS: Final[int] = int(
    os.getenv("VIRUSTOTAL_CACHE_TTL_SECONDS", "900")
)
VIRUSTOTAL_CACHE_MAX_ENTRIES: Final[int] = 512

VIRUSTOTAL_UNAVAILABLE_MESSAGE: Final[str] = (
    "VirusTotal integration is currently unavailable."
)


# --------------------------------------------------------------------------
# Phishing detection configuration
# --------------------------------------------------------------------------
# Column contract for the Random Forest phishing model. Training and
# inference both consume features in exactly this order; the order is stored
# with the model and re-validated on load to prevent silent schema drift.
PHISHING_TARGET: Final[str] = "label"
PHISHING_URL_COLUMN: Final[str] = "url"
PHISHING_LABEL_PHISHING: Final[int] = 1
PHISHING_LABEL_LEGITIMATE: Final[int] = 0

PHISHING_CLASS_NAMES: Final[dict[int, str]] = {
    PHISHING_LABEL_LEGITIMATE: "Likely legitimate",
    PHISHING_LABEL_PHISHING: "Potential phishing",
}

# Risk thresholds keyed on the predicted probability of the phishing class.
PHISHING_RISK_THRESHOLDS: Final[dict[str, float]] = {
    "HIGH": 0.60,
    "MEDIUM": 0.25,
}
PHISHING_DEFAULT_RISK: Final[str] = "LOW"

# Below this probability the classifier is considered undecided and the UI
# reports low confidence rather than asserting a verdict.
PHISHING_LOW_CONFIDENCE_BELOW: Final[float] = 0.60

# --- Rule layer ------------------------------------------------------------
# The dataset's legitimate class contains only 820 bare apex domains, so the
# classifier alone assigns a phishing probability of ~1.0 to any URL that
# contains a path segment. The final risk is therefore an explicit aggregation
# of the ML probability and a transparent rule layer, and the ML signal alone
# is deliberately capped at MEDIUM risk.
PHISHING_RULE_WEIGHTS: Final[dict[str, float]] = {
    "HIGH": 0.55,
    "MEDIUM": 0.20,
    "INFO": 0.03,
}
RULE_HIGH_COMBO: Final[float] = 0.55
RULE_MEDIUM_COMBO: Final[float] = 0.15
ML_ONLY_MAX_RISK: Final[str] = "MEDIUM"
ML_MEDIUM_THRESHOLD: Final[float] = 0.25

# Extensions that have no legitimate purpose in a link a user is invited to
# click, and that indicate a direct malware payload.
DANGEROUS_EXTENSIONS: Final[tuple[str, ...]] = (
    ".exe", ".bat", ".cmd", ".scr", ".pif", ".com", ".msi", ".msp", ".apk",
    ".jar", ".vbs", ".vbe", ".js", ".jse", ".ps1", ".sh", ".bash", ".dll",
    ".bin", ".run", ".hta", ".cpl", ".reg", ".gadget", ".iso", ".img",
)

# Class imbalance handling. The dataset is ~99.5% phishing, so the training
# script undersamples the majority class and also uses balanced class weights.
PHISHING_RANDOM_STATE: Final[int] = 42
PHISHING_TEST_SIZE: Final[float] = 0.20
PHISHING_VALID_SIZE: Final[float] = 0.10
PHISHING_MAX_MAJORITY_SAMPLES: Final[int] = 20_000
PHISHING_N_ESTIMATORS: Final[int] = 400
PHISHING_MAX_DEPTH: Final[int | None] = 18
PHISHING_MIN_SAMPLES_LEAF: Final[int] = 1

# Keywords counted by the `suspicious_words` feature.
SUSPICIOUS_KEYWORDS: Final[tuple[str, ...]] = (
    "login", "log-in", "signin", "verify", "verification", "secure",
    "account", "update", "confirm", "password", "credential", "billing",
    "invoice", "wallet", "banking", "suspend", "unlock", "webscr",
    "prize", "bonus", "reward", "winner", "claim", "free", "offer",
    "limited", "urgent", "alert", "confirm", "authorize", "auth",
    "customer", "support", "helpdesk", "recovery", "validate",
)

# Domains/TLDs that are structurally suspicious in bulk-registration
# campaigns. Used only for the transparent "indicators" list, never to
# override the model prediction.
HIGH_RISK_TLDS: Final[tuple[str, ...]] = (
    "tk", "ml", "ga", "cf", "gq", "top", "xyz", "click", "link", "work",
    "country", "kim", "rest", "review", "stream", "download", "loan",
    "racing", "win", "bid", "date", "faith", "party", "science", "gdn",
    "cam", "surf", "quest", "cyou", "sbs", "buzz",
)


# --------------------------------------------------------------------------
# Email spam detection configuration
# --------------------------------------------------------------------------
SPAM_LABEL_COLUMN: Final[str] = "Category"
SPAM_TEXT_COLUMN: Final[str] = "Message"
SPAM_LABEL_SPAM: Final[str] = "spam"
SPAM_LABEL_HAM: Final[str] = "ham"

SPAM_RANDOM_STATE: Final[int] = 42
SPAM_TEST_SIZE: Final[float] = 0.20
SPAM_NB_ALPHA: Final[float] = 0.1
SPAM_TFIDF_NGRAM_RANGE: Final[tuple[int, int]] = (1, 2)
SPAM_TFIDF_MIN_DF: Final[int] = 1
SPAM_TFIDF_MAX_DF: Final[float] = 0.95
SPAM_TFIDF_MAX_FEATURES: Final[int] = 60_000

SPAM_RISK_THRESHOLDS: Final[dict[str, float]] = {
    "HIGH": 0.80,
    "MEDIUM": 0.50,
}
SPAM_LOW_CONFIDENCE_BELOW: Final[float] = 0.60
SPAM_MIN_CHARACTERS: Final[int] = 5

EMAIL_MAX_CHARACTERS: Final[int] = 100_000
URL_MAX_CHARACTERS: Final[int] = 4_096


# --------------------------------------------------------------------------
# Password analyzer configuration
# --------------------------------------------------------------------------
PASSWORD_MIN_LENGTH: Final[int] = 12
PASSWORD_GOOD_LENGTH: Final[int] = 16
PASSWORD_MAX_LENGTH: Final[int] = 128

PASSWORD_LENGTH_POINTS: Final[int] = 35
PASSWORD_CHARCLASS_POINTS: Final[int] = 25
PASSWORD_DIVERSITY_POINTS: Final[int] = 20
PASSWORD_SEQUENCE_POINTS: Final[int] = 10
PASSWORD_BREACH_PATTERN_POINTS: Final[int] = 10
# The five components above sum to exactly 100, so a flawless password can
# reach the top of the scale.
PASSWORD_MAX_SCORE: Final[int] = 100

PASSWORD_STRENGTH_THRESHOLDS: Final[tuple[int, str]] = (
    (80, "STRONG"),
    (55, "MEDIUM"),
    (0, "WEAK"),
)

COMMON_PASSWORDS: Final[frozenset[str]] = frozenset({
    "password", "password1", "password123", "passw0rd", "123456", "1234567",
    "12345678", "123456789", "1234567890", "123123", "111111", "000000",
    "qwerty", "qwertyuiop", "qwerty123", "letmein", "welcome", "welcome1",
    "admin", "admin123", "administrator", "root", "toor", "guest", "login",
    "abc123", "abcd1234", "iloveyou", "monkey", "dragon", "sunshine",
    "princess", "football", "baseball", "master", "shadow", "superman",
    "trustno1", "starwars", "whatever", "pass", "test", "test123",
    "changeme", "secret", "secret123", "google", "facebook", "instagram",
    "qwerty1", "zaq1zaq1", "1q2w3e4r", "1qaz2wsx", "asdfghjkl",
    "asdf1234", "p@ssw0rd", "p@ssword", "p@ss1234", "welcome123",
    "Summer2024", "Winter2024", "Company123", "Qwerty123",
})

WEAK_PATTERNS: Final[tuple[str, ...]] = (
    "password", "qwerty", "admin", "welcome", "letmein", "iloveyou",
    "monkey", "dragon", "football", "baseball", "master", "shadow",
    "sunshine", "princess", "trustno1", "starwars", "login", "secret",
    "changeme", "test123", "abc123", "1q2w3e", "asdfgh", "zxcvbn",
)


# --------------------------------------------------------------------------
# Security score configuration (rule based and transparent)
# --------------------------------------------------------------------------
SCORE_WEIGHTS: Final[dict[str, int]] = {
    "password": 25,
    "url": 25,
    "email": 25,
    "qr": 25,
}
SCORE_MAX: Final[int] = sum(SCORE_WEIGHTS.values())

# An un-analysed component scores 0 and is reported as "Not analyzed" so the
# dashboard can never imply a perfect score from missing data.
SCORE_UNANALYZED_POINTS: Final[int] = 0

SCORE_BANDS: Final[tuple[tuple[int, str], ...]] = (
    (80, "GOOD"),
    (60, "MODERATE"),
    (0, "NEEDS_ATTENTION"),
)


# --------------------------------------------------------------------------
# Application metadata
# --------------------------------------------------------------------------
APP_NAME: Final[str] = "CyberShield AI"
APP_TAGLINE: Final[str] = "AI-Powered Cybersecurity Assistant"
APP_VERSION: Final[str] = "1.0.0"

# Disclosed limitation about the phishing training data. Surfaced in the UI so
# the model is never presented as a guarantee of safety.
PHISHING_DATA_CAVEAT: Final[str] = (
    "Training-data limitation: the legitimate class of the supplied dataset "
    "contains only 820 bare apex domains with no path segments, while the "
    "phishing class contains full URLs with paths. This detector has "
    "therefore not seen legitimate URLs that contain a path, so it may "
    "over-flag them. Treat the output as one signal among many."
)