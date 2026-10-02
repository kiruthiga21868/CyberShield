"""Dataset loading, validation and cleaning helpers.

These routines are used only by the training scripts. The deployed application
never loads the raw datasets, which keeps runtime memory small and guarantees
no training/inference data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    PHISHING_LABEL_PHISHING,
    PHISHING_URL_COLUMN,
    SPAM_LABEL_COLUMN,
    SPAM_TEXT_COLUMN,
)


class DatasetError(RuntimeError):
    """Raised when a dataset is missing, unreadable or structurally wrong."""


@dataclass
class DatasetAudit:
    """Structural summary of a dataset, computed from the real file."""

    name: str
    path: Path
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)
    duplicate_rows: int = 0
    class_distribution: dict[str, int] = field(default_factory=dict)
    class_percentages: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "missing": self.missing,
            "duplicate_rows": self.duplicate_rows,
            "class_distribution": self.class_distribution,
            "class_percentages": self.class_percentages,
            "notes": self.notes,
        }


def require_file(path: Path, label: str) -> None:
    """Raise :class:`DatasetError` when ``path`` is absent."""
    if not path.exists():
        raise DatasetError(
            f"{label} dataset not found at {path}. "
            f"Place the CSV in the project's data/ directory."
        )


def load_phishing_dataset(path: Path) -> pd.DataFrame:
    """Load the phishing CSV, dropping rows unusable for supervised learning."""
    require_file(path, "Phishing")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surfaced with context
        raise DatasetError(f"Could not parse phishing dataset: {exc}") from exc

    missing_cols = {PHISHING_URL_COLUMN, "label"} - set(frame.columns)
    if missing_cols:
        raise DatasetError(
            f"Phishing dataset is missing required columns: {sorted(missing_cols)}"
        )

    frame = frame.dropna(subset=[PHISHING_URL_COLUMN, "label"]).copy()
    frame["url"] = frame["url"].astype(str).str.strip()
    frame = frame[frame["url"].str.len() > 0]

    # Coerce the label to a strict 0/1 integer without silently inventing
    # classes for unexpected values.
    numeric_label = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame[numeric_label.isin([0, 1])].copy()
    frame["label"] = numeric_label[numeric_label.isin([0, 1])].astype(int)

    frame = frame.drop_duplicates(subset=[PHISHING_URL_COLUMN]).reset_index(drop=True)
    return frame


def load_spam_dataset(path: Path) -> pd.DataFrame:
    """Load the spam CSV with robust text decoding and duplicate removal."""
    require_file(path, "Spam")
    try:
        frame = pd.read_csv(path)
    except UnicodeDecodeError:
        # The corpus contains at least one mis-decoded byte sequence.
        frame = pd.read_csv(path, encoding_errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise DatasetError(f"Could not parse spam dataset: {exc}") from exc

    missing_cols = {SPAM_LABEL_COLUMN, SPAM_TEXT_COLUMN} - set(frame.columns)
    if missing_cols:
        raise DatasetError(
            f"Spam dataset is missing required columns: {sorted(missing_cols)}"
        )

    frame = frame.dropna(subset=[SPAM_LABEL_COLUMN, SPAM_TEXT_COLUMN]).copy()
    frame[SPAM_TEXT_COLUMN] = (
        frame[SPAM_TEXT_COLUMN].astype(str).str.replace("\ufffd", " ", regex=False)
    )
    frame[SPAM_LABEL_COLUMN] = (
        frame[SPAM_LABEL_COLUMN].astype(str).str.strip().str.lower()
    )
    frame = frame[frame[SPAM_LABEL_COLUMN].isin(["ham", "spam"])]
    frame = frame[frame[SPAM_TEXT_COLUMN].str.strip().str.len() > 0]

    before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    removed = before - len(frame)
    frame.attrs["duplicates_removed"] = removed
    return frame


def audit_phishing(path: Path) -> DatasetAudit:
    """Describe the phishing dataset without modifying it."""
    require_file(path, "Phishing")
    frame = pd.read_csv(path)
    audit = DatasetAudit(name="phishing_features", path=path)
    audit.rows = int(len(frame))
    audit.columns = list(frame.columns)
    audit.dtypes = {c: str(t) for c, t in frame.dtypes.items()}
    audit.missing = {c: int(n) for c, n in frame.isna().sum().items() if n}
    audit.duplicate_rows = int(frame.duplicated().sum())

    counts = frame["label"].value_counts().to_dict()
    audit.class_distribution = {str(k): int(v) for k, v in counts.items()}
    total = max(len(frame), 1)
    audit.class_percentages = {
        str(k): round(float(v) / total * 100, 4) for k, v in counts.items()
    }

    phish_n = int(counts.get(PHISHING_LABEL_PHISHING, 0))
    legit_n = int(len(frame) - phish_n)
    audit.notes.append(
        f"Pre-extracted feature columns present: "
        f"{[c for c in frame.columns if c not in ('url', 'label')]}"
    )
    audit.notes.append(
        f"Class imbalance: {phish_n} phishing vs {legit_n} legitimate "
        f"(~{phish_n / max(legit_n, 1):.0f}:1). Training uses balanced "
        f"class weights plus majority-class undersampling."
    )
    scheme_less = int((~frame["url"].astype(str).str.contains("://", regex=False)
                       & (frame["label"] == 0)).sum())
    audit.notes.append(
        f"{scheme_less}/{legit_n} legitimate URLs are stored without a scheme "
        f"(bare apex domains), while all phishing URLs include one. This is why "
        f"features are re-derived from the raw URL instead of reusing the "
        f"shipped pre-extracted columns."
    )
    return audit


def audit_spam(path: Path) -> DatasetAudit:
    """Describe the spam dataset without modifying it."""
    require_file(path, "Spam")
    frame = pd.read_csv(path, encoding_errors="replace")
    audit = DatasetAudit(name="spam-emails", path=path)
    audit.rows = int(len(frame))
    audit.columns = list(frame.columns)
    audit.dtypes = {c: str(t) for c, t in frame.dtypes.items()}
    audit.missing = {c: int(n) for c, n in frame.isna().sum().items() if n}
    audit.duplicate_rows = int(frame.duplicated().sum())

    counts = frame[SPAM_LABEL_COLUMN].value_counts().to_dict()
    audit.class_distribution = {str(k): int(v) for k, v in counts.items()}
    total = max(len(frame), 1)
    audit.class_percentages = {
        str(k): round(float(v) / total * 100, 4) for k, v in counts.items()
    }
    audit.notes.append(
        f"{audit.duplicate_rows} fully duplicated rows are removed at training time."
    )
    return audit