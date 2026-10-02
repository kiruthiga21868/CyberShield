"""Train the Random Forest phishing URL detector.

Run with::

    python training/train_phishing.py

Pipeline
--------
raw URL -> :func:`utils.url_features.extract_url_features` -> column
transformer (ordinal TLD encoding) -> RandomForestClassifier -> joblib artifact

Design decisions driven by the dataset audit
--------------------------------------------
* **Features are re-derived from the raw URL.** The shipped pre-extracted
  columns were produced by an inconsistent extractor (``tld``/``has_https`` were
  broken for the scheme-less legitimate rows). Re-deriving every column with
  one implementation removes that artifact and guarantees that inference uses
  the identical feature function as training.
* **The majority class is undersampled and class weights are balanced.** The
  corpus is ~99.5% phishing; without this the model degenerates to always
  answering "phishing" and accuracy becomes meaningless.
* **Metrics are computed on a stratified held-out test split** and reported
  per class. Macro F1 is the headline number because accuracy alone is
  misleading under imbalance.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    METRICS_PATH,
    MODELS_DIR,
    PHISHING_DATASET,
    PHISHING_MAX_DEPTH,
    PHISHING_MAX_MAJORITY_SAMPLES,
    PHISHING_MIN_SAMPLES_LEAF,
    PHISHING_MODEL,
    PHISHING_N_ESTIMATORS,
    PHISHING_RANDOM_STATE,
    PHISHING_TEST_SIZE,
    PHISHING_VALID_SIZE,
)
from training.evaluate_models import classification_block  # noqa: E402
from utils.preprocessing import audit_phishing, load_phishing_dataset  # noqa: E402
from utils.url_features import (  # noqa: E402
    ALL_FEATURE_NAMES,
    CATEGORICAL_FEATURES,
    extract_feature_frame,
)


def build_dataset() -> tuple[pd.DataFrame, dict]:
    """Load, clean, re-featurise and balance the phishing corpus."""
    audit = audit_phishing(PHISHING_DATASET)
    print("=" * 70)
    print("DATASET AUDIT - phishing_features.csv")
    print("=" * 70)
    print(f"rows                : {audit.rows}")
    print(f"columns             : {audit.columns}")
    print(f"duplicate rows      : {audit.duplicate_rows}")
    print(f"missing values      : {audit.missing or 'none'}")
    print(f"class distribution  : {audit.class_distribution}")
    print(f"class percentages   : {audit.class_percentages}")
    for note in audit.notes:
        print(f"  * {note}")

    frame = load_phishing_dataset(PHISHING_DATASET)
    print(f"\nafter cleaning      : {len(frame)} rows "
          f"({len(frame) - audit.rows} removed as invalid/duplicate)")

    urls = frame["url"].tolist()
    labels = frame["label"].to_numpy()
    features, valid_urls, dropped = extract_feature_frame(urls)
    if dropped:
        print(f"feature extraction  : dropped {len(dropped)} unparseable URL(s)")
    valid_set = set(valid_urls)
    keep_mask = frame["url"].isin(valid_set).to_numpy()
    y = labels[keep_mask]
    features = features.reset_index(drop=True)
    features["label"] = y
    print(f"feature matrix      : {features.shape}")

    return features, audit.to_dict()


def balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Undersample the majority class so both classes contribute evenly.

    Sampling is done on the *training* split only (see ``main``) to avoid
    leaking distribution information from the test set. This function simply
    reports the target size; the split happens first.
    """
    counts = frame["label"].value_counts()
    minority = int(counts.min())
    target = min(int(counts.max()), max(minority * 5, PHISHING_MAX_MAJORITY_SAMPLES))
    print(f"\nclass counts        : {counts.to_dict()}")
    print(f"balanced target     : {target} majority / {minority} minority "
          f"(cap {PHISHING_MAX_MAJORITY_SAMPLES})")
    return frame


def main() -> int:
    if not PHISHING_DATASET.exists():
        print(f"ERROR: dataset not found at {PHISHING_DATASET}", file=sys.stderr)
        return 1

    frame, audit_dict = build_dataset()
    balance(frame)

    # ---- split before any resampling so the test set stays untouched ----
    from sklearn.model_selection import train_test_split

    y = frame["label"].to_numpy()
    X = frame.drop(columns=["label"])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=PHISHING_TEST_SIZE + PHISHING_VALID_SIZE,
        random_state=PHISHING_RANDOM_STATE, stratify=y,
    )
    relative_test = PHISHING_TEST_SIZE / (PHISHING_TEST_SIZE + PHISHING_VALID_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        random_state=PHISHING_RANDOM_STATE, stratify=y_temp,
    )
    print(f"\nsplit               : train={len(X_train)} val={len(X_val)} "
          f"test={len(X_test)}")

    # ---- undersample the majority class inside the training split only ----
    counts = pd.Series(y_train).value_counts()
    minority_n = int(counts.min())
    target_n = min(int(counts.max()), max(minority_n * 5, PHISHING_MAX_MAJORITY_SAMPLES))
    majority_label = int(counts.idxmax())
    majority_idx = np.where(y_train == majority_label)[0]
    minority_idx = np.where(y_train != majority_label)[0]
    rng = np.random.default_rng(PHISHING_RANDOM_STATE)
    keep_idx = np.concatenate([
        minority_idx,
        rng.choice(majority_idx, size=target_n, replace=False),
    ])
    X_train_bal = X_train.iloc[keep_idx]
    y_train_bal = y_train[keep_idx]
    print(f"balanced train      : {len(X_train_bal)} rows "
          f"(from {len(X_train)}) -> {pd.Series(y_train_bal).value_counts().to_dict()}")

    # ---- pipeline ------------------------------------------------------
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    categorical = [c for c in CATEGORICAL_FEATURES if c in X_train_bal.columns]
    numeric = [c for c in ALL_FEATURE_NAMES if c not in CATEGORICAL_FEATURES]

    # Validate the schema before fitting so column drift is impossible.
    assert list(X_train_bal.columns) == list(ALL_FEATURE_NAMES), (
        "Feature order drifted from utils.url_features.ALL_FEATURE_NAMES"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), categorical),
        ],
        remainder="drop",
    )

    classifier = RandomForestClassifier(
        n_estimators=PHISHING_N_ESTIMATORS,
        max_depth=PHISHING_MAX_DEPTH,
        min_samples_leaf=PHISHING_MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
        random_state=PHISHING_RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline([("preprocess", preprocessor), ("rf", classifier)])
    pipeline.fit(X_train_bal, y_train_bal)
    print(f"\nmodel trained       : RandomForestClassifier("
          f"n_estimators={PHISHING_N_ESTIMATORS}, "
          f"class_weight='balanced_subsample')")

    # ---- evaluate -------------------------------------------------------
    from sklearn.metrics import ConfusionMatrixDisplay

    val_metrics = classification_block(y_val, pipeline.predict(X_val), "VALIDATION")
    test_metrics = classification_block(y_test, pipeline.predict(X_test), "TEST")

    figure = ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, display_labels=["Legitimate", "Phishing"],
        cmap="Blues", colorbar=False,
    )
    figure.ax_.set_title("Phishing detector - test confusion matrix")
    figure.figure_.tight_layout()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    confusion_path = MODELS_DIR / "phishing_confusion_matrix.png"
    figure.figure_.savefig(confusion_path, dpi=110)

    importances = pd.Series(
        pipeline.named_steps["rf"].feature_importances_, index=ALL_FEATURE_NAMES
    ).sort_values(ascending=False)
    print("\ntop feature importances:")
    for name, value in importances.head(12).items():
        print(f"   {name:<22}{value:.5f}")

    # ---- persist -------------------------------------------------------
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "RandomForestClassifier",
        "n_estimators": PHISHING_N_ESTIMATORS,
        "class_weight": "balanced_subsample",
        "feature_names": list(ALL_FEATURE_NAMES),
        "n_train_rows": int(len(X_train_bal)),
        "n_test_rows": int(len(X_test)),
        "raw_class_distribution": audit_dict["class_distribution"],
        "metrics": test_metrics,
        "validation_metrics": val_metrics,
        "feature_importances": {k: round(float(v), 6)
                                for k, v in importances.items()},
        "confusion_matrix_path": confusion_path.name,
        "limitations": [
            "The legitimate class contains only 820 bare apex domains with no "
            "path segments, while every phishing URL includes one. The model has "
            "therefore never seen a legitimate URL containing a path and will "
            "over-flag such URLs.",
            "Accuracy on the held-out test split is not a generalisation estimate "
            "for internet traffic; the classes are close to linearly separable in "
            "this feature space.",
        ],
    }

    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_names": list(ALL_FEATURE_NAMES),
            "metadata": metadata,
        },
        PHISHING_MODEL,
        compress=3,
    )
    print(f"\nsaved model         : {PHISHING_MODEL.name} "
          f"({PHISHING_MODEL.stat().st_size / 1024:.0f} KB)")
    print(f"saved confusion matrix: {confusion_path.name}")

    existing: dict = {}
    if METRICS_PATH.exists():
        try:
            existing = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt metrics file
            existing = {}
    existing["phishing"] = metadata
    METRICS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"saved metrics       : {METRICS_PATH.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())