"""Train the TF-IDF + Multinomial Naive Bayes email spam detector.

Run with::

    python training/train_spam.py

Pipeline
--------
raw message -> :func:`utils.helpers.clean_email_text` -> TfidfVectorizer
(1-2 grams, sublinear tf) -> MultinomialNB -> two joblib artifacts.

The dataset is 86.6% ham / 13.4% spam, which is mild enough that no
resampling or class weighting is required; accuracy and per-class F1 are both
reported so the spam class is not hidden by the majority.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    METRICS_PATH,
    MODELS_DIR,
    SPAM_DATASET,
    SPAM_LABEL_SPAM,
    SPAM_MODEL,
    SPAM_NB_ALPHA,
    SPAM_RANDOM_STATE,
    SPAM_TEST_SIZE,
    SPAM_TFIDF_MAX_DF,
    SPAM_TFIDF_MAX_FEATURES,
    SPAM_TFIDF_MIN_DF,
    SPAM_TFIDF_NGRAM_RANGE,
    SPAM_VECTORIZER,
)
from training.evaluate_models import classification_block  # noqa: E402
from utils.helpers import clean_email_text  # noqa: E402
from utils.preprocessing import audit_spam, load_spam_dataset  # noqa: E402


def main() -> int:
    if not SPAM_DATASET.exists():
        print(f"ERROR: dataset not found at {SPAM_DATASET}", file=sys.stderr)
        return 1

    audit = audit_spam(SPAM_DATASET)
    print("=" * 70)
    print("DATASET AUDIT - spam-emails.csv")
    print("=" * 70)
    print(f"rows                : {audit.rows}")
    print(f"columns             : {audit.columns}")
    print(f"duplicate rows      : {audit.duplicate_rows}")
    print(f"missing values      : {audit.missing or 'none'}")
    print(f"class distribution  : {audit.class_distribution}")
    print(f"class percentages   : {audit.class_percentages}")
    for note in audit.notes:
        print(f"  * {note}")

    frame = load_spam_dataset(SPAM_DATASET)
    removed = frame.attrs.get("duplicates_removed", 0)
    print(f"\nafter cleaning      : {len(frame)} rows ({removed} duplicates removed)")

    frame["Message"] = frame["Message"].map(clean_email_text)
    frame = frame[frame["Message"].str.len() > 0].reset_index(drop=True)

    X = frame["Message"]
    y = frame["Category"]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SPAM_TEST_SIZE, random_state=SPAM_RANDOM_STATE, stratify=y
    )
    print(f"split               : train={len(X_train)} test={len(X_test)}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=SPAM_TFIDF_NGRAM_RANGE,
        min_df=SPAM_TFIDF_MIN_DF,
        max_df=SPAM_TFIDF_MAX_DF,
        max_features=SPAM_TFIDF_MAX_FEATURES,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"vocabulary size     : {len(vectorizer.vocabulary_)}")

    classifier = MultinomialNB(alpha=SPAM_NB_ALPHA)
    classifier.fit(X_train_vec, y_train)
    print(f"model trained       : MultinomialNB(alpha={SPAM_NB_ALPHA})")

    test_metrics = classification_block(
        y_test, classifier.predict(X_test_vec), "TEST"
    )
    train_metrics = classification_block(
        y_train, classifier.predict(X_train_vec), "TRAIN"
    )

    figure = ConfusionMatrixDisplay.from_estimator(
        classifier, X_test_vec, y_test, display_labels=["Ham", "Spam"],
        cmap="Blues", colorbar=False,
    )
    figure.ax_.set_title("Spam detector - test confusion matrix")
    figure.figure_.tight_layout()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    confusion_path = MODELS_DIR / "spam_confusion_matrix.png"
    figure.figure_.savefig(confusion_path, dpi=110)

    top_features = vectorizer.get_feature_names_out()
    spam_index = list(classifier.classes_).index(SPAM_LABEL_SPAM)
    log_ratios = classifier.feature_log_prob_[spam_index] - (
        classifier.feature_log_prob_[1 - spam_index]
    )
    ranked = log_ratios.argsort()[::-1][:25]
    print("\ntop 25 most spam-indicative terms:")
    for position in ranked:
        print(f"   {top_features[position]:<24}{log_ratios[position]:+.3f}")

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "MultinomialNB",
        "vectorizer": "TfidfVectorizer",
        "alpha": SPAM_NB_ALPHA,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "class_distribution": audit.class_distribution,
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "confusion_matrix_path": confusion_path.name,
        "top_spam_terms": [str(top_features[p]) for p in ranked],
        "limitations": [
            "The corpus is SMS-style short messages from the Enron/SMS Spam "
            "collection, so performance on long HTML marketing emails will be "
            "lower than reported here.",
            "No model can guarantee that a message is safe; treat the result as "
            "one signal.",
        ],
    }

    joblib.dump(vectorizer, SPAM_VECTORIZER, compress=3)
    joblib.dump({"model": classifier, "metadata": metadata}, SPAM_MODEL, compress=3)
    print(f"\nsaved vectorizer     : {SPAM_VECTORIZER.name} "
          f"({SPAM_VECTORIZER.stat().st_size / 1024:.0f} KB)")
    print(f"saved model          : {SPAM_MODEL.name} "
          f"({SPAM_MODEL.stat().st_size / 1024:.0f} KB)")
    print(f"saved confusion matrix: {confusion_path.name}")

    existing: dict = {}
    if METRICS_PATH.exists():
        try:
            existing = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            existing = {}
    existing["spam"] = metadata
    METRICS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"saved metrics        : {METRICS_PATH.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())