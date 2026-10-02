"""Shared evaluation helpers.

All numbers reported by the project are computed here from a real held-out
split. Nothing is hard-coded or hand-entered.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_block(
    y_true: Any, y_pred: Any, title: str = "TEST"
) -> dict[str, Any]:
    """Compute and print the standard metrics for a classification task.

    Returns a JSON-serialisable dict with per-class and macro/weighted averages.
    """
    report = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )
    # Keep the original label objects for computation, and a string form only
    # for display. Mixing the two silently produces an all-zero matrix.
    labels = sorted({*np.unique(y_true), *np.unique(y_pred)}, key=str)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    label_names = [str(label) for label in labels]

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(precision_score(
            y_true, y_pred, average="macro", zero_division=0)), 6),
        "macro_recall": round(float(recall_score(
            y_true, y_pred, average="macro", zero_division=0)), 6),
        "macro_f1": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0)), 6),
        "labels": label_names,
        "confusion_matrix": matrix.tolist(),
        "support": {label: int(report[label]["support"])
                    for label in label_names if label in report},
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 6),
                "recall": round(float(report[label]["recall"]), 6),
                "f1": round(float(report[label]["f1-score"]), 6),
                "support": int(report[label]["support"]),
            }
            for label in label_names if label in report
        },
    }

    print("\n" + "-" * 70)
    print(f"{title} SET METRICS")
    print("-" * 70)
    print(f"accuracy       : {metrics['accuracy']:.4f}")
    print(f"macro F1       : {metrics['macro_f1']:.4f}")
    print(f"macro precision: {metrics['macro_precision']:.4f}")
    print(f"macro recall   : {metrics['macro_recall']:.4f}")
    print(f"weighted F1    : {metrics['weighted_f1']:.4f}")
    print("\nper class:")
    for label, values in metrics["per_class"].items():
        print(f"  {label:<10} P={values['precision']:.4f} "
              f"R={values['recall']:.4f} F1={values['f1']:.4f} "
              f"n={values['support']}")
    print("\nconfusion matrix (rows = actual, cols = predicted):")
    header = "".join(f"{label:>10}" for label in label_names)
    print(f"{'':<10}{header}")
    for label, row in zip(label_names, matrix):
        cells = "".join(f"{int(value):>10}" for value in row)
        print(f"{label:<10}{cells}")
    return metrics