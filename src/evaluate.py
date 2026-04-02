"""Evaluation utilities for per-level classifiers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

logger = logging.getLogger(__name__)


def evaluate_level(
    model: Any,
    X_test: list[str] | np.ndarray,
    y_test: list[str] | np.ndarray,
    level_name: str,
) -> dict[str, Any]:
    """Compute classification metrics for a single level model."""

    predictions = model.predict(X_test)
    report_dict = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(y_test, predictions, zero_division=0)

    per_class_precision = {
        label: metrics["precision"]
        for label, metrics in report_dict.items()
        if isinstance(metrics, dict)
    }
    per_class_recall = {
        label: metrics["recall"]
        for label, metrics in report_dict.items()
        if isinstance(metrics, dict)
    }
    per_class_f1 = {
        label: metrics["f1-score"]
        for label, metrics in report_dict.items()
        if isinstance(metrics, dict)
    }

    metrics = {
        "level_name": level_name,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "classification_report": report_text,
    }

    logger.info(
        "Evaluated %s: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f",
        level_name,
        metrics["accuracy"],
        metrics["macro_f1"],
        metrics["weighted_f1"],
    )
    return metrics


def plot_confusion_matrix(
    model: Any,
    X_test: list[str] | np.ndarray,
    y_test: list[str] | np.ndarray,
    level_name: str,
    output_dir: str,
) -> str | None:
    """Save a confusion matrix plot if the class count is manageable."""

    labels = sorted(np.unique(y_test))
    if len(labels) > 20:
        logger.info(
            "Skipping confusion matrix for %s because it has %s classes",
            level_name,
            len(labels),
        )
        return None

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=labels)

    figure, axis = plt.subplots(figsize=(8, 6))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=axis, cmap="Blues", xticks_rotation=45, colorbar=False)
    axis.set_title(f"{level_name} confusion matrix")
    figure.tight_layout()

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / f"{level_name.lower()}_confusion_matrix.png"
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    logger.info("Saved confusion matrix for %s to %s", level_name, output_path)
    return str(output_path)


def save_metrics(metrics: dict[str, Any], output_path: str) -> None:
    """Persist level metrics to JSON."""

    with Path(output_path).open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)
