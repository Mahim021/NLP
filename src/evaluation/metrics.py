"""Shared evaluation code.

Every model is scored through this module so the reported numbers are
comparable by construction (same metrics, same positive class, same test set).

The positive class is "Yes" = Actionable, so a False Negative is an actionable
e-mail predicted non-actionable -- the error that makes a user miss something
(PROJECT_SPEC.md Sec. 10).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.labels import CLASS_NAMES, POSITIVE_ID  # noqa: E402


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Accuracy / precision / recall / F1 / confusion matrix and its four cells."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Positive class = Actionable ("Yes")
        "precision": float(precision_score(y_true, y_pred, pos_label=POSITIVE_ID, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POSITIVE_ID, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_ID, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": {"rows": "true", "cols": "predicted",
                                    "order": ["No (0)", "Yes (1)"]},
        "support": {"n": int(len(y_true)),
                    "actual_yes": int((y_true == 1).sum()),
                    "actual_no": int((y_true == 0).sum())},
    }
    if y_prob is not None and len(set(y_true.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_prob)))
    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=CLASS_NAMES, zero_division=0)
    return metrics


def print_metrics(name: str, split: str, m: dict) -> None:
    print(f"\n=== {name} | {split} ===")
    print(f"Accuracy : {m['accuracy']:.4f}")
    print(f"Precision: {m['precision']:.4f}   (positive class = Actionable/Yes)")
    print(f"Recall   : {m['recall']:.4f}")
    print(f"F1-score : {m['f1']:.4f}")
    if "roc_auc" in m:
        print(f"ROC-AUC  : {m['roc_auc']:.4f}")
    print(f"Confusion matrix (rows=true [No,Yes], cols=pred [No,Yes]): {m['confusion_matrix']}")
    print(f"TP={m['true_positives']}  TN={m['true_negatives']}  "
          f"FP={m['false_positives']}  FN={m['false_negatives']}")
    print(m["classification_report"])


def save_results(model_name: str, results_dir: Path, metrics_by_split: dict,
                 config: dict, history: list | None = None,
                 predictions: pd.DataFrame | None = None) -> None:
    """Write Git-friendly artefacts: JSON metrics, CSV confusion matrix, CSV predictions."""
    results_dir = Path(results_dir)
    model_dir = results_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    payload = {"model": model_name, "config": config, "metrics": metrics_by_split}
    if history:
        payload["history"] = history
    with open(model_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    for split, m in metrics_by_split.items():
        cm = pd.DataFrame(m["confusion_matrix"],
                          index=["true_No", "true_Yes"],
                          columns=["pred_No", "pred_Yes"])
        cm.to_csv(model_dir / f"confusion_matrix_{split}.csv")

    if history:
        pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)
    if predictions is not None:
        predictions.to_csv(model_dir / "test_predictions.csv", index=False, encoding="utf-8")

    row = {"model": model_name}
    row.update({k: v for k, v in metrics_by_split["test"].items()
                if isinstance(v, (int, float))})
    pd.DataFrame([row]).to_csv(model_dir / "test_summary.csv", index=False)
    print(f"\nSaved results to {model_dir}")
