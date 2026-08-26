from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def safe_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def probability_or_score_matrix(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X_test)
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        exp_scores = np.exp(decision - np.max(decision, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return None


def false_rates(conf_matrix):
    fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    fn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    tp = np.diag(conf_matrix)
    fpr = fp / np.maximum(fp + tp, 1)
    fnr = fn / np.maximum(fn + tp, 1)
    return fpr, fnr


def evaluate_model(model_name: str, model, X_train, X_test, y_train, y_test, class_labels: Iterable[str], output_dir: Path) -> Dict[str, float]:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    testing_time = time.perf_counter() - start

    labels = sorted(np.unique(y_train))
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)
    fpr, fnr = false_rates(conf_matrix)

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=[class_labels[i] for i in labels],
        zero_division=0,
        output_dict=True,
    )

    model_slug = safe_name(model_name)
    pd.DataFrame(conf_matrix).to_csv(output_dir / f"{model_slug}_confusion_matrix.csv", index=False)
    pd.DataFrame(report).transpose().to_csv(output_dir / f"{model_slug}_classification_report.csv")
    pd.DataFrame({"class": [class_labels[i] for i in labels], "false_positive_rate": fpr, "false_negative_rate": fnr}).to_csv(
        output_dir / f"{model_slug}_false_rates.csv", index=False
    )

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision Macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Recall Macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "F1 Macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Precision Weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall Weighted": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1 Weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Training Time (s)": training_time,
        "Testing Time (s)": testing_time,
    }

    y_score = probability_or_score_matrix(model, X_test)
    if y_score is not None:
        y_test_bin = label_binarize(y_test, classes=labels)
        metrics["ROC-AUC Macro"] = roc_auc_score(y_test_bin, y_score, average="macro", multi_class="ovr")
        metrics["PR-AUC Macro"] = average_precision_score(y_test_bin, y_score, average="macro")
        metrics["ROC-AUC Weighted"] = roc_auc_score(y_test_bin, y_score, average="weighted", multi_class="ovr")
        metrics["PR-AUC Weighted"] = average_precision_score(y_test_bin, y_score, average="weighted")
    else:
        metrics["ROC-AUC Macro"] = np.nan
        metrics["PR-AUC Macro"] = np.nan
        metrics["ROC-AUC Weighted"] = np.nan
        metrics["PR-AUC Weighted"] = np.nan

    return metrics


def save_summary_table(rows, output_dir: Path) -> pd.DataFrame:
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "model_summary_metrics.csv", index=False)

    table = summary.set_index("Model").transpose().reset_index().rename(columns={"index": "Metric"})
    table.to_csv(output_dir / "model_comparison_table.csv", index=False)
    return table
