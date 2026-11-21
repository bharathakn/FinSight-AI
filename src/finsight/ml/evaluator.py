"""Metrics (Plan §7.1): PR-AUC (primary), ROC-AUC, F1 / precision / recall at a validation-tuned
threshold, per-scenario recall, BORDERLINE flag rate, paired significance tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y_true, proba) -> float:
    """Threshold maximising F1 on (validation) data."""
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(t) else 0
    return float(t[i]) if len(t) else 0.5


def scores(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "threshold": threshold, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def per_scenario(frame: pd.DataFrame, proba, threshold: float) -> pd.DataFrame:
    """Flag rate per scenario × label: recall for FRAUDULENT, false-alarm rate for the others."""
    d = frame[["scenario", "risk_label"]].assign(flag=np.asarray(proba) >= threshold)
    return d.groupby(["scenario", "risk_label"], observed=True)["flag"].agg(["mean", "size"]) \
        .rename(columns={"mean": "flag_rate", "size": "n"}).reset_index()


def paired_test(a, b) -> dict:
    """Paired comparison of two score vectors on identical CV folds (b − a)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t = stats.ttest_rel(b, a)
    try:
        w = stats.wilcoxon(b, a).pvalue
    except ValueError:
        w = float("nan")
    return {"mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)), "wins": int((d > 0).sum()),
            "n": len(d), "t_pvalue": float(t.pvalue), "wilcoxon_pvalue": float(w)}
