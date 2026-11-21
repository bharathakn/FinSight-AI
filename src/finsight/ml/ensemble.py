"""Module 5: stacking ensemble + probability calibration (Plan §7.7).

Stack      tuned RF + XGBoost + MLP (each a full preprocessing pipeline) → logistic-regression meta-learner
           trained on 5-fold out-of-fold probabilities (`StackingClassifier(cv=5)`), so the meta-learner
           never sees a base learner's in-sample predictions.
Calibrate  isotonic regression with `CalibratedClassifierCV(cv=5, ensemble=False)`: the calibrator is learned
           on out-of-fold predictions of the whole stack, and **one** stack refitted on all training rows is
           kept. Storing one stack instead of five keeps the artifact small enough for a 512 MB server (D-029).
Everything is fitted on the training split only; validation is used for model choice, threshold and tiers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

STACK_BASE = ["random_forest", "xgboost", "mlp"]


def build_stack(tuned: dict, cv=5, n_jobs: int = -1) -> StackingClassifier:
    """`tuned`: name → fitted/unfitted pipeline with tuned hyperparameters (cloned here)."""
    return StackingClassifier(
        estimators=[(n, clone(tuned[n])) for n in STACK_BASE],
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=cv, stack_method="predict_proba", n_jobs=n_jobs, passthrough=False)


def calibrated(estimator, cv=5, method: str = "isotonic") -> CalibratedClassifierCV:
    return CalibratedClassifierCV(clone(estimator), method=method, cv=cv, ensemble=False)


def uncalibrated_inner(cal: CalibratedClassifierCV):
    """The single estimator refitted on all training rows inside a CalibratedClassifierCV(ensemble=False)."""
    return cal.calibrated_classifiers_[0].estimator


def meta_weights(stack: StackingClassifier) -> dict:
    """Logistic-regression coefficients on each base learner's P(fraud) column."""
    lr = stack.final_estimator_
    names = [n for n, _ in stack.estimators]
    coef = lr.coef_.ravel()
    # predict_proba with 2 classes → StackingClassifier keeps only the positive column per estimator
    return {"intercept": float(lr.intercept_[0]), **{n: float(c) for n, c in zip(names, coef)}}


def reliability(y, p, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean, "fraction_positive": frac}), float(brier_score_loss(y, p))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(n_bins) if (idx == b).any()))

