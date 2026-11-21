"""Module 7: explainability (Plan §7.9).

* Global + local SHAP for the tuned XGBoost (TreeSHAP via the `shap` library, used offline for the report)
* Serve-time explanations via XGBoost's native `pred_contribs=True`, which computes the same TreeSHAP
  values without importing `shap` (D-024). `contributions()` is what ml-service will call; a parity check
  against `shap.TreeExplainer` is part of the Phase 7 run.
* Permutation importance (PR-AUC drop) for the production model, which is model-agnostic and so also
  covers the stacking ensemble.
All values are in log-odds units of XGBoost's raw margin: positive pushes towards fraud.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finsight import config


def xgb_parts(pipe):
    """(fitted preprocessor, XGBClassifier, output feature names) from a prep → model pipeline."""
    prep = pipe.named_steps["prep"]
    return prep, pipe.named_steps["model"], list(prep.get_feature_names_out())


def contributions(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Native TreeSHAP contributions (n × features) plus the 'bias' column; rows sum to the raw margin."""
    import xgboost as xgb

    prep, model, names = xgb_parts(pipe)
    Xt = np.asarray(prep.transform(X), dtype=np.float32)
    c = model.get_booster().predict(xgb.DMatrix(Xt, feature_names=None), pred_contribs=True)
    return pd.DataFrame(c, columns=names + ["bias"], index=X.index)


def shap_values(pipe, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    """(shap values, transformed X, expected value, feature names) using shap.TreeExplainer."""
    import shap

    prep, model, names = xgb_parts(pipe)
    Xt = np.asarray(prep.transform(X), dtype=np.float32)
    ex = shap.TreeExplainer(model)
    sv = np.asarray(ex.shap_values(Xt))
    ev = float(np.ravel(ex.expected_value)[0])
    return sv, Xt, ev, names


def parity(pipe, X: pd.DataFrame) -> dict:
    """Max |native pred_contribs − shap.TreeExplainer| and additivity error vs the model margin."""
    sv, Xt, ev, _ = shap_values(pipe, X)
    c = contributions(pipe, X)
    native = c.drop(columns="bias").to_numpy()
    margin = pipe.named_steps["model"].predict(Xt, output_margin=True)
    return {"rows": int(len(X)), "max_abs_diff_vs_shap": float(np.abs(native - sv).max()),
            "max_additivity_error": float(np.abs(c.to_numpy().sum(1) - margin).max())}


def global_importance(sv: np.ndarray, names: list[str]) -> pd.DataFrame:
    return (pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


def top_reasons(contrib_row: pd.Series, raw_row: pd.Series | None = None, k: int = 5) -> list[dict]:
    """Largest absolute contributions for one transaction (what the API returns as 'reasons')."""
    c = contrib_row.drop("bias", errors="ignore")
    top = c.reindex(c.abs().sort_values(ascending=False).index[:k])
    out = []
    for f, v in top.items():
        val = None
        if raw_row is not None:
            if f in raw_row.index:
                val = raw_row[f]
            else:   # one-hot column, e.g. sender_kyc_status_VERIFIED → raw column sender_kyc_status
                src = max((c for c in raw_row.index if f.startswith(f"{c}_")), key=len, default=None)
                val = raw_row[src] if src else None
        out.append({"feature": f, "value": None if val is None else (val.item() if hasattr(val, "item") else val),
                    "contribution": round(float(v), 4), "direction": "towards fraud" if v > 0 else "towards legitimate"})
    return out


def permutation(estimator, X: pd.DataFrame, y, n_repeats: int = 5, n_jobs: int = -1) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    r = permutation_importance(estimator, X, y, scoring="average_precision", n_repeats=n_repeats,
                               random_state=config.SEED, n_jobs=n_jobs)
    return (pd.DataFrame({"feature": X.columns, "pr_auc_drop_mean": r.importances_mean, "pr_auc_drop_std": r.importances_std})
            .sort_values("pr_auc_drop_mean", ascending=False).reset_index(drop=True))
