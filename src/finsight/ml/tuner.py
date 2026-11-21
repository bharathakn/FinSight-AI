"""Module 4: hyperparameter tuning, cross-validation and learning curves (Plan §7.6, RQ4).

* GridSearchCV for the small discrete spaces (Decision Tree, KNN)
* RandomizedSearchCV (n_iter = 20) for the large spaces (Random Forest, XGBoost); n_iter = 10 for the MLP
* 5-fold stratified CV on the **training split only** (stratified on risk_label), scored by average
  precision (PR-AUC). Validation stays untouched for calibration, thresholds and risk-score weights.
* Imbalance per D-020: class_weight for DT / RF / XGB, no resampling for KNN / MLP.
* The same folds score the default (Phase 5) configuration, so default vs tuned is a paired comparison.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate, learning_curve

from finsight import config
from finsight.ml import classifiers as C

TUNED_MODELS = ["decision_tree", "knn", "random_forest", "xgboost", "mlp"]
IMBALANCE = {"decision_tree": "class_weight", "random_forest": "class_weight", "xgboost": "class_weight",
             "knn": "none", "mlp": "none", "naive_bayes": "none"}
N_ITER = {"random_forest": 20, "xgboost": 20, "mlp": 10}


def search_space(name: str, pos_weight: float) -> tuple[str, dict]:
    """('grid' | 'random', parameter space) with the imblearn Pipeline's `model__` prefix."""
    if name == "decision_tree":
        return "grid", {"model__max_depth": [3, 4, 6, 8, 10, 14, None], "model__min_samples_leaf": [1, 5, 10, 20, 40],
                        "model__criterion": ["gini", "entropy"]}
    if name == "knn":
        return "grid", {"model__n_neighbors": [5, 9, 15, 25, 41, 61], "model__weights": ["uniform", "distance"],
                        "model__p": [1, 2]}
    if name == "random_forest":
        return "random", {"model__n_estimators": [200, 300, 500], "model__max_depth": [None, 8, 12, 20],
                          "model__min_samples_leaf": [1, 2, 4, 8], "model__max_features": ["sqrt", 0.2, 0.35, 0.5],
                          "model__class_weight": ["balanced", "balanced_subsample", None]}
    if name == "xgboost":
        return "random", {"model__n_estimators": randint(150, 700), "model__max_depth": randint(3, 9),
                          "model__learning_rate": loguniform(0.02, 0.3), "model__subsample": uniform(0.6, 0.4),
                          "model__colsample_bytree": uniform(0.4, 0.6), "model__min_child_weight": [1, 3, 5, 10],
                          "model__gamma": [0, 0.5, 1, 2], "model__scale_pos_weight": [1.0, float(np.sqrt(pos_weight)), pos_weight]}
    if name == "mlp":
        return "random", {"model__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
                          "model__alpha": loguniform(1e-5, 1e-1), "model__learning_rate_init": loguniform(3e-4, 3e-3),
                          "model__batch_size": [64, 128, 256]}
    raise KeyError(name)


def folds(strat: np.ndarray, n_splits: int = 5, seed: int = config.SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Positional (train, test) folds over the rows passed in, stratified on risk_label."""
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(np.zeros(len(strat)), strat))


def pos_weight(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def default_pipeline(name: str, exp: str = "C", pw: float = 1.0):
    return C.build(name, exp, IMBALANCE[name], pos_weight=pw)


def tune(name: str, X: pd.DataFrame, y: pd.Series, cv, exp: str = "C", n_jobs: int = -1, seed: int = config.SEED):
    """Run the search; returns (fitted search object, tidy cv_results DataFrame)."""
    pw = pos_weight(y)
    kind, space = search_space(name, pw)
    est = default_pipeline(name, exp, pw)
    common = dict(scoring="average_precision", cv=cv, n_jobs=n_jobs, refit=True, return_train_score=True)
    search = GridSearchCV(est, space, **common) if kind == "grid" else \
        RandomizedSearchCV(est, space, n_iter=N_ITER[name], random_state=seed, **common)
    search.fit(X, y)
    res = pd.DataFrame(search.cv_results_)
    keep = [c for c in res.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score",
                                                                 "mean_train_score", "rank_test_score", "mean_fit_time"]
    res = res[keep].rename(columns=lambda c: c.replace("param_model__", "")).assign(model=name, search=kind)
    return search, res


def default_cv_scores(name: str, X, y, cv, exp: str = "C", n_jobs: int = -1) -> np.ndarray:
    """Per-fold PR-AUC of the untuned (Phase 5 default) configuration on the same folds."""
    r = cross_validate(default_pipeline(name, exp, pos_weight(y)), X, y, cv=cv, scoring="average_precision", n_jobs=n_jobs)
    return r["test_score"]


def tuned_fold_scores(search) -> np.ndarray:
    i = search.best_index_
    return np.array([search.cv_results_[f"split{k}_test_score"][i] for k in range(search.n_splits_)])


def learning_curves(estimator, X, y, cv, sizes=np.linspace(0.1, 1.0, 6), n_jobs: int = -1) -> pd.DataFrame:
    n, tr, te = learning_curve(estimator, X, y, cv=cv, train_sizes=sizes, scoring="average_precision",
                               n_jobs=n_jobs, shuffle=True, random_state=config.SEED)
    return pd.DataFrame({"n_train": n, "train_mean": tr.mean(1), "train_std": tr.std(1),
                         "cv_mean": te.mean(1), "cv_std": te.std(1)})
