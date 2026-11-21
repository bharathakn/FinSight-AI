"""Model zoo (Plan §7.3): DT · Gaussian NB · KNN · RF · XGBoost · MLP, each wrapped in an
imblearn Pipeline:  preprocess → [SMOTE] → model.

Preprocessing
  * trees (DT, RF, XGB): numeric passthrough, one-hot categoricals (scale-invariant models)
  * distance / gradient / Gaussian models (KNN, MLP, NB): signed-log1p → StandardScaler on numerics
    (amounts, durations and distances are heavy-tailed), one-hot categoricals
Imbalance strategies
  * "smote"        — SMOTE inside the pipeline (re-fit per CV fold ⇒ no leakage into validation folds)
  * "class_weight" — cost-sensitive learning where the model supports it (DT, RF, XGB); else none
  * "none"
Hyperparameters here are sensible defaults; tuning is Phase 7.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from finsight import config
from finsight.ml.feature_sets import EXPERIMENTS

MODELS = ["decision_tree", "naive_bayes", "knn", "random_forest", "xgboost", "mlp"]
MODEL_LABELS = {"decision_tree": "Decision Tree", "naive_bayes": "Naive Bayes", "knn": "KNN",
                "random_forest": "Random Forest", "xgboost": "XGBoost", "mlp": "MLP"}
SCALED = {"knn", "mlp", "naive_bayes"}
SUPPORTS_CLASS_WEIGHT = {"decision_tree", "random_forest", "xgboost"}
IMBALANCE = ["none", "class_weight", "smote"]


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def preprocessor(exp: str, scaled: bool) -> ColumnTransformer:
    e = EXPERIMENTS[exp]
    num = make_pipeline(FunctionTransformer(signed_log1p, feature_names_out="one-to-one"), StandardScaler()) \
        if scaled else "passthrough"
    return ColumnTransformer(
        [("num", num, e["numeric"]),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), e["categorical"])],
        verbose_feature_names_out=False,
    )


def estimator(name: str, imbalance: str = "smote", pos_weight: float = 1.0, seed: int = config.SEED):
    cw = "balanced" if imbalance == "class_weight" else None
    if name == "decision_tree":
        return DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight=cw, random_state=seed)
    if name == "naive_bayes":
        return GaussianNB()
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=15, weights="distance")
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                      class_weight=cw, n_jobs=1, random_state=seed)
    if name == "xgboost":
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                             colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
                             scale_pos_weight=pos_weight if imbalance == "class_weight" else 1.0,
                             n_jobs=1, random_state=seed, verbosity=0)
    if name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, max_iter=400,
                             early_stopping=True, n_iter_no_change=15, random_state=seed)
    raise KeyError(name)


def build(name: str, exp: str, imbalance: str = "smote", pos_weight: float = 1.0, seed: int = config.SEED) -> Pipeline:
    steps = [("prep", preprocessor(exp, scaled=name in SCALED))]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    steps.append(("model", estimator(name, imbalance, pos_weight, seed)))
    return Pipeline(steps)


def effective_imbalance(name: str, imbalance: str) -> str:
    """class_weight silently means 'none' for models that don't support it."""
    return "none" if imbalance == "class_weight" and name not in SUPPORTS_CLASS_WEIGHT else imbalance
