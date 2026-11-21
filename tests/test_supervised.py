import numpy as np
import pandas as pd
import pytest

from finsight import config
from finsight.ml import classifiers as C
from finsight.ml.evaluator import best_f1_threshold, paired_test, scores

MASTER = config.CURATED_DIR / "master_features_v1_1.parquet"


def test_best_f1_threshold_and_scores():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    thr = best_f1_threshold(y, p)
    assert 0.6 < thr <= 0.7
    s = scores(y, p, thr)
    assert s["f1"] == 1.0 and s["tp"] == 2 and s["fp"] == 0


def test_paired_test_direction():
    a = np.full(10, 0.80); b = a + np.linspace(0.01, 0.05, 10)
    r = paired_test(a, b)
    assert r["mean_diff"] > 0 and r["wins"] == 10 and r["t_pvalue"] < 0.01


@pytest.mark.parametrize("name", C.MODELS)
def test_pipeline_structure(name):
    p = C.build(name, "C", "smote")
    assert list(p.named_steps) == ["prep", "smote", "model"]          # SMOTE after prep, inside the pipeline
    assert "smote" not in C.build(name, "C", "none").named_steps
    if name not in C.SUPPORTS_CLASS_WEIGHT:
        assert C.effective_imbalance(name, "class_weight") == "none"


@pytest.mark.skipif(not MASTER.exists(), reason="run make pipeline")
def test_splits_disjoint_and_stratified():
    from finsight.eda import load_master
    from finsight.ml.splits import cv_folds, stratified_split, temporal_split

    m = load_master("v1_1")
    sp = stratified_split(m)
    assert sp.sizes() == {"train": 6000, "val": 2000, "test": 2000}
    sets = [set(sp.train), set(sp.val), set(sp.test)]
    assert all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])
    rate = lambda idx: m["is_fraud"].iloc[idx].mean()
    assert abs(rate(sp.train) - rate(sp.test)) < 0.005
    folds = cv_folds(m, sp.dev)
    assert len(folds) == 10 and all(not set(te) & set(sp.test) for _, te in folds)   # test never in CV
    t = temporal_split(m)
    assert m["timestamp"].iloc[t.train].max() < m["timestamp"].iloc[t.test].min()


@pytest.mark.skipif(not MASTER.exists(), reason="run make pipeline")
def test_model_never_sees_leakage_columns():
    from finsight.eda import load_master
    from finsight.ml.feature_sets import get_xy

    m = load_master("v1_1")
    X, y = get_xy(m, "C")
    pipe = C.build("decision_tree", "C", "none").fit(X.head(500), y.head(500))
    feats = set(pipe.named_steps["prep"].get_feature_names_out())
    assert not feats & config.LEAKAGE_COLUMNS
    assert not any(f.startswith(tuple(config.ID_COLUMNS)) for f in feats)
