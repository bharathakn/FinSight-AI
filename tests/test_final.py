import json

import numpy as np
import pandas as pd
import pytest

from finsight import config
from finsight.ml import tuner as T
from finsight.scoring import risk_scorer as RS

FINAL = config.ARTIFACTS_DIR / "models" / "final"
MASTER = config.CURATED_DIR / "master_features_v1_1.parquet"


def test_simplex_grid_sums_to_one():
    pts = list(RS.simplex(0.25))
    assert len(pts) == 15 and all(abs(p.sum() - 1) < 1e-12 and (p >= 0).all() for p in pts)


def test_search_spaces_use_pipeline_prefix_and_d020_imbalance():
    for n in T.TUNED_MODELS:
        kind, space = T.search_space(n, 5.0)
        assert kind in ("grid", "random") and all(k.startswith("model__") for k in space)
        steps = list(T.default_pipeline(n).named_steps)
        assert "smote" not in steps                       # D-020: no SMOTE from Phase 7 on


def test_risk_scorer_overrides_and_tiers():
    s = RS.RiskScorer(weights={"model": 0.8, "anomaly": 0.1, "graph": 0.1},
                      cutoffs={"MEDIUM": 20.0, "HIGH": 50.0, "CRITICAL": 75.0})
    p = np.array([0.0, 0.0, 0.5, 0.95, 0.0])
    r = s.score(p, np.zeros(5), np.zeros(5), pep=[False, True, False, False, False],
                structuring=[False, False, True, False, False], sanctions=[0, 0, 0, 0, 95])
    assert list(r["tier"]) == ["LOW", "MEDIUM", "HIGH", "CRITICAL", "CRITICAL"]
    assert r["score"][2] == pytest.approx(50.0)                    # 40 + structuring bonus 10
    assert r["override_pep"].tolist() == [False, True, False, False, False]


def test_fit_cutoffs_are_ordered_and_weights_prefer_signal():
    rng = np.random.default_rng(0)
    y = rng.random(2000) < 0.15
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.15, 2000), 0, 1)
    noise = rng.random(2000)
    s = RS.RiskScorer()
    s.fit_weights(p, noise, noise, y)
    assert s.weights["model"] >= 0.8
    c = s.fit_cutoffs(s.combine(p, noise, noise), y)
    assert c["MEDIUM"] <= c["HIGH"] <= c["CRITICAL"]


def test_graph_risk_takes_strongest_flag():
    flags = pd.DataFrame({n: [False, False] for n in RS.NETWORK_ITEMS})
    flags.loc[0, ["PASS_THROUGH", "BURST_SENDER"]] = True
    s = RS.RiskScorer(flag_rates={"PASS_THROUGH": 0.99, "BURST_SENDER": 0.3})
    assert s.graph_risk(flags).tolist() == [0.99, 0.0]


def test_scorer_json_roundtrip(tmp_path):
    s = RS.RiskScorer(flag_rates={"PASS_THROUGH": 0.9}, fit_info={"x": 1})
    assert RS.RiskScorer.load(s.save(tmp_path / "r.json")) == s


@pytest.mark.skipif(not (FINAL / "metrics.json").exists(), reason="run make pipeline STAGES=train_final")
def test_final_artifacts_reproduce_test_metrics():
    import joblib

    from finsight.eda import load_master
    from finsight.ml.evaluator import scores
    from finsight.ml.feature_sets import get_xy
    from finsight.ml.splits import stratified_split

    metrics = json.loads((FINAL / "metrics.json").read_text())
    m = load_master("v1_1")
    sp = stratified_split(m)
    X, y = get_xy(m, "C")
    model = joblib.load(FINAL / "production_model.joblib")
    p = model.predict_proba(X.iloc[sp.test])[:, 1]
    s = scores(y.iloc[sp.test], p, metrics["decision_threshold"])
    assert s["pr_auc"] == pytest.approx(metrics["test"]["test_pr_auc"], abs=1e-9)
    assert s["f1"] == pytest.approx(metrics["test"]["test_f1"], abs=1e-9)
    assert RS.RiskScorer.load(FINAL / "risk_scorer.json").weights == metrics["risk_scorer"]["weights"]


@pytest.mark.skipif(not (FINAL / "xgboost_explainer.joblib").exists(), reason="run make pipeline STAGES=train_final")
def test_native_contributions_match_shap():
    import joblib

    from finsight.eda import load_master
    from finsight.ml import explain
    from finsight.ml.feature_sets import get_xy

    m = load_master("v1_1")
    X, _ = get_xy(m, "C")
    r = explain.parity(joblib.load(FINAL / "xgboost_explainer.joblib"), X.iloc[:300])
    assert r["max_abs_diff_vs_shap"] < 1e-3 and r["max_additivity_error"] < 1e-3
