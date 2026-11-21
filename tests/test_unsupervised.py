import numpy as np
import pandas as pd
import pytest

from finsight import config
from finsight.ml import anomaly as A
from finsight.ml import clustering as Cl

MASTER = config.CURATED_DIR / "master_features_v1_1.parquet"


def test_knee_finds_elbow_of_convex_curve():
    y = np.concatenate([np.linspace(0, 1, 90), np.linspace(1.5, 10, 10)])
    assert 85 <= Cl.knee(y) <= 92


def test_pick_k_uses_db_within_silhouette_plateau():
    sw = pd.DataFrame({"k": [2, 3, 4, 5], "silhouette": [0.161, 0.150, 0.158, 0.10],
                       "davies_bouldin": [3.5, 2.0, 2.4, 1.0]})
    assert Cl.pick_k(sw) == 4            # k=3 is outside the plateau, k=5 far below it


def test_evaluate_and_recall_at_fpr():
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.linspace(0, 1, 100), np.full(10, 2.0)])
    r = A.evaluate(y, s, {"p95": 0.95})
    assert r["pr_auc"] == pytest.approx(1.0) and r["recall_at_1pct_fpr"] == 1.0
    assert r["p95_recall"] == 1.0 and 0.6 < r["p95_precision"] < 1.0


def test_autoencoder_shape_and_onnx_parity(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12)).astype(np.float32)
    model, hist = A.train_autoencoder(X[:240], X[240:], max_epochs=5, patience=2)
    assert [l.out_features for l in model if hasattr(l, "out_features")] == [32, 16, 8, 16, 32, 12]
    path = A.export_onnx(model, 12, tmp_path / "ae.onnx")
    r_t, e_t = A.torch_ae_score(model, X)
    r_o, e_o = A.OnnxAutoencoder(path).run(X)
    assert np.abs(r_t - r_o).max() < 1e-5 and np.abs(e_t - e_o).max() < 1e-5
    assert np.allclose(e_t, ((X - r_t) ** 2).mean(axis=1), atol=1e-6)


def test_anomaly_pct_is_monotone():
    d = A.Detector("isolation_forest", "C", None, None, reference=np.sort(np.arange(100.0)))
    p = d.anomaly_pct(np.array([-1.0, 49.5, 1000.0]))
    assert p[0] == 0 and p[2] == 1 and 0.45 < p[1] < 0.55


@pytest.mark.skipif(not MASTER.exists(), reason="run make pipeline")
def test_detectors_fit_on_legit_training_rows_only():
    from finsight.eda import load_master
    from finsight.ml.splits import stratified_split

    m = load_master("v1_1")
    sp = stratified_split(m)
    dets, _ = A.fit_detectors(m, sp, "B", ocsvm_nus=(0.05,))
    n_legit_train = int((m["risk_label"].astype(str).to_numpy()[sp.train] == "LEGITIMATE").sum())
    assert dets["isolation_forest"].model.n_features_in_ == dets["autoencoder"].transform(m.iloc[:1]).shape[1]
    assert dets["ocsvm_nu0.05"].model.fit_status_ == 0
    assert dets["isolation_forest"].model._max_samples <= n_legit_train
    n_legit_val = int((m["risk_label"].astype(str).to_numpy()[sp.val] == "LEGITIMATE").sum())
    assert all(len(d.reference) == n_legit_val for d in dets.values())
    # detectors must beat chance on the test set
    s = dets["autoencoder"].score(m.iloc[sp.test])
    assert A.evaluate(m["is_fraud"].to_numpy()[sp.test], s, {})["roc_auc"] > 0.7


def test_paysim_features_lazy_plan():
    import polars as pl

    lf = pl.LazyFrame({"step": [1, 25], "type": ["TRANSFER", "PAYMENT"], "amount": [100.0, 5.0],
                       "oldbalanceOrg": [100.0, 5.0], "newbalanceOrig": [0.0, 0.0], "oldbalanceDest": [0.0, 0.0],
                       "newbalanceDest": [0.0, 5.0], "isFraud": [1, 0]})
    df = A.paysim_features(lf).collect()
    assert len(df) == 1 and df["is_transfer"][0] == 1 and df["slog_err_orig"][0] == 0.0
    assert df["slog_err_dest"][0] == pytest.approx(np.log1p(100.0))
