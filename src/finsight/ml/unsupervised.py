"""Phase 6 experiment runner: unsupervised learning + anomaly detection (Plan §7.4–7.5, RQ5, RQ6).

Experiments (v1.1, same 60/20/20 stratified split as Phase 5)
  1. clustering      K-Means K = 2..8 on scaled Exp B and Exp C (all 10 K rows, labels unused);
                     DBSCAN (ε from the k-distance knee); PCA + t-SNE projections (Exp C);
                     post hoc: fraud rate / scenario mix per cluster, ARI + NMI vs risk_label
  2. anomaly         IF · OC-SVM (ν 0.01, 0.05) · autoencoder on Exp B and Exp C, fitted on LEGIT train rows;
                     thresholds = 95th / 99th pct of legit validation scores; scored once on test
  3. per scenario    flag rate per scenario at the p95 threshold, plus the Phase 5 XGBoost for reference
  4. complementarity of the test frauds XGBoost misses, how many does each detector catch?
  5. PaySim scale    Isolation Forest on 2.77 M TRANSFER/CASH_OUT rows (Polars), evaluated against isFraud
Artifacts → artifacts/models/anomaly/ (prep, IF, OC-SVM, AE .onnx + .pt, reference scores),
            artifacts/models/clustering/ (prep + K-Means)
Reports   → reports/unsup_*.csv, anomaly_*.csv, unsupervised_summary.json, figures/unsup_*.png
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from finsight import config
from finsight.eda import load_master
from finsight.ml import anomaly as A
from finsight.ml import clustering as Cl
from finsight.ml.evaluator import per_scenario
from finsight.ml.feature_sets import feature_columns
from finsight.ml.splits import stratified_split

EXPS = ["B", "C"]
DBSCAN_MIN_SAMPLES = 10
MAIN_EXP = "C"
ONNX_TOL = 1e-5


def run_clustering(m: pd.DataFrame, with_tsne: bool = True) -> dict:
    out = {"sweeps": [], "clusters": [], "summary": {}}
    for exp in EXPS:
        X, prep = Cl.scaled_matrix(m, exp)
        sw = Cl.kmeans_sweep(X).assign(exp=exp)
        k = Cl.pick_k(sw)
        km = Cl.fit_kmeans(X, k)
        desc = Cl.describe_clusters(m, km.labels_).assign(exp=exp, method="kmeans")
        db, eps, kd = Cl.fit_dbscan(X, DBSCAN_MIN_SAMPLES)
        noise = db.labels_ == -1
        y = m["is_fraud"].to_numpy()
        lab = m["risk_label"].astype(str).to_numpy()
        out["sweeps"].append(sw)
        out["clusters"].append(desc)
        out["summary"][exp] = {
            "n_features_after_encoding": int(X.shape[1]), "k": k,
            "silhouette": float(sw.set_index("k").loc[k, "silhouette"]),
            "davies_bouldin": float(sw.set_index("k").loc[k, "davies_bouldin"]),
            "kmeans_vs_label": Cl.agreement(m, km.labels_),
            "flag_high_fraud_clusters": Cl.best_fraud_cluster_scores(m, km.labels_),
            "profile": {str(c): v for c, v in Cl.cluster_profile(km, prep.get_feature_names_out()).items()},
            "dbscan": {"min_samples": DBSCAN_MIN_SAMPLES, "eps": eps, "clusters": int(db.labels_.max() + 1),
                       "noise_rate": float(noise.mean()), "noise_n": int(noise.sum()),
                       "fraud_rate_noise": float(y[noise].mean()), "fraud_rate_clustered": float(y[~noise].mean()),
                       "noise_rate_by_label": {l: float(noise[lab == l].mean()) for l in ("LEGITIMATE", "FRAUDULENT", "BORDERLINE")},
                       "vs_label": Cl.agreement(m, db.labels_)},
        }
        if exp == MAIN_EXP:
            out["main"] = {"X": X, "kmeans": km, "prep": prep, "kdist": kd, "eps": eps, "dbscan_labels": db.labels_,
                           "proj": Cl.projections(X, tsne=with_tsne)}
    out["sweeps"] = pd.concat(out["sweeps"], ignore_index=True)
    out["clusters"] = pd.concat(out["clusters"], ignore_index=True)
    return out


def _xgb_reference(m: pd.DataFrame, sp) -> tuple[np.ndarray, float]:
    model = joblib.load(config.ARTIFACTS_DIR / "models" / "phase5" / "xgboost_expC_v1_1.joblib")
    res = pd.read_csv(config.REPORTS_DIR / "ml_experiment_results.csv")
    thr = float(res.query("version=='v1_1' and model=='xgboost' and exp=='C'")["test_threshold"].iloc[0])
    X = m[feature_columns("C")].iloc[sp.test]
    return model.predict_proba(X)[:, 1], thr


def run_anomaly(m: pd.DataFrame, sp) -> dict:
    y = m["is_fraud"].to_numpy()
    bl = (m["risk_label"].astype(str) == "BORDERLINE").to_numpy()
    te = m.iloc[sp.test]
    rows, scen, scores, dets_main, hist_main = [], [], {}, None, None
    for exp in EXPS:
        dets, hist = A.fit_detectors(m, sp, exp)
        for name, d in dets.items():
            s = d.score(te)
            thr = {f"p{p}": d.threshold(p) for p in A.PERCENTILES}
            rows.append({"exp": exp, "detector": name, **A.evaluate(y[sp.test], s, thr, bl[sp.test])})
            if exp == MAIN_EXP:
                scores[name] = s
                scen.append(per_scenario(te, s, thr["p95"]).assign(detector=name))
        if exp == MAIN_EXP:
            dets_main, hist_main = dets, hist
    xgb_p, xgb_thr = _xgb_reference(m, sp)
    scen.append(per_scenario(te, xgb_p, xgb_thr).assign(detector="xgboost_supervised"))
    results = pd.DataFrame(rows)

    # complementarity: test frauds XGBoost misses, caught by each detector at p95
    yt = y[sp.test].astype(bool)
    missed = yt & (xgb_p < xgb_thr)
    comp = {"xgb_threshold": xgb_thr, "test_frauds": int(yt.sum()), "xgb_missed": int(missed.sum()),
            "caught_by": {n: int((missed & (scores[n] >= dets_main[n].threshold(95))).sum()) for n in scores},
            "missed_scenarios": te.loc[missed, "scenario"].value_counts().to_dict()}
    # rank correlation between detector scores and the supervised probability on test
    corr = pd.DataFrame(scores | {"xgboost": xgb_p}).corr(method="spearman")["xgboost"].drop("xgboost").to_dict()
    return {"results": results, "per_scenario": pd.concat(scen, ignore_index=True), "scores": scores,
            "xgb_proba": xgb_p, "xgb_thr": xgb_thr, "detectors": dets_main, "ae_history": hist_main,
            "complementarity": comp, "spearman_vs_xgb": corr}


def save_artifacts(dets: dict, km_bundle: dict, X_check: np.ndarray) -> dict:
    """Persist serving artifacts and verify ONNX ≡ PyTorch on `X_check`."""
    import torch

    d_an = config.ARTIFACTS_DIR / "models" / "anomaly"
    d_cl = config.ARTIFACTS_DIR / "models" / "clustering"
    d_an.mkdir(parents=True, exist_ok=True)
    d_cl.mkdir(parents=True, exist_ok=True)
    ae = dets["autoencoder"]
    joblib.dump(ae.prep, d_an / f"prep_exp{ae.exp}.joblib")
    joblib.dump(dets["isolation_forest"].model, d_an / f"isolation_forest_exp{ae.exp}.joblib")
    joblib.dump(dets["ocsvm_nu0.05"].model, d_an / f"ocsvm_exp{ae.exp}.joblib")
    torch.save(ae.model.state_dict(), d_an / f"autoencoder_exp{ae.exp}.pt")
    onnx_path = A.export_onnx(ae.model, X_check.shape[1], d_an / f"autoencoder_exp{ae.exp}.onnx")
    np.savez(d_an / f"reference_scores_exp{ae.exp}.npz", **{n: d.reference for n, d in dets.items()})
    meta = {"exp": ae.exp, "features": feature_columns(ae.exp), "input_dim": int(X_check.shape[1]),
            "score_direction": "higher = more anomalous",
            "thresholds": {n: {f"p{p}": d.threshold(p) for p in A.PERCENTILES} for n, d in dets.items()},
            "reference": "legit validation rows (v1.1 stratified split)"}
    (d_an / "anomaly_meta.json").write_text(json.dumps(meta, indent=2))
    joblib.dump({"prep": km_bundle["prep"], "kmeans": km_bundle["kmeans"], "exp": MAIN_EXP}, d_cl / f"kmeans_exp{MAIN_EXP}.joblib")

    r_t, e_t = A.torch_ae_score(ae.model, X_check)
    r_o, e_o = A.OnnxAutoencoder(onnx_path).run(X_check)
    parity = {"max_abs_diff_reconstruction": float(np.abs(r_t - r_o).max()),
              "max_abs_diff_error": float(np.abs(e_t - e_o).max()), "rows": int(len(X_check)),
              "onnx_bytes": onnx_path.stat().st_size}
    parity["pass"] = parity["max_abs_diff_reconstruction"] < ONNX_TOL and parity["max_abs_diff_error"] < ONNX_TOL
    return parity


def run_all(paysim: bool = True, with_tsne: bool = True) -> dict:
    from finsight.ml import unsupervised_figures as F

    t0 = time.perf_counter()
    rep = config.REPORTS_DIR
    m = load_master("v1_1")
    sp = stratified_split(m)

    cl = run_clustering(m, with_tsne=with_tsne)
    cl["sweeps"].to_csv(rep / "unsup_kmeans_sweep.csv", index=False)
    cl["clusters"].to_csv(rep / "unsup_clusters.csv", index=False)

    an = run_anomaly(m, sp)
    an["results"].to_csv(rep / "anomaly_results.csv", index=False)
    an["per_scenario"].to_csv(rep / "anomaly_per_scenario.csv", index=False)
    an["ae_history"].to_csv(rep / "anomaly_ae_history.csv", index=False)

    X_test = an["detectors"]["autoencoder"].transform(m.iloc[sp.test])
    parity = save_artifacts(an["detectors"], cl["main"], X_test)
    if not parity["pass"]:
        raise AssertionError(f"ONNX parity failed: {parity}")

    ps = A.paysim_isolation_forest() if paysim else None
    if ps is not None:
        (rep / "anomaly_paysim_if.json").write_text(json.dumps({k: v for k, v in ps.items() if not k.startswith("_")}, indent=2))

    res = an["results"]
    best = res[res.exp == MAIN_EXP].sort_values("pr_auc", ascending=False).iloc[0]
    summary = {
        "clustering": cl["summary"],
        "anomaly": {
            "train_rows_legit": int((m["risk_label"].astype(str).to_numpy()[sp.train] == "LEGITIMATE").sum()),
            "reference_rows_legit_val": int((m["risk_label"].astype(str).to_numpy()[sp.val] == "LEGITIMATE").sum()),
            "test_rows": len(sp.test),
            "ae_epochs": int(len(an["ae_history"])),
            "ae_best_val_loss": float(an["ae_history"]["val_loss"].min()),
            "best_detector_expC": {"detector": best.detector, **{k: round(float(best[k]), 4) for k in
                                   ("pr_auc", "roc_auc", "recall_at_1pct_fpr", "p95_precision", "p95_recall",
                                    "p95_borderline_flag_rate")}},
            "results": res.round(4).to_dict(orient="records"),
            "complementarity": an["complementarity"],
            "spearman_vs_xgb": {k: round(v, 3) for k, v in an["spearman_vs_xgb"].items()},
        },
        "onnx_parity": parity,
        "paysim_if": {k: v for k, v in ps.items() if not k.startswith("_")} if ps else None,
        "runtime_seconds": None,
    }
    F.render_all(m=m, sp=sp, cl=cl, an=an, ps=ps, summary=summary)
    summary["runtime_seconds"] = round(time.perf_counter() - t0, 1)
    (rep / "unsupervised_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
