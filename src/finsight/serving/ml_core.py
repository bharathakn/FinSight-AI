"""ml-service core (Plan §8.1): classifier, explanations, anomaly detectors, clustering, model comparison.

* P(fraud)      the Phase 7 production model (isotonic-calibrated tuned XGBoost)
* reasons       native TreeSHAP via XGBoost `pred_contribs` on the exact XGBoost inside the calibrated model
                (no `shap` import, D-024); log-odds contributions of the uncalibrated margin
* anomaly       ONNX autoencoder (onnxruntime) → anomaly_pct for the risk score; IF and OC-SVM on /anomaly
* cluster       Phase 6 K-Means (K = 4, Exp C) with the cluster's size, fraud rate and defining features
Serving path (D-034): the fitted sklearn preprocessors are compiled to numpy (`CompiledPreprocessor`), the
calibrated probability is computed as isotonic(XGBoost(Xt)), and XGBoost runs single-threaded per request,
which avoids oversubscribing cores when requests are served concurrently. Results equal the sklearn objects exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from finsight.ml import explain
from finsight.ml.anomaly import OnnxAutoencoder
from finsight.serving.fastpath import CompiledPreprocessor
from finsight.serving.store import CATEGORICAL, FEATURES

DETECTORS = ["isolation_forest", "ocsvm", "autoencoder"]


class MLCore:
    def __init__(self, root: Path):
        root = Path(root)
        fin, an, cl, rep = root / "models" / "final", root / "models" / "anomaly", root / "models" / "clustering", root / "reports"
        self.model = joblib.load(fin / "production_model.joblib")
        inner = getattr(self.model, "calibrated_classifiers_", None)
        self.explainer = inner[0].estimator if inner else self.model     # the XGBoost pipeline being calibrated
        self.metrics = json.loads((fin / "metrics.json").read_text())
        self.threshold = float(self.metrics["decision_threshold"])
        # anomaly
        self.anomaly_meta = json.loads((an / "anomaly_meta.json").read_text())
        exp = self.anomaly_meta["exp"]
        self.an_prep = joblib.load(an / f"prep_exp{exp}.joblib")
        self.an_features = self.anomaly_meta["features"]
        self.iforest = joblib.load(an / f"isolation_forest_exp{exp}.joblib")
        self.ocsvm = joblib.load(an / f"ocsvm_exp{exp}.joblib")
        self.ae = OnnxAutoencoder(an / f"autoencoder_exp{exp}.onnx")
        ref = np.load(an / f"reference_scores_exp{exp}.npz")
        self.refs = {"isolation_forest": ref["isolation_forest"], "ocsvm": ref["ocsvm_nu0.05"], "autoencoder": ref["autoencoder"]}
        th = self.anomaly_meta["thresholds"]
        self.an_thresholds = {"isolation_forest": th["isolation_forest"], "ocsvm": th["ocsvm_nu0.05"], "autoencoder": th["autoencoder"]}
        # clustering
        kb = joblib.load(cl / "kmeans_expC.joblib")
        self.km_prep, self.kmeans = kb["prep"], kb["kmeans"]
        clusters = pd.read_csv(rep / "unsup_clusters.csv").query("exp == 'C' and method == 'kmeans'").set_index("cluster")
        names = list(self.km_prep.get_feature_names_out())
        centers = pd.DataFrame(self.kmeans.cluster_centers_, columns=names)
        self.cluster_info = {int(c): {"cluster": int(c), "size": int(clusters.loc[c, "n"]),
                                      "fraud_rate": float(clusters.loc[c, "fraud_rate"]),
                                      "top_features": [{"feature": f, "z": round(float(centers.loc[c, f]), 2)}
                                                       for f in centers.loc[c].abs().sort_values(ascending=False).index[:3]]}
                             for c in clusters.index}
        # comparison tables
        self.reports = {n: pd.read_csv(rep / n) for n in ("ml_experiment_results.csv", "final_model_comparison.csv",
                                                           "final_tuning_summary.csv", "anomaly_results.csv",
                                                           "final_shap_importance.csv", "final_per_scenario.csv",
                                                           "final_learning_curves.csv", "unsup_clusters.csv",
                                                           "unsup_kmeans_sweep.csv", "ml_paired_tests.csv")}
        self.embedding = pd.read_parquet(root / "data" / "embedding.parquet")
        self.embedding_meta = json.loads((rep / "embedding_meta.json").read_text())
        # fast path
        dtypes = pd.read_parquet(root / "data" / "transactions.parquet", columns=FEATURES).dtypes.to_dict()
        self.fp_prod = CompiledPreprocessor(self.explainer.named_steps["prep"], dtypes)
        self.fp_an = CompiledPreprocessor(self.an_prep, dtypes)
        self.fp_km = CompiledPreprocessor(self.km_prep, dtypes)
        self.xgb = self.explainer.named_steps["model"]
        self.xgb.set_params(n_jobs=1)
        self.booster = self.xgb.get_booster()
        self.booster.set_param({"nthread": 1})
        self.isotonic = inner[0].calibrators[0] if inner else None
        self.out_names = self.fp_prod.feature_names + ["bias"]

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def frame(rows: list[dict], columns=FEATURES) -> pd.DataFrame:
        df = pd.DataFrame(rows).reindex(columns=columns)
        for c in df.columns:
            if c in CATEGORICAL:
                df[c] = df[c].astype("object").where(df[c].notna(), None).astype(str)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        return df

    def _pct(self, det: str, s: np.ndarray) -> np.ndarray:
        ref = self.refs[det]
        return np.searchsorted(ref, s, side="right") / len(ref)

    # ---------------------------------------------------------------- endpoints
    def predict(self, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """(calibrated P(fraud), transformed matrix) via the fast path."""
        Xt = self.fp_prod.transform(rows)
        raw = self.xgb.predict_proba(Xt)[:, 1]
        p = np.clip(self.isotonic.predict(raw), 0.0, 1.0) if self.isotonic is not None \
            else self.model.predict_proba(self.frame(rows))[:, 1]
        return p, Xt

    def score(self, rows: list[dict], explain_top: int = 6) -> list[dict]:
        import xgboost as xgb

        p, Xt = self.predict(rows)
        ae = self.ae.score_samples(self.fp_an.transform(rows).astype(np.float32))
        ae_pct = self._pct("autoencoder", ae)
        clusters = self.kmeans.predict(self.fp_km.transform(rows))
        contrib = (pd.DataFrame(self.booster.predict(xgb.DMatrix(Xt), pred_contribs=True), columns=self.out_names)
                   if explain_top else None)
        out = []
        for i in range(len(rows)):
            r = {"p_fraud": float(p[i]), "flagged": bool(p[i] >= self.threshold), "threshold": self.threshold,
                 "anomaly": {"detector": "autoencoder", "error": float(ae[i]), "pct": float(ae_pct[i]),
                             "above_p95": bool(ae[i] >= self.an_thresholds["autoencoder"]["p95"])},
                 "cluster": self.cluster_info.get(int(clusters[i]), {"cluster": int(clusters[i])})}
            if contrib is not None:
                r["reasons"] = explain.top_reasons(contrib.iloc[i], pd.Series(rows[i]), k=explain_top)
                r["base_log_odds"] = float(contrib.iloc[i]["bias"])
            out.append(r)
        return out

    def anomaly(self, rows: list[dict]) -> list[dict]:
        Xa = self.fp_an.transform(rows).astype(np.float32)
        raw = {"isolation_forest": -self.iforest.score_samples(Xa), "ocsvm": -self.ocsvm.decision_function(Xa),
               "autoencoder": self.ae.score_samples(Xa)}
        out = []
        for i in range(len(rows)):
            out.append({d: {"score": float(raw[d][i]), "pct": float(self._pct(d, raw[d][i:i + 1])[0]),
                            "above_p95": bool(raw[d][i] >= self.an_thresholds[d]["p95"]),
                            "above_p99": bool(raw[d][i] >= self.an_thresholds[d]["p99"])} for d in DETECTORS})
        return out

    def compare(self) -> dict:
        return {"production": {k: self.metrics[k] for k in ("production_model", "decision_threshold", "test",
                                                            "candidates_val_pr_auc", "artifact_sizes_mb")},
                "phase5_grid": self.reports["ml_experiment_results.csv"].to_dict(orient="records"),
                "phase7_final": self.reports["final_model_comparison.csv"].to_dict(orient="records"),
                "phase7_tuning": self.reports["final_tuning_summary.csv"].to_dict(orient="records"),
                "phase6_anomaly": self.reports["anomaly_results.csv"].to_dict(orient="records"),
                "phase5_paired_tests": self.reports["ml_paired_tests.csv"].to_dict(orient="records"),
                "shap_importance": self.reports["final_shap_importance.csv"].head(20).to_dict(orient="records"),
                "per_scenario": self.reports["final_per_scenario.csv"].to_dict(orient="records"),
                "learning_curves": self.reports["final_learning_curves.csv"].to_dict(orient="records")}

    def clusters(self, sample: int = 4000) -> dict:
        """K-Means selection sweep, cluster profiles and a 2-D PCA embedding (stratified sample keeps all non-legit rows)."""
        e = self.embedding
        legit = e[e["risk_label"] == "LEGITIMATE"]
        rest = e[e["risk_label"] != "LEGITIMATE"]
        n_legit = max(0, min(len(legit), sample - len(rest)))
        pts = pd.concat([legit.sample(n_legit, random_state=42), rest]).sort_index()
        return {"sweep": self.reports["unsup_kmeans_sweep.csv"].to_dict(orient="records"),
                "clusters": self.reports["unsup_clusters.csv"].to_dict(orient="records"),
                "profiles": list(self.cluster_info.values()),
                "embedding": {"method": self.embedding_meta["method"], "explained_variance": self.embedding_meta["explained_variance"],
                              "points": pts.round(4).to_dict(orient="records"), "sampled_from": int(len(e)),
                              "note": "all FRAUDULENT and BORDERLINE rows + a random sample of LEGITIMATE rows"},
                "detectors": self.reports["anomaly_results.csv"].to_dict(orient="records")}

    def health(self) -> dict:
        return {"production_model": self.metrics["production_model"], "features": len(FEATURES),
                "detectors": DETECTORS, "clusters": len(self.cluster_info)}
