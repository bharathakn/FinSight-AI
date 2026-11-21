"""Phase 7 runner: tuning, stacking, calibration, risk score, explanations (Plan §7.6–7.9, RQ4).

Protocol (v1.1, Exp C, the same stratified 60/20/20 split as Phases 5–6)
  train (60%)  hyperparameter search (5-fold stratified CV, PR-AUC); learning curves; stack + calibration fit;
               network-flag fraud rates for graph_risk
  val   (20%)  choice of production model; decision threshold; risk-score weights and tier cut-offs;
               permutation importance
  test  (20%)  scored ONCE at the end for every final candidate; nothing is tuned on it
Artifacts → artifacts/models/final/ (production model, XGBoost explainer, risk_scorer.json, metrics.json);
            reloaded at the end and checked to reproduce the test metrics.
Reports   → reports/final_*.csv|json, reports/figures/final_*.png
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
from finsight.ml import ensemble as E
from finsight.ml import explain as X_
from finsight.ml import tuner as T
from finsight.ml.evaluator import best_f1_threshold, paired_test, per_scenario, scores
from finsight.ml.feature_sets import get_xy
from finsight.ml.splits import stratified_split
from finsight.scoring import risk_scorer as RS

EXP = "C"
FINAL_DIR = config.ARTIFACTS_DIR / "models" / "final"
CANDIDATES = ["xgboost_calibrated", "stack_calibrated"]


def anomaly_component(m: pd.DataFrame) -> np.ndarray:
    """anomaly_pct for every row from the Phase 6 serving artifacts (ONNX autoencoder + legit reference)."""
    d = config.ARTIFACTS_DIR / "models" / "anomaly"
    meta = json.loads((d / "anomaly_meta.json").read_text())
    prep = joblib.load(d / f"prep_exp{meta['exp']}.joblib")
    ae = A.OnnxAutoencoder(d / f"autoencoder_exp{meta['exp']}.onnx")
    ref = np.load(d / f"reference_scores_exp{meta['exp']}.npz")["autoencoder"]
    err = ae.score_samples(prep.transform(m[meta["features"]]).astype(np.float32))
    return np.searchsorted(ref, err, side="right") / len(ref)


def _tier_table(frame: pd.DataFrame, tiers: np.ndarray) -> pd.DataFrame:
    t = pd.crosstab(frame["risk_label"].astype(str).to_numpy(), tiers, normalize="index").reindex(columns=RS.TIERS, fill_value=0)
    n = pd.crosstab(frame["risk_label"].astype(str).to_numpy(), tiers).reindex(columns=RS.TIERS, fill_value=0)
    return t.add_suffix("_share").join(n.add_suffix("_n")).rename_axis("risk_label").reset_index()


def run_tuning(Xtr, ytr, cv) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    searches, results, summary = {}, [], []
    for name in T.TUNED_MODELS:
        t0 = time.perf_counter()
        s, res = T.tune(name, Xtr, ytr, cv, EXP)
        d, tu = T.default_cv_scores(name, Xtr, ytr, cv, EXP), T.tuned_fold_scores(s)
        searches[name] = s
        results.append(res)
        summary.append({"model": name, "search": res["search"].iloc[0], "configs_tried": len(res),
                        "default_cv_pr_auc": d.mean(), "default_cv_std": d.std(ddof=1),
                        "tuned_cv_pr_auc": tu.mean(), "tuned_cv_std": tu.std(ddof=1),
                        **{f"paired_{k}": v for k, v in paired_test(d, tu).items() if k in ("mean_diff", "wins", "t_pvalue")},
                        "seconds": round(time.perf_counter() - t0, 1),
                        "best_params": json.dumps({k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                                                   for k, v in s.best_params_.items()}, default=str)})
    return searches, pd.concat(results, ignore_index=True), pd.DataFrame(summary)


def run_all() -> dict:
    from finsight.ml import final_figures as F

    t_start = time.perf_counter()
    rep = config.REPORTS_DIR
    m = load_master("v1_1")
    sp = stratified_split(m)
    X, y = get_xy(m, EXP)
    yv = y.to_numpy()
    Xtr, ytr, Xva, yva, Xte, yte = X.iloc[sp.train], y.iloc[sp.train], X.iloc[sp.val], yv[sp.val], X.iloc[sp.test], yv[sp.test]
    cv = T.folds(m["risk_label"].astype(str).to_numpy()[sp.train])

    # 1. tuning (train, 5-fold CV)
    searches, tuning_res, tuning_sum = run_tuning(Xtr, ytr, cv)
    tuning_res.to_csv(rep / "final_tuning_results.csv", index=False)
    tuning_sum.to_csv(rep / "final_tuning_summary.csv", index=False)
    tuned = {n: s.best_estimator_ for n, s in searches.items()}

    # 2. learning curves (bias–variance): tuned XGBoost vs tuned decision tree
    lc = pd.concat([T.learning_curves(tuned[n], Xtr, ytr, cv).assign(model=n) for n in ("xgboost", "decision_tree")],
                   ignore_index=True)
    lc.to_csv(rep / "final_learning_curves.csv", index=False)

    # 3. stacking + calibration (train only)
    t0 = time.perf_counter()
    stack_cal = E.calibrated(E.build_stack(tuned)).fit(Xtr, ytr)
    xgb_cal = E.calibrated(tuned["xgboost"]).fit(Xtr, ytr)
    stack_raw = E.uncalibrated_inner(stack_cal)
    fit_seconds = round(time.perf_counter() - t0, 1)

    models = {**{n: tuned[n] for n in T.TUNED_MODELS}, "stack_uncalibrated": stack_raw,
              "xgboost_calibrated": xgb_cal, "stack_calibrated": stack_cal}
    p_val = {k: mdl.predict_proba(Xva)[:, 1] for k, mdl in models.items()}

    # 4. production model + threshold chosen on validation
    val_rank = {k: float(scores(yva, p_val[k])["pr_auc"]) for k in CANDIDATES}
    production = max(val_rank, key=val_rank.get)
    thr = best_f1_threshold(yva, p_val[production])

    # calibration diagnostics on validation
    calib = {}
    for k in ("stack_uncalibrated", "stack_calibrated", "xgboost", "xgboost_calibrated"):
        curve, brier = E.reliability(yva, p_val[k])
        calib[k] = {"brier": brier, "ece": E.expected_calibration_error(yva, p_val[k]), "curve": curve}

    # 5. risk score (components; weights + cut-offs on validation)
    flags = RS.network_flags(m)
    scorer = RS.RiskScorer(flag_rates=RS.flag_rates(flags.iloc[sp.train], ytr))
    a_all = anomaly_component(m)
    g_all = scorer.graph_risk(flags)
    weight_grid = scorer.fit_weights(p_val[production], a_all[sp.val], g_all[sp.val], yva)
    scorer.fit_cutoffs(scorer.combine(p_val[production], a_all[sp.val], g_all[sp.val]), yva)
    scorer.fit_info |= {"production_model": production, "decision_threshold_probability": thr}
    weight_grid.to_csv(rep / "final_risk_weight_grid.csv", index=False)

    # 6. TEST — scored once
    p_test = {k: mdl.predict_proba(Xte)[:, 1] for k, mdl in models.items()}
    te = m.iloc[sp.test]
    rows = []
    for k in models:
        t_k = best_f1_threshold(yva, p_val[k])
        s_te = scores(yte, p_test[k], t_k)
        rows.append({"model": k, "val_pr_auc": float(scores(yva, p_val[k])["pr_auc"]), "val_threshold": t_k,
                     **{f"test_{a}": b for a, b in s_te.items()}})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(rep / "final_model_comparison.csv", index=False)

    pep = te["sender_pep_flag"].to_numpy().astype(bool)
    struct = RS.structuring_fired(flags.iloc[sp.test])
    risk = scorer.score(p_test[production], a_all[sp.test], g_all[sp.test], pep=pep, structuring=struct)
    risk_thr = scorer.cutoffs["HIGH"]
    risk_scores = {"base": scores(yte, risk["base"], risk_thr), "final": scores(yte, risk["score"], risk_thr)}
    tiers = _tier_table(te, risk["tier"])
    tiers.to_csv(rep / "final_risk_tiers.csv", index=False)
    scen = per_scenario(te, p_test[production], thr)
    scen.to_csv(rep / "final_per_scenario.csv", index=False)

    # 7. explanations (tuned XGBoost; serve-time via native pred_contribs)
    sv, Xt_te, ev, names = X_.shap_values(tuned["xgboost"], Xte)
    shap_imp = X_.global_importance(sv, names)
    shap_imp.to_csv(rep / "final_shap_importance.csv", index=False)
    contrib_parity = X_.parity(tuned["xgboost"], Xte)
    perm = X_.permutation(models[production], Xva, yva)
    perm.to_csv(rep / "final_permutation_importance.csv", index=False)
    # one caught fraud and one BORDERLINE false alarm for the local-explanation figure
    lab = te["risk_label"].astype(str).to_numpy()
    flag = p_test[production] >= thr
    ex_fraud = int(np.argmax(np.where((yte == 1) & flag, p_test[production], -1)))
    bl = np.where((lab == "BORDERLINE") & flag)[0]
    ex_border = int(bl[np.argmax(p_test[production][bl])]) if len(bl) else None
    examples = {"fraud": ex_fraud, "borderline": ex_border}
    contrib_te = X_.contributions(tuned["xgboost"], Xte)
    reasons = {k: {"transaction_id": str(te.iloc[i]["transaction_id"]), "scenario": str(te.iloc[i]["scenario"]),
                   "risk_label": lab[i], "p_fraud": float(p_test[production][i]), "risk_score": float(risk["score"][i]),
                   "tier": str(risk["tier"][i]), "reasons": X_.top_reasons(contrib_te.iloc[i], Xte.iloc[i])}
               for k, i in examples.items() if i is not None}

    # 8. artifacts + reload check
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[production], FINAL_DIR / "production_model.joblib", compress=3)
    joblib.dump(tuned["xgboost"], FINAL_DIR / "xgboost_explainer.joblib", compress=3)
    scorer.save(FINAL_DIR / "risk_scorer.json")
    sizes_mb = {p.name: round(p.stat().st_size / 1e6, 2) for p in FINAL_DIR.glob("*.joblib")}
    prod_test = comparison.set_index("model").loc[production]
    metrics = {
        "dataset": "custom_india_v1_1", "experiment": EXP, "split_sizes": sp.sizes(),
        "production_model": production, "candidates_val_pr_auc": val_rank,
        "decision_threshold": thr,
        "test": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                 for k, v in prod_test.items() if k.startswith("test_")},
        "risk_scorer": {"weights": scorer.weights, "cutoffs": scorer.cutoffs, "flag_rates": scorer.flag_rates,
                        **scorer.fit_info},
        "risk_score_test": risk_scores,
        "phase5_reference_xgboost_default_test_pr_auc": float(
            pd.read_csv(rep / "ml_experiment_results.csv").query("version=='v1_1' and model=='xgboost' and exp=='C'").test_pr_auc.iloc[0]),
        "artifact_sizes_mb": sizes_mb,
    }
    (FINAL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    reloaded = joblib.load(FINAL_DIR / "production_model.joblib").predict_proba(Xte)[:, 1]
    sc2 = RS.RiskScorer.load(FINAL_DIR / "risk_scorer.json")
    r2 = sc2.score(reloaded, a_all[sp.test], g_all[sp.test], pep=pep, structuring=struct)
    reload_check = {"max_abs_diff_proba": float(np.abs(reloaded - p_test[production]).max()),
                    "max_abs_diff_risk": float(np.abs(r2["score"] - risk["score"]).max()),
                    "test_pr_auc_reloaded": float(scores(yte, reloaded)["pr_auc"])}
    reload_check["pass"] = reload_check["max_abs_diff_proba"] < 1e-9 and reload_check["max_abs_diff_risk"] < 1e-9

    summary = {
        **metrics,
        "tuning": tuning_sum.round(4).to_dict(orient="records"),
        "stack_meta_weights": E.meta_weights(stack_raw),
        "stack_fit_seconds": fit_seconds,
        "calibration_val": {k: {"brier": round(v["brier"], 4), "ece": round(v["ece"], 4)} for k, v in calib.items()},
        "comparison": comparison.round(4).to_dict(orient="records"),
        "risk_tiers_test": tiers.round(4).to_dict(orient="records"),
        "risk_overrides_test": {"structuring": int(risk["override_structuring"].sum()), "pep_floor": int(risk["override_pep"].sum())},
        "explanations": {"parity_native_vs_shap": contrib_parity, "expected_value_log_odds": ev,
                         "top_shap": shap_imp.head(10).round(4).to_dict(orient="records"),
                         "top_permutation": perm.head(10).round(4).to_dict(orient="records"), "examples": reasons},
        "reload_check": reload_check,
        "runtime_seconds": None,
    }
    F.render_all(tuning_sum=tuning_sum, tuning_res=tuning_res, lc=lc, calib=calib, comparison=comparison,
                 p_test=p_test, y_test=yte, risk=risk, scorer=scorer, tiers=tiers, weight_grid=weight_grid,
                 shap=(sv, Xt_te, names), perm=perm, contrib=contrib_te, examples=examples, test_frame=te,
                 production=production, summary=summary)
    if not reload_check["pass"]:
        raise AssertionError(f"artifact reload does not reproduce test metrics: {reload_check}")
    summary["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    (rep / "final_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary

