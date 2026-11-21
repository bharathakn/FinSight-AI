"""Phase 5 experiment runner — supervised classification (Plan §7.1–7.3, RQ3, RQ4, RQ7).

Experiments
  1. main grid      6 models × Exp A/B/C × {v1.0, v1.1}; SMOTE-in-pipeline
                    - 5×2 repeated stratified CV on dev (80%) → PR-AUC per fold (paired across configs)
                    - fit on train (60%), threshold = max-F1 on val (20%), score once on test (20%)
  2. imbalance      none vs class_weight vs SMOTE (v1.1, Exp C, same CV folds)
  3. temporal       RF + XGB × A/B/C, train Aug 1–19 / val 20–23 / test 24–30 (v1.1)
  4. rule baseline  Phase 4 rules re-mined on TRAIN only, k chosen on val, scored on test
  5. extension      3-class risk_label with XGBoost (v1.1, Exp C)
Outputs → reports/ml_*.csv|json, reports/figures/ml_*.png, artifacts/models/phase5/*.joblib
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from finsight import config
from finsight.eda import load_master
from finsight.ml import classifiers as C
from finsight.ml.evaluator import best_f1_threshold, paired_test, per_scenario, scores
from finsight.ml.feature_sets import feature_columns, get_xy
from finsight.ml.splits import cv_folds, stratified_split, temporal_split

EXPS = ["A", "B", "C"]
VERSIONS = ["v1_0", "v1_1"]
MAIN_IMBALANCE = "smote"


def _pos_weight(y) -> float:
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def _fit_predict(name, exp, imbalance, X, y, tr, te):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = C.build(name, exp, imbalance, pos_weight=_pos_weight(y.iloc[tr]))
        model.fit(X.iloc[tr], y.iloc[tr])
        return model, model.predict_proba(X.iloc[te])[:, 1]


def _cv_task(version, name, exp, imbalance, fold, X, y, tr, te):
    t0 = time.perf_counter()
    _, p = _fit_predict(name, exp, imbalance, X, y, tr, te)
    s = scores(y.iloc[te], p, 0.5)
    return {"version": version, "model": name, "exp": exp, "imbalance": imbalance, "fold": fold,
            "pr_auc": s["pr_auc"], "roc_auc": s["roc_auc"], "f1_at_0.5": s["f1"], "fit_seconds": time.perf_counter() - t0}


def run_cv(masters: dict, configs: list[tuple], n_jobs: int = -1) -> pd.DataFrame:
    """configs: (version, model, exp, imbalance). Same folds for every config of a version ⇒ paired."""
    tasks = []
    for version, m in masters.items():
        folds = cv_folds(m, stratified_split(m).dev)
        Xs = {e: get_xy(m, e) for e in EXPS}
        for v, name, exp, imb in configs:
            if v != version:
                continue
            X, y = Xs[exp]
            tasks += [delayed(_cv_task)(v, name, exp, imb, k, X, y, tr, te) for k, (tr, te) in enumerate(folds)]
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(tasks))


def holdout(m: pd.DataFrame, name: str, exp: str, split, imbalance: str = MAIN_IMBALANCE) -> dict:
    """Fit on train, choose the max-F1 threshold on val, score on test (once)."""
    X, y = get_xy(m, exp)
    model, p_val = _fit_predict(name, exp, imbalance, X, y, split.train, split.val)
    thr = best_f1_threshold(y.iloc[split.val], p_val)
    p_test = model.predict_proba(X.iloc[split.test])[:, 1]
    s = scores(y.iloc[split.test], p_test, thr)
    s["f1_at_0.5"] = scores(y.iloc[split.test], p_test, 0.5)["f1"]
    test_frame = m.iloc[split.test]
    lab = test_frame["risk_label"].astype(str).to_numpy()
    s["borderline_flag_rate"] = float((p_test[lab == "BORDERLINE"] >= thr).mean())
    s["legit_flag_rate"] = float((p_test[lab == "LEGITIMATE"] >= thr).mean())
    return {"model": model, "proba": p_test, "threshold": thr, "scores": s,
            "per_scenario": per_scenario(test_frame, p_test, thr)}


def rule_baseline(m: pd.DataFrame, split) -> dict:
    """Phase 4 carry-over: mine rules on TRAIN rows only; pick top-k by val F1; score on test."""
    from finsight.mining import association as A
    from finsight.mining.discretiser import india_items

    items = india_items(m)
    tr_items = items.iloc[split.train]
    fs = A.mine(tr_items, "fpgrowth", 0.01, track_memory=False).itemsets
    rules = A.productive_rules(A.filter_rules(A.target_rules(fs, len(tr_items))))
    ranked = A.top_rules(rules, "FRAUDULENT", k=len(rules))
    feat = items.drop(columns=["FRAUDULENT", "BORDERLINE"])
    fired_val = A.rule_matches(feat.iloc[split.val], ranked)
    fired_test = A.rule_matches(feat.iloc[split.test], ranked)
    y = m["is_fraud"].to_numpy()
    yv, yt = y[split.val], y[split.test]
    best_k, best_f1 = 1, -1.0
    for k in range(1, len(ranked) + 1):
        f = f1_score(yv, fired_val[:, :k].any(axis=1))
        if f > best_f1:
            best_k, best_f1 = k, f
    # "score" = max lift of fired rules → gives a ranking for PR-AUC
    lift = ranked["lift"].to_numpy()
    score_test = np.where(fired_test, lift, 0).max(axis=1)
    s = scores(yt, score_test, threshold=np.sort(lift)[::-1][best_k - 1])
    return {"n_rules_train": len(ranked), "k_selected_on_val": best_k, "val_f1": round(best_f1, 4),
            "test": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()}}


def three_class(m: pd.DataFrame, split) -> dict:
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline

    X, _ = get_xy(m, "C")
    labels = ["LEGITIMATE", "FRAUDULENT", "BORDERLINE"]
    y = m["risk_label"].astype(str).map({l: i for i, l in enumerate(labels)}).to_numpy()
    w = pd.Series(y).map(len(y) / (3 * pd.Series(y).value_counts())).to_numpy()
    pipe = Pipeline([("prep", C.preprocessor("C", scaled=False)),
                     ("model", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
                                             colsample_bytree=0.8, tree_method="hist", n_jobs=4,
                                             random_state=config.SEED, verbosity=0))])
    pipe.fit(X.iloc[split.train], y[split.train], model__sample_weight=w[split.train])
    pred = pipe.predict(X.iloc[split.test])
    rep = classification_report(y[split.test], pred, target_names=labels, output_dict=True, zero_division=0)
    return {"labels": labels, "macro_f1": round(rep["macro avg"]["f1-score"], 4),
            "per_class_f1": {l: round(rep[l]["f1-score"], 4) for l in labels},
            "confusion": confusion_matrix(y[split.test], pred).tolist()}


def decision_tree_for_plot(m: pd.DataFrame, split, depth: int = 4):
    from sklearn.tree import DecisionTreeClassifier

    X, y = get_xy(m, "C")
    prep = C.preprocessor("C", scaled=False)
    Xt = prep.fit_transform(X.iloc[split.train])
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, class_weight="balanced",
                                  random_state=config.SEED).fit(Xt, y.iloc[split.train])
    return tree, list(prep.get_feature_names_out())


# ---------------------------------------------------------------- orchestration

def run_all(n_jobs: int = -1) -> dict:
    from finsight.ml import supervised_figures as F

    t_start = time.perf_counter()
    rep = config.REPORTS_DIR
    masters = {v: load_master(v) for v in VERSIONS}
    m11 = masters["v1_1"]

    # 1. main CV grid
    main_cfg = [(v, n, e, MAIN_IMBALANCE) for v in VERSIONS for n in C.MODELS for e in EXPS]
    # 2. imbalance strategies (v1.1, Exp C); smote runs are shared with the main grid
    imb_cfg = [("v1_1", n, "C", imb) for n in C.MODELS for imb in ("none", "class_weight")
               if C.effective_imbalance(n, imb) == imb]
    cv = run_cv(masters, main_cfg + imb_cfg, n_jobs=n_jobs)
    cv.to_csv(rep / "ml_cv_folds.csv", index=False)
    main = cv[cv["imbalance"] == MAIN_IMBALANCE]

    # paired significance: B vs A, C vs B per model/version
    sig = []
    for (v, n), g in main.groupby(["version", "model"]):
        piv = g.pivot(index="fold", columns="exp", values="pr_auc")
        for a, b in (("A", "B"), ("B", "C")):
            sig.append({"version": v, "model": n, "comparison": f"{b} vs {a}", **paired_test(piv[a], piv[b])})
    sig = pd.DataFrame(sig)
    sig.to_csv(rep / "ml_paired_tests.csv", index=False)

    # hold-out test evaluation (main grid)
    rows, curves, scen = [], {}, []
    model_dir = config.ARTIFACTS_DIR / "models" / "phase5"
    model_dir.mkdir(parents=True, exist_ok=True)
    for v, m in masters.items():
        sp = stratified_split(m)
        for n in C.MODELS:
            for e in EXPS:
                h = holdout(m, n, e, sp)
                cvs = main[(main.version == v) & (main.model == n) & (main.exp == e)]["pr_auc"]
                rows.append({"version": v, "model": n, "exp": e, "cv_pr_auc_mean": cvs.mean(), "cv_pr_auc_std": cvs.std(),
                             **{f"test_{k}": val for k, val in h["scores"].items()}})
                if v == "v1_1":
                    curves[(n, e)] = h["proba"]
                    scen.append(h["per_scenario"].assign(model=n, exp=e))
                    if e == "C":
                        joblib.dump(h["model"], model_dir / f"{n}_expC_v1_1.joblib")
    results = pd.DataFrame(rows)
    results.to_csv(rep / "ml_experiment_results.csv", index=False)
    scen = pd.concat(scen, ignore_index=True)
    scen.to_csv(rep / "ml_per_scenario.csv", index=False)
    sp11 = stratified_split(m11)
    y_test = m11["is_fraud"].to_numpy()[sp11.test]

    # imbalance summary (CV)
    imb = cv[(cv.version == "v1_1") & (cv.exp == "C")].groupby(["model", "imbalance"])[["pr_auc", "f1_at_0.5"]] \
        .agg(["mean", "std"]).reset_index()
    imb.columns = ["_".join(c).strip("_") for c in imb.columns]
    imb.to_csv(rep / "ml_imbalance.csv", index=False)

    # 3. temporal robustness
    tsp = temporal_split(m11)
    temporal = []
    for n in ("random_forest", "xgboost"):
        for e in EXPS:
            h = holdout(m11, n, e, tsp)
            temporal.append({"model": n, "exp": e, "split": "temporal", **{k: h["scores"][k] for k in ("pr_auc", "roc_auc", "f1")}})
            s = results[(results.version == "v1_1") & (results.model == n) & (results.exp == e)].iloc[0]
            temporal.append({"model": n, "exp": e, "split": "stratified", "pr_auc": s.test_pr_auc,
                             "roc_auc": s.test_roc_auc, "f1": s.test_f1})
    temporal = pd.DataFrame(temporal)
    temporal.to_csv(rep / "ml_temporal.csv", index=False)

    # 4. rule baseline, 5. three-class, DT plot
    rules = rule_baseline(m11, sp11)
    tri = three_class(m11, sp11)
    tree, tree_feats = decision_tree_for_plot(m11, sp11)

    best = results[results.version == "v1_1"].sort_values("test_pr_auc", ascending=False).iloc[0]
    summary = {
        "split_sizes": sp11.sizes(), "temporal_split_sizes": tsp.sizes(),
        "cv": "RepeatedStratifiedKFold(5×2) on dev (80%), stratified on risk_label; SMOTE inside pipeline",
        "n_features": {e: len(feature_columns(e)) for e in EXPS},
        "best_v1_1": {"model": best.model, "exp": best.exp, **{k: round(float(best[k]), 4) for k in
                      ("cv_pr_auc_mean", "test_pr_auc", "test_roc_auc", "test_f1", "test_precision", "test_recall",
                       "test_borderline_flag_rate")}},
        "acceptance_C_gt_B_v1_1": {n: bool(results.query("version=='v1_1' and model==@n and exp=='C'").cv_pr_auc_mean.iloc[0]
                                         > results.query("version=='v1_1' and model==@n and exp=='B'").cv_pr_auc_mean.iloc[0])
                                   for n in ("random_forest", "xgboost")},
        "paired_tests": sig.round(5).to_dict(orient="records"),
        "rule_baseline_v1_1": rules,
        "three_class_xgb_expC": tri,
        "temporal": temporal.round(4).to_dict(orient="records"),
        "runtime_seconds": None,
    }

    F.render_all(results=results, cv=main, sig=sig, curves=curves, y_test=y_test, scen=scen, imb=cv,
                 temporal=temporal, rules=rules, tree=tree, tree_feats=tree_feats, tri=tri,
                 test_frame=m11.iloc[sp11.test], summary=summary)
    summary["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    (rep / "ml_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
