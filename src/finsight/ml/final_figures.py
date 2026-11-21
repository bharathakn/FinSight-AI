"""Phase 7 figures (reports/figures/final_*.png).

Colours: base models keep their Phase 5 slots (DT, NB, KNN, RF, XGB, MLP → slots 1–6); the ensembles and
calibrated variants are drawn in ink (neutral, never a categorical slot). Risk labels keep LABEL_COLORS;
tiers use one sequential blue ramp (ordered magnitude, not identity).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve

from finsight import viz
from finsight.ml.supervised_figures import MODEL_COLORS
from finsight.scoring.risk_scorer import TIERS

LABELS = {"decision_tree": "Decision Tree", "knn": "KNN", "random_forest": "Random Forest", "xgboost": "XGBoost",
          "mlp": "MLP", "stack_uncalibrated": "Stack (RF+XGB+MLP → LR)", "xgboost_calibrated": "XGBoost + isotonic",
          "stack_calibrated": "Stack + isotonic"}
TIER_COLORS = {"LOW": viz.BLUE_RAMP[0], "MEDIUM": viz.BLUE_RAMP[2], "HIGH": viz.BLUE_RAMP[4], "CRITICAL": viz.BLUE_RAMP[6]}


def _color(k):
    return MODEL_COLORS.get(k, viz.INK)


def fig_tuning(ts: pd.DataFrame):
    d = ts.sort_values("tuned_cv_pr_auc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    y = np.arange(len(d))
    for i, r in d.iterrows():
        c = _color(r.model)
        ax.plot([r.default_cv_pr_auc, r.tuned_cv_pr_auc], [i, i], color=viz.BASELINE, linewidth=2, zorder=1)
        ax.scatter([r.default_cv_pr_auc], [i], s=70, facecolor=viz.SURFACE, edgecolor=c, linewidth=2, zorder=3)
        ax.scatter([r.tuned_cv_pr_auc], [i], s=80, color=c, edgecolor=viz.SURFACE, linewidth=1.5, zorder=4)
        sig = "p < 0.01" if r.paired_t_pvalue < 0.01 else f"p = {r.paired_t_pvalue:.2f}"
        ax.text(max(r.default_cv_pr_auc, r.tuned_cv_pr_auc) + 0.004, i,
                f"{r.default_cv_pr_auc:.3f} → {r.tuned_cv_pr_auc:.3f}  ({r.paired_mean_diff:+.3f}, {sig}, {r.configs_tried} configs)",
                va="center", fontsize=8.5, color=viz.INK_2)
    ax.set_yticks(y, [LABELS[m] for m in d.model])
    ax.set_xlim(d[["default_cv_pr_auc", "tuned_cv_pr_auc"]].min().min() - 0.01, 1.0)
    ax.set_xlabel("PR-AUC, 5-fold stratified CV on the training split (same folds for default and tuned)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Hyperparameter tuning: every model improves (4 of 5 significantly); the decision tree gains most", pad=16)
    viz.subtitle(ax, "hollow = Phase 5 default settings · filled = tuned (Grid search for DT/KNN, randomized search for RF/XGB/MLP)")
    fig.tight_layout()
    return fig


def fig_bias_variance(tuning_res: pd.DataFrame, lc: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    dt = tuning_res[tuning_res.model == "decision_tree"].copy()
    dt["depth"] = dt["max_depth"].fillna(99).astype(float)          # unlimited depth sorts last
    best = dt.loc[dt.groupby("depth")["mean_test_score"].idxmax()].sort_values("depth")
    x = np.arange(len(best))
    ax = axes[0]
    ax.plot(x, best["mean_train_score"], marker="o", color=viz.SERIES[0], label="training folds")
    ax.plot(x, best["mean_test_score"], marker="o", color=viz.SERIES[1], label="held-out folds")
    ax.set_xticks(x, [("none" if v >= 99 else str(int(v))) for v in best["depth"]])
    ax.set_xlabel("max_depth (best other settings per depth)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Decision tree: depth controls bias vs variance", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    for ax, mname in zip(axes[1:], ("decision_tree", "xgboost")):
        d = lc[lc.model == mname]
        for col, c, lab in (("train", viz.SERIES[0], "training"), ("cv", viz.SERIES[1], "cross-validation")):
            ax.plot(d.n_train, d[f"{col}_mean"], marker="o", color=c, label=lab)
            ax.fill_between(d.n_train, d[f"{col}_mean"] - d[f"{col}_std"], d[f"{col}_mean"] + d[f"{col}_std"],
                            color=c, alpha=0.15, linewidth=0)
        gap = d.train_mean.iloc[-1] - d.cv_mean.iloc[-1]
        ax.set_title(f"Learning curve, tuned {LABELS[mname]} (final gap {gap:.3f})", fontsize=10)
        ax.set_xlabel("training rows")
        ax.legend(fontsize=8, loc="lower right")
    lo = min(lc[["train_mean", "cv_mean"]].min().min(), best[["mean_train_score", "mean_test_score"]].min().min())
    for ax in axes:
        ax.set_ylim(max(0, lo - 0.05), 1.01)
    fig.suptitle("Bias–variance: deeper trees overfit; more data still helps both models (CV curves still rising)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def fig_calibration(calib: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    pairs = [("stack_uncalibrated", "stack_calibrated", "Stacking ensemble"), ("xgboost", "xgboost_calibrated", "XGBoost (tuned)")]
    for ax, (raw, cal, title) in zip(axes, pairs):
        ax.plot([0, 1], [0, 1], color=viz.MUTED, linewidth=1, label="perfect calibration")
        for k, c, style in ((raw, viz.SERIES[0], "--"), (cal, viz.SERIES[1], "-")):
            cv = calib[k]["curve"]
            ax.plot(cv.mean_predicted, cv.fraction_positive, marker="o", markersize=6, color=c, linestyle=style,
                    label=f"{'isotonic-calibrated' if 'calibrated' in k and 'un' not in k else 'raw'}: "
                          f"Brier {calib[k]['brier']:.4f}, ECE {calib[k]['ece']:.3f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("mean predicted P(fraud) in bin")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("observed fraud rate in bin")
    fig.suptitle("Reliability on the validation split (10 equal-count bins)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def fig_test_comparison(comp: pd.DataFrame, production: str, phase5_xgb: float):
    d = comp.sort_values("test_pr_auc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(d))
    ax.barh(y, d.test_pr_auc, color=[_color(k) for k in d.model], height=0.6)
    for i, r in d.iterrows():
        txt = f"{r.test_pr_auc:.3f}   F1 {r.test_f1:.3f}   (val {r.val_pr_auc:.3f})"
        ax.text(r.test_pr_auc + 0.003, i, txt + ("   ← production" if r.model == production else ""),
                va="center", fontsize=8.5, color=viz.INK, fontweight="bold" if r.model == production else "normal")
    ax.axvline(phase5_xgb, color=viz.MUTED, linewidth=1, linestyle="--")
    ax.text(phase5_xgb, -0.75, f" Phase 5 XGBoost (default): {phase5_xgb:.3f}", fontsize=8, color=viz.INK_2, va="center")
    ax.set_yticks(y, [LABELS[k] for k in d.model])
    ax.set_xlim(max(0, d.test_pr_auc.min() - 0.08), 1.04)
    ax.set_xlabel("test PR-AUC (scored once; the production model was chosen on validation)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Final models on the held-out test set (v1.1, Exp C)")
    fig.tight_layout()
    return fig


def fig_risk(risk: dict, scorer, tiers: pd.DataFrame, test_frame: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    lab = test_frame["risk_label"].astype(str).to_numpy()
    grid = np.linspace(0, 100, 401)
    for l in viz.LABEL_ORDER:
        sc = np.sort(risk["score"][lab == l])
        a1.plot(grid, 1 - np.searchsorted(sc, grid, side="left") / len(sc), color=viz.LABEL_COLORS[l], label=l.title())
    for t in ("MEDIUM", "HIGH", "CRITICAL"):
        c = scorer.cutoffs[t]
        a1.axvline(c, color=viz.MUTED, linewidth=1)
        a1.text(c + 0.8, 1.03, f"{t} ≥ {c:.0f}", fontsize=8, color=viz.INK_2, va="bottom")
    a1.set_ylim(0, 1.1)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_xlabel("risk score 0–100 (after overrides)")
    a1.set_ylabel("share of the label scoring ≥ x")
    a1.legend(fontsize=8, loc="center right")
    w = scorer.weights
    a1.set_title(f"Share of each label above a score · weights: model {w['model']:.2f}, anomaly {w['anomaly']:.2f}, "
                 f"graph {w['graph']:.2f}", fontsize=10, pad=18)
    t = tiers.set_index("risk_label").reindex(viz.LABEL_ORDER)
    y = np.arange(len(t))
    left = np.zeros(len(t))
    for tier in TIERS:
        v = t[f"{tier}_share"].to_numpy()
        a2.barh(y, v, left=left, color=TIER_COLORS[tier], edgecolor=viz.SURFACE, linewidth=2, height=0.6, label=tier)
        for i, (lv, vv) in enumerate(zip(left, v)):
            if vv >= 0.06:
                a2.text(lv + vv / 2, i, f"{vv:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if tier in ("HIGH", "CRITICAL") else viz.INK)
        left += v
    a2.set_yticks(y, [f"{l.title()} (n={int(t.loc[l, [f'{x}_n' for x in TIERS]].sum())})" for l in viz.LABEL_ORDER])
    a2.invert_yaxis()
    a2.set_xlim(0, 1)
    a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.grid(axis="y", visible=False)
    a2.set_title("Tier assignment on the test set", fontsize=10)
    a2.legend(loc="lower center", ncols=4, fontsize=8, bbox_to_anchor=(0.5, -0.3))
    fig.suptitle("Risk score: cut-offs from validation (MEDIUM = 98% recall review band, HIGH = max-F1 alert, CRITICAL = 95% precision)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


NICE = {"log_amount": "log amount", "amount_ratio_vs_30d_avg": "amount vs 30-day avg",
        "time_since_beneficiary_added_min": "minutes since payee added", "distance_from_home_km": "distance from home",
        "sec_since_sender_last_inflow": "seconds since last inflow", "receiver_distinct_senders_24h": "receiver senders 24 h",
        "receiver_distinct_senders_7d": "receiver senders 7 d", "sender_tx_24h": "sender tx 24 h",
        "is_new_device": "new device", "device_is_registered": "registered device", "forward_ratio": "forward ratio",
        "is_pass_through": "pass-through", "sender_bene_min_prior_tx": "payee prior tx (min)"}


def nice(f: str) -> str:
    return NICE.get(f, f.replace("_", " "))


def fig_importance(shap_imp: pd.DataFrame, perm: pd.DataFrame, production: str, k: int = 15):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4))
    s = shap_imp.head(k)[::-1]
    a1.barh(np.arange(len(s)), s.mean_abs_shap, color=viz.SERIES[4], height=0.6)
    a1.set_yticks(np.arange(len(s)), [nice(f) for f in s.feature])
    a1.set_xlabel("mean |SHAP| (log-odds)")
    a1.set_title("SHAP global importance (tuned XGBoost, test set)", fontsize=10)
    p = perm.head(k)[::-1]
    a2.barh(np.arange(len(p)), p.pr_auc_drop_mean, xerr=p.pr_auc_drop_std, color=viz.INK, height=0.6,
            error_kw={"ecolor": viz.MUTED, "linewidth": 1})
    a2.set_yticks(np.arange(len(p)), [nice(f) for f in p.feature])
    a2.set_xlabel("drop in validation PR-AUC when the column is shuffled")
    a2.set_title(f"Permutation importance ({LABELS[production]})", fontsize=10)
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    fig.suptitle("What drives the predictions: amount context, payee recency and network flow features",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def fig_beeswarm(sv: np.ndarray, Xt: np.ndarray, names: list[str], k: int = 12):
    order = np.argsort(-np.abs(sv).mean(0))[:k]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for row, j in enumerate(order[::-1]):
        v = Xt[:, j]
        lo, hi = np.nanpercentile(v, [5, 95])
        col = np.clip((v - lo) / max(hi - lo, 1e-12), 0, 1)
        ax.scatter(sv[:, j], row + rng.uniform(-0.32, 0.32, len(v)), c=col, cmap=viz.DIV, s=6, linewidths=0,
                   alpha=0.8, rasterized=True)
    ax.axvline(0, color=viz.BASELINE, linewidth=1)
    ax.set_yticks(range(len(order)), [nice(names[j]) for j in order[::-1]])
    ax.set_xlabel("SHAP value (log-odds; > 0 pushes towards fraud)")
    ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=viz.DIV)
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"]); cb.set_label("feature value"); cb.outline.set_visible(False)
    ax.set_title("SHAP summary: direction of each feature's effect (tuned XGBoost, 2,000 test rows)")
    fig.tight_layout()
    return fig


def fig_local(contrib: pd.DataFrame, examples: dict, summary: dict, k: int = 8):
    ex = {k_: v for k_, v in examples.items() if v is not None}
    fig, axes = plt.subplots(1, len(ex), figsize=(7 * len(ex), 4.6), squeeze=False)
    info = summary["explanations"]["examples"]
    for ax, (kind, i) in zip(axes[0], ex.items()):
        c = contrib.iloc[i]
        bias = c["bias"]
        top = c.drop("bias").reindex(c.drop("bias").abs().sort_values(ascending=False).index[:k])
        rest = c.drop("bias").drop(top.index).sum()
        vals = list(top.values) + [rest]
        labs = [nice(f) for f in top.index] + ["all other features"]
        y = np.arange(len(vals))[::-1]
        ax.barh(y, vals, color=[viz.SERIES[1] if v > 0 else viz.SERIES[0] for v in vals], height=0.6)
        for yy, v in zip(y, vals):
            ax.text(v + (0.05 if v >= 0 else -0.05), yy, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color=viz.BASELINE, linewidth=1)
        ax.set_yticks(y, labs)
        ax.grid(axis="y", visible=False)
        lo, hi = min(vals + [0]), max(vals + [0])
        ax.set_xlim(lo - 0.25 * (hi - lo) - 0.2, hi + 0.25 * (hi - lo) + 0.2)
        d = info[kind]
        ax.set_title(f"{'Caught fraud' if kind == 'fraud' else 'BORDERLINE false alarm'}: {d['scenario']}", fontsize=10, pad=16)
        viz.subtitle(ax, f"P(fraud) {d['p_fraud']:.2f} · risk {d['risk_score']:.0f} ({d['tier']}) · base log-odds {bias:+.2f}")
        ax.set_xlabel("contribution to log-odds (XGBoost pred_contribs = TreeSHAP)")
    handles = [Line2D([], [], color=viz.SERIES[1], linewidth=8, label="pushes towards fraud"),
               Line2D([], [], color=viz.SERIES[0], linewidth=8, label="pushes towards legitimate")]
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncols=2, fontsize=9)
    return fig


def fig_pr_test(p_test: dict, y, production: str):
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for k in ("random_forest", "xgboost", "mlp", production):
        p, r, _ = precision_recall_curve(y, p_test[k])
        ax.plot(r, p, color=_color(k), linewidth=2.5 if k == production else 1.8,
                linestyle="-" if k == production else "-", label=LABELS[k] + (" (production)" if k == production else ""))
    ax.axhline(y.mean(), color=viz.MUTED, linewidth=1)
    ax.text(0.01, y.mean() + 0.015, f"no-skill = {y.mean():.1%}", fontsize=8, color=viz.MUTED)
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02)
    ax.set_title("Test precision–recall: the stack's base learners vs the production model")
    ax.legend(fontsize=8.5, loc="lower left", bbox_to_anchor=(0.0, 0.2))
    fig.tight_layout()
    return fig


def render_all(tuning_sum, tuning_res, lc, calib, comparison, p_test, y_test, risk, scorer, tiers, weight_grid,
               shap, perm, contrib, examples, test_frame, production, summary) -> list:
    viz.apply_style()
    sv, Xt, names = shap
    from finsight.ml.explain import global_importance

    figs = {
        "final_01_tuning": fig_tuning(tuning_sum),
        "final_02_bias_variance": fig_bias_variance(tuning_res, lc),
        "final_03_calibration": fig_calibration(calib),
        "final_04_test_comparison": fig_test_comparison(comparison, production,
                                                        summary["phase5_reference_xgboost_default_test_pr_auc"]),
        "final_05_pr_test": fig_pr_test(p_test, y_test, production),
        "final_06_risk_score": fig_risk(risk, scorer, tiers, test_frame),
        "final_07_importance": fig_importance(global_importance(sv, names), perm, production),
        "final_08_shap_beeswarm": fig_beeswarm(sv, Xt, names),
        "final_09_local_explanations": fig_local(contrib, examples, summary),
    }
    out = []
    for name, fig in figs.items():
        out.append(viz.save(fig, name))
        plt.close(fig)
    summary["figures"] = [p.name for p in out]
    return out
