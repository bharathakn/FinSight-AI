"""Phase 5 figures (reports/figures/ml_*.png). Model colours follow the fixed palette order
(DT, NB, KNN, RF, XGB, MLP → slots 1–6) in every figure; experiments A/B/C → slots 1–3."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.tree import plot_tree

from finsight import viz
from finsight.ml.classifiers import MODEL_LABELS, MODELS

MODEL_COLORS = {m: viz.SERIES[i] for i, m in enumerate(MODELS)}
EXP_COLORS = {"A": viz.SERIES[0], "B": viz.SERIES[1], "C": viz.SERIES[2]}
EXP_NAMES = {"A": "A · transaction", "B": "B · + behavioural", "C": "C · + network"}
VERSION_NAMES = {"v1_0": "v1.0 (control: no graph structure)", "v1_1": "v1.1 (graph structure injected)"}


def fig_abc_lines(results: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    x = np.arange(3)
    for ax, v in zip(axes, ("v1_0", "v1_1")):
        r = results[results.version == v]
        for n in MODELS:
            d = r[r.model == n].set_index("exp").loc[["A", "B", "C"]]
            ax.errorbar(x, d["cv_pr_auc_mean"], yerr=d["cv_pr_auc_std"], color=MODEL_COLORS[n], marker="o",
                        markersize=6, capsize=3, linewidth=2, label=MODEL_LABELS[n])
            if v == "v1_1":                       # v1.0 endpoints overlap; the legend carries identity there
                ax.annotate(MODEL_LABELS[n], (2, d["cv_pr_auc_mean"].iloc[-1]), xytext=(8, 0),
                            textcoords="offset points", fontsize=8, color=viz.INK_2, va="center")
        ax.set_xticks(x, [EXP_NAMES[e] for e in "ABC"])
        ax.set_xlim(-0.2, 2.7)
        ax.set_title(VERSION_NAMES[v], fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("PR-AUC (5×2 CV mean ± sd)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("RQ3: behavioural features lift every model; network features add a further lift only where structure exists (v1.1)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def fig_test_heatmap(results: pd.DataFrame):
    piv = results.pivot_table(index="model", columns=["version", "exp"], values="test_pr_auc").loc[MODELS]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    im = ax.imshow(piv.to_numpy(), cmap=viz.SEQ, vmin=0.3, vmax=1, aspect="auto")
    ax.set_yticks(range(len(piv)), [MODEL_LABELS[m] for m in piv.index])
    ax.set_xticks(range(piv.shape[1]), [f"{'v1.0' if v == 'v1_0' else 'v1.1'} · {e}" for v, e in piv.columns])
    ax.axvline(2.5, color=viz.SURFACE, linewidth=4)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9, color="white" if val > 0.75 else viz.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.outline.set_visible(False)
    ax.set_title("Held-out test PR-AUC (fit on 60% train, scored once on 20% test)")
    fig.tight_layout()
    return fig


def fig_pr_roc(curves: dict, y_test, exp: str = "C", rule_point: dict | None = None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    base = y_test.mean()
    for n in MODELS:
        p = curves[(n, exp)]
        pr, rc, _ = precision_recall_curve(y_test, p)
        a1.plot(rc, pr, color=MODEL_COLORS[n], label=MODEL_LABELS[n])
        fpr, tpr, _ = roc_curve(y_test, p)
        a2.plot(fpr, tpr, color=MODEL_COLORS[n], label=MODEL_LABELS[n])
    a1.axhline(base, color=viz.MUTED, linewidth=1)
    a1.text(0.99, base + 0.015, f"no-skill = fraud rate {base:.1%}", fontsize=8, color=viz.MUTED, ha="right")
    if rule_point:
        a1.scatter([rule_point["recall"]], [rule_point["precision"]], s=90, marker="D", color=viz.INK,
                   edgecolor=viz.SURFACE, linewidth=2, zorder=5)
        a1.annotate("association rules\n(mined on train)", (rule_point["recall"], rule_point["precision"]),
                    xytext=(-95, -40), textcoords="offset points", fontsize=8, color=viz.INK,
                    arrowprops={"arrowstyle": "-", "color": viz.MUTED, "linewidth": 1})
    a1.set_xlabel("recall"); a1.set_ylabel("precision"); a1.set_ylim(0, 1.02)
    a1.set_title(f"Precision–recall, test set, Exp {exp} (v1.1)", fontsize=10)
    a2.plot([0, 1], [0, 1], color=viz.MUTED, linewidth=1)
    a2.set_xlabel("false positive rate"); a2.set_ylabel("true positive rate")
    a2.set_title(f"ROC, test set, Exp {exp} (v1.1)", fontsize=10)
    a2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_per_scenario(scen: pd.DataFrame, model: str = "xgboost"):
    d = scen[(scen.model == model) & (scen.risk_label == "FRAUDULENT")]
    piv = d.pivot(index="scenario", columns="exp", values="flag_rate")
    piv = piv.loc[sorted(piv.index, key=lambda s: int(s.split()[0][1:]))][::-1]
    n = d.drop_duplicates("scenario").set_index("scenario")["n"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = np.arange(len(piv))
    for k, e in enumerate("ABC"):
        ax.barh(y + (1 - k) * 0.26, piv[e], height=0.24, color=EXP_COLORS[e], label=EXP_NAMES[e])
    for i, s in enumerate(piv.index):
        ax.text(1.01, i, f"n={n[s]}", va="center", fontsize=8, color=viz.MUTED)
    ax.set_yticks(y, piv.index)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("recall on FRAUDULENT test rows (validation-tuned threshold)")
    ax.set_title(f"RQ7: per-scenario recall — {MODEL_LABELS[model]}, v1.1 test set")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.05), fontsize=9)
    return fig


def fig_imbalance(cv: pd.DataFrame):
    d = cv[(cv.version == "v1_1") & (cv.exp == "C")]
    agg = d.groupby(["model", "imbalance"])["pr_auc"].agg(["mean", "std"]).reset_index()
    strategies = ["none", "class_weight", "smote"]
    cols = {"none": viz.SERIES[0], "class_weight": viz.SERIES[1], "smote": viz.SERIES[2]}
    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(MODELS))
    for k, s in enumerate(strategies):
        sub = agg[agg.imbalance == s].set_index("model").reindex(MODELS)
        ax.bar(x + (k - 1) * 0.26, sub["mean"], yerr=sub["std"], width=0.24, color=cols[s], label=s,
               capsize=2, error_kw={"ecolor": viz.MUTED, "linewidth": 1})
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS])
    ax.set_ylim(0.3, 1); ax.set_ylabel("PR-AUC (5×2 CV)")
    ax.set_title("Imbalance handling (v1.1, Exp C): class_weight only exists for DT / RF / XGB; missing bars = not applicable")
    ax.legend(loc="upper left", ncols=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_confusion_by_label(test_frame: pd.DataFrame, proba, thr: float, model: str):
    lab = test_frame["risk_label"].astype(str).to_numpy()
    flag = np.asarray(proba) >= thr
    rows = viz.LABEL_ORDER
    mat = np.array([[((lab == r) & ~flag).sum(), ((lab == r) & flag).sum()] for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    share = mat / mat.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap=viz.SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(3):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}\n({share[i, j]:.0%})", ha="center", va="center", fontsize=10,
                    color="white" if share[i, j] > 0.6 else viz.INK)
    ax.set_xticks([0, 1], ["not flagged", "flagged as fraud"])
    ax.set_yticks(range(3), rows)
    ax.grid(False)
    ax.set_title(f"{MODEL_LABELS[model]} (v1.1, Exp C) on test — errors by true label", fontsize=10)
    fig.tight_layout()
    return fig


def fig_temporal(temporal: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, n in zip(axes, ("random_forest", "xgboost")):
        d = temporal[temporal.model == n]
        x = np.arange(3)
        for k, (sp, col) in enumerate((("stratified", viz.SERIES[0]), ("temporal", viz.SERIES[1]))):
            s = d[d.split == sp].set_index("exp").loc[["A", "B", "C"]]
            ax.bar(x + (k - 0.5) * 0.36, s["pr_auc"], width=0.34, color=col, label=f"{sp} split")
            for i, val in enumerate(s["pr_auc"]):
                ax.text(x[i] + (k - 0.5) * 0.36, val + 0.01, f"{val:.2f}", ha="center", fontsize=8, color=viz.INK_2)
        ax.set_xticks(x, [EXP_NAMES[e] for e in "ABC"])
        ax.set_title(MODEL_LABELS[n], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("test PR-AUC"); axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Temporal robustness (v1.1): train Aug 1–19, validate Aug 20–23, test Aug 24–30",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def fig_borderline(results: pd.DataFrame):
    d = results[(results.version == "v1_1") & (results.exp == "C")].set_index("model").loc[MODELS]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(MODELS))
    series = (("test_recall", "fraud recall", viz.SERIES[1]), ("test_borderline_flag_rate", "BORDERLINE flagged", viz.SERIES[2]),
              ("test_legit_flag_rate", "LEGITIMATE flagged", viz.SERIES[0]))
    for k, (col, name, c) in enumerate(series):
        ax.bar(x + (k - 1) * 0.26, d[col], width=0.24, color=c, label=name)
        for i, val in enumerate(d[col]):
            ax.text(x[i] + (k - 1) * 0.26, val + 0.012, f"{val:.0%}", ha="center", fontsize=7, color=viz.INK_2)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODELS])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.1)
    ax.set_title("Who gets flagged (v1.1, Exp C, test): BORDERLINE is flagged far more than LEGITIMATE")
    ax.legend(loc="upper left", ncols=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_tree(tree, feats):
    fig, ax = plt.subplots(figsize=(34, 10))
    plot_tree(tree, feature_names=feats, class_names=["legit", "fraud"], filled=False, impurity=False,
              proportion=True, rounded=True, fontsize=9, ax=ax)
    ax.set_title("Decision Tree (depth 4, Exp C, v1.1, class-balanced) — readable rules learned from data", fontsize=14)
    return fig


def render_all(results, cv, sig, curves, y_test, scen, imb, temporal, rules, tree, tree_feats, tri, test_frame, summary):
    viz.apply_style()
    best = summary["best_v1_1"]
    figs = {
        "ml_01_abc_cv_pr_auc": lambda: fig_abc_lines(results),
        "ml_02_test_pr_auc_heatmap": lambda: fig_test_heatmap(results),
        "ml_03_pr_roc_expC": lambda: fig_pr_roc(curves, y_test, "C", rules["test"]),
        "ml_04_per_scenario_xgb": lambda: fig_per_scenario(scen, "xgboost"),
        "ml_05_imbalance": lambda: fig_imbalance(imb),
        "ml_06_confusion_by_label": lambda: fig_confusion_by_label(
            test_frame, curves[(best["model"], "C")],
            float(results.query("version=='v1_1' and model==@best['model'] and exp=='C'").test_threshold.iloc[0]),
            best["model"]),
        "ml_07_temporal": lambda: fig_temporal(temporal),
        "ml_08_borderline": lambda: fig_borderline(results),
        "ml_09_decision_tree": lambda: fig_tree(tree, tree_feats),
    }
    paths = {}
    for name, make in figs.items():
        f = make(); paths[name] = str(viz.save(f, name)); plt.close(f)
    summary["figures"] = paths
