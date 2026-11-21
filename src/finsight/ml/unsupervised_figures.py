"""Phase 6 figures (reports/figures/unsup_*.png).

Colour assignment (fixed, never cycled)
  * risk labels: viz.LABEL_COLORS (LEGIT blue, FRAUD orange, BORDERLINE aqua), as in every phase
  * experiments: B = slot 2, C = slot 3 (same as the Phase 5 figures)
  * detectors: IF, OC-SVM ν0.01, OC-SVM ν0.05, AE = slots 4–7; the supervised XGBoost reference is ink, dashed
  * K-Means clusters: slots 4–8 (kept off slots 1–3 so a cluster is never mistaken for a label)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import precision_recall_curve

from finsight import viz
from finsight.ml.anomaly import DETECTOR_LABELS

EXP_COLORS = {"B": viz.SERIES[1], "C": viz.SERIES[2]}
EXP_NAMES = {"B": "Exp B · transaction + behavioural", "C": "Exp C · + network"}
DETECTORS = list(DETECTOR_LABELS)
DET_COLORS = {d: viz.SERIES[3 + i] for i, d in enumerate(DETECTORS)}
CLUSTER_COLORS = viz.SERIES[3:]
PCT = lambda v, _=None: f"{v:.0%}"  # noqa: E731

FEATURE_NICE = {
    "sender_tx_10m": "sender tx in 10 min", "sender_tx_1h": "sender tx in 1 h",
    "time_since_last_tx_sec": "time since last tx", "receiver_tx_in_24h": "receiver tx in 24 h",
    "receiver_distinct_senders_24h": "receiver distinct senders 24 h", "receiver_pagerank": "receiver PageRank",
    "amount_ratio_vs_30d_avg": "amount vs 30-day avg", "log_amount": "log amount",
    "same_community": "same Louvain community", "forward_ratio": "forward ratio",
}


def _nice(f: str) -> str:
    return FEATURE_NICE.get(f, f.replace("_", " "))


def fig_kmeans_selection(sweeps: pd.DataFrame, chosen: dict):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    specs = [("inertia", "inertia (within-cluster SSE)", "elbow: lower is tighter"),
             ("silhouette", "silhouette", "higher is better"),
             ("davies_bouldin", "Davies–Bouldin index", "lower is better")]
    for ax, (col, ylab, note) in zip(axes, specs):
        for exp in ("B", "C"):
            d = sweeps[sweeps.exp == exp]
            ax.plot(d["k"], d[col], marker="o", markersize=6, color=EXP_COLORS[exp], label=EXP_NAMES[exp])
            k = chosen[exp]
            v = d.set_index("k").loc[k, col]
            ax.scatter([k], [v], s=150, facecolor="none", edgecolor=viz.INK, linewidth=1.5, zorder=5)
        ax.set_xlabel("K")
        ax.set_title(ylab, fontsize=10, pad=16)
        viz.subtitle(ax, note)
        ax.set_xticks(range(2, 9))
        ax.grid(axis="x", visible=False)
    axes[1].set_ylim(0, None)
    handles = [Line2D([], [], color=EXP_COLORS[e], marker="o", label=EXP_NAMES[e]) for e in ("B", "C")] + \
              [Line2D([], [], marker="o", markersize=11, markerfacecolor="none", markeredgecolor=viz.INK,
                      linestyle="none", label="chosen K (best Davies–Bouldin on the silhouette plateau)")]
    fig.suptitle("K-Means model selection (v1.1, all 10,000 rows, labels unused): silhouette is low and flat, so cluster structure is weak",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncols=3, fontsize=9, bbox_to_anchor=(0.5, 0))
    return fig


def fig_cluster_composition(clusters: pd.DataFrame, profile: dict, exp: str = "C"):
    d = clusters[(clusters.exp == exp)].sort_values("fraud_rate").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 0.9 * len(d) + 1.6))
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for lab in viz.LABEL_ORDER:
        w = d[f"share_{lab}"].to_numpy()
        ax.barh(y, w, left=left, height=0.6, color=viz.LABEL_COLORS[lab], edgecolor=viz.SURFACE, linewidth=2,
                label=lab.title())
        left += w
    names = []
    for _, r in d.iterrows():
        top = profile[str(int(r.cluster))][:2]
        names.append(f"cluster {int(r.cluster)}  (n={int(r.n):,})\n" + " · ".join(f"{_nice(f)} {v:+.1f}σ" for f, v in top))
    ax.set_yticks(y, names, fontsize=8.5)
    for i, r in d.iterrows():
        ax.text(1.01, i, f"fraud {r.fraud_rate:.0%}", va="center", fontsize=9, color=viz.INK)
    ax.set_xlim(0, 1.13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(PCT))
    ax.set_xlabel("share of the cluster's rows by risk label (labels attached after clustering)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"RQ5: K-Means (K={len(d)}, Exp {exp}) finds one normal cluster and {len(d) - 1} fraud-heavy ones", pad=16)
    viz.subtitle(ax, "each cluster is named by its two most extreme centroid features (standard-deviation units)")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncols=3, fontsize=9)
    return fig


def _scatter(ax, P, groups, order, colors, names, size=4):
    for g in order:
        sel = groups == g
        ax.scatter(P[sel, 0], P[sel, 1], s=size, color=colors[g], alpha=0.55 if g in ("LEGITIMATE", 0) else 0.8,
                   linewidths=0, label=f"{names[g]} ({sel.sum():,})", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)


def fig_embeddings(proj: dict, m: pd.DataFrame, clusters: np.ndarray, cluster_fraud: dict):
    lab = m["risk_label"].astype(str).to_numpy()
    ks = sorted(np.unique(clusters), key=lambda c: cluster_fraud[c])       # lowest-fraud cluster drawn first
    ccol = {c: CLUSTER_COLORS[i] for i, c in enumerate(ks)}
    cname = {c: f"cluster {c} · fraud {cluster_fraud[c]:.0%}" for c in ks}
    lname = {l: l.title() for l in viz.LABEL_ORDER}
    panels = [("pca", "PCA"), ("tsne", "t-SNE")] if "tsne" in proj else [("pca", "PCA")]
    fig, axes = plt.subplots(len(panels), 2, figsize=(12, 5.4 * len(panels)), squeeze=False)
    for r, (key, title) in enumerate(panels):
        P = proj[key]
        sub = f" (PC1 {proj['pca_var'][0]:.0%}, PC2 {proj['pca_var'][1]:.0%} of variance)" if key == "pca" else ""
        _scatter(axes[r, 0], P, lab, viz.LABEL_ORDER, viz.LABEL_COLORS, lname)
        axes[r, 0].set_title(f"{title}{sub}: coloured by true label", fontsize=10)
        _scatter(axes[r, 1], P, clusters, ks, ccol, cname)
        axes[r, 1].set_title(f"{title}: coloured by K-Means cluster", fontsize=10)
        for ax in axes[r]:
            leg = ax.legend(loc="best", fontsize=8, markerscale=3, handletextpad=0.3)
            for h in leg.legend_handles:
                h.set_alpha(1)
    fig.suptitle("2-D projections of the scaled Exp C features (v1.1, 10,000 rows): fraud concentrates in regions, but they overlap with normal traffic",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def fig_dbscan(kdist: np.ndarray, eps: float, dbs: dict):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    a1.plot(np.arange(len(kdist)), kdist, color=viz.SERIES[0])
    a1.axhline(eps, color=viz.MUTED, linewidth=1)
    a1.text(len(kdist) * 0.02, eps, f"ε = {eps:.2f} (knee)", va="bottom", fontsize=9, color=viz.INK_2)
    a1.set_xlabel("points sorted by distance"); a1.set_ylabel(f"distance to {dbs['min_samples']}-th neighbour")
    a1.set_title("k-distance plot used to pick ε", fontsize=10)
    labs = viz.LABEL_ORDER
    vals = [dbs["noise_rate_by_label"][l] for l in labs]
    a2.bar(range(3), vals, color=[viz.LABEL_COLORS[l] for l in labs], width=0.6)
    for i, v in enumerate(vals):
        a2.text(i, v, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    a2.set_xticks(range(3), [l.title() for l in labs])
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_ylabel("share labelled noise by DBSCAN")
    a2.set_title(f"DBSCAN noise: {dbs['noise_n']} points, fraud rate {dbs['fraud_rate_noise']:.0%} vs {dbs['fraud_rate_clustered']:.0%} elsewhere",
                 fontsize=10)
    a2.grid(axis="x", visible=False)
    fig.suptitle("DBSCAN (Exp C): the few points it calls noise are fraud-heavy, but noise catches only a small share of fraud",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def fig_detector_pr(scores: dict, xgb_p: np.ndarray, y: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for d in DETECTORS:
        p, r, _ = precision_recall_curve(y, scores[d])
        ax.plot(r, p, color=DET_COLORS[d], label=DETECTOR_LABELS[d])
    p, r, _ = precision_recall_curve(y, xgb_p)
    ax.plot(r, p, color=viz.INK, linestyle="--", linewidth=1.5, label="XGBoost (supervised, Phase 5 reference)")
    ax.axhline(y.mean(), color=viz.MUTED, linewidth=1)
    ax.text(0.01, y.mean() + 0.015, f"no-skill = fraud rate {y.mean():.1%}", fontsize=8, color=viz.MUTED, ha="left")
    ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_ylim(0, 1.02)
    ax.set_title("RQ6: detectors trained only on legitimate rows vs the supervised model (Exp C, v1.1 test)")
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    return fig


def fig_detector_bars(results: pd.DataFrame):
    metrics = [("pr_auc", "PR-AUC"), ("recall_at_1pct_fpr", "recall at 1% false-positive rate"),
               ("p95_borderline_flag_rate", "BORDERLINE rows flagged at p95")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    y = np.arange(len(DETECTORS))
    for ax, (col, title) in zip(axes, metrics):
        for k, exp in enumerate(("B", "C")):
            d = results[results.exp == exp].set_index("detector").loc[DETECTORS]
            ax.barh(y + (0.5 - k) * 0.38, d[col], height=0.36, color=EXP_COLORS[exp], label=EXP_NAMES[exp])
            for i, v in enumerate(d[col]):
                ax.text(v + 0.01, y[i] + (0.5 - k) * 0.38, f"{v:.2f}", va="center", fontsize=8, color=viz.INK_2)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [DETECTOR_LABELS[d] for d in DETECTORS])
    axes[0].invert_yaxis()
    fig.suptitle("Anomaly detectors on the test set: network features help the Isolation Forest most; every detector flags most BORDERLINE rows",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=2, fontsize=9)
    return fig


def fig_scenario_heatmap(per_scen: pd.DataFrame):
    cols = DETECTORS + ["xgboost_supervised"]
    d = per_scen[per_scen["n"] >= 5].copy()
    d["row"] = d["scenario"] + " · " + d["risk_label"].astype(str).str.lower()
    order = sorted(d["row"].unique(), key=lambda s: (int(s.split()[0][1:]), s))
    piv = d.pivot_table(index="row", columns="detector", values="flag_rate").reindex(index=order, columns=cols)
    n = d.drop_duplicates("row").set_index("row")["n"].reindex(order)
    fig, ax = plt.subplots(figsize=(11, 0.36 * len(order) + 2))
    im = ax.imshow(piv.to_numpy(), cmap=viz.SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(order)), [f"{r}  (n={n[r]})" for r in order], fontsize=8.5)
    ax.set_xticks(range(len(cols)), [DETECTOR_LABELS.get(c, "XGBoost\n(supervised)").replace(" (", "\n(") for c in cols], fontsize=8.5)
    ax.axvline(len(DETECTORS) - 0.5, color=viz.SURFACE, linewidth=5)
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else viz.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, format=plt.FuncFormatter(PCT)); cb.outline.set_visible(False)
    ax.set_title("Flag rate per scenario, test set (detectors at p95; XGBoost at its validation threshold)", pad=16)
    viz.subtitle(ax, "fraudulent rows: higher is better (recall). Legitimate / borderline rows: lower is better (false alarms)")
    fig.tight_layout()
    return fig


def fig_autoencoder(hist: pd.DataFrame, err: np.ndarray, test_frame: pd.DataFrame, thr: float):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    a1.plot(hist["epoch"], hist["train_loss"], color=viz.SERIES[0], label="train (legit train rows)")
    a1.plot(hist["epoch"], hist["val_loss"], color=viz.SERIES[1], label="validation (legit val rows)")
    best = hist.loc[hist["val_loss"].idxmin()]
    a1.scatter([best.epoch], [best.val_loss], s=70, color=viz.INK, zorder=5)
    a1.annotate(f"best epoch {int(best.epoch)}", (best.epoch, best.val_loss), xytext=(-10, 18), textcoords="offset points",
                fontsize=8, ha="right")
    a1.set_yscale("log"); a1.set_xlabel("epoch"); a1.set_ylabel("MSE (log scale)")
    a1.set_title("Autoencoder training with early stopping", fontsize=10)
    a1.legend(fontsize=8)
    lab = test_frame["risk_label"].astype(str).to_numpy()
    bins = np.logspace(np.log10(max(err.min(), 1e-4)), np.log10(err.max()), 50)
    for l in viz.LABEL_ORDER:
        a2.hist(err[lab == l], bins=bins, density=True, histtype="step", linewidth=2, color=viz.LABEL_COLORS[l],
                label=l.title())
    a2.axvline(thr, color=viz.MUTED, linewidth=1)
    a2.text(thr, a2.get_ylim()[1] * 0.95, " p95 threshold", fontsize=8, color=viz.INK_2, va="top")
    a2.set_xscale("log"); a2.set_xlabel("reconstruction error (log scale)"); a2.set_ylabel("density")
    a2.set_title("Reconstruction error by label on the test set", fontsize=10)
    a2.legend(fontsize=8)
    fig.suptitle("Autoencoder (d → 32 → 16 → 8 → 16 → 32 → d), trained only on legitimate rows, served as ONNX",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def fig_complementarity(comp: dict):
    caught = comp["caught_by"]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    y = np.arange(len(DETECTORS))
    vals = [caught[d] for d in DETECTORS]
    ax.barh(y, vals, color=[DET_COLORS[d] for d in DETECTORS], height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + 0.2, i, f"{v} of {comp['xgb_missed']}", va="center", fontsize=9)
    ax.set_yticks(y, [DETECTOR_LABELS[d] for d in DETECTORS])
    ax.invert_yaxis()
    ax.set_xlim(0, comp["xgb_missed"] * 1.2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("test frauds missed by XGBoost that the detector flags (p95)")
    ax.set_title(f"Complementarity: XGBoost misses {comp['xgb_missed']} of {comp['test_frauds']} test frauds")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_paysim(ps: dict):
    y = ps["_y"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    cols = {"raw": viz.SERIES[0], "raw+balance_error": viz.SERIES[1]}
    names = {"raw": "amount + balances", "raw+balance_error": "+ balance-error features"}
    for k, s in ps["_scores"].items():
        p, r, _ = precision_recall_curve(y, s)
        a1.plot(r, p, color=cols[k], rasterized=True, label=f"{names[k]} (PR-AUC {ps['feature_sets'][k]['pr_auc']:.3f})")
    a1.axhline(y.mean(), color=viz.MUTED, linewidth=1)
    a1.text(0.01, y.mean() + 0.015, f"no-skill = {y.mean():.2%}", fontsize=8, color=viz.MUTED, ha="left")
    a1.set_xlabel("recall"); a1.set_ylabel("precision"); a1.set_ylim(0, 1.02)
    a1.set_title("Isolation Forest vs PaySim isFraud", fontsize=10)
    a1.legend(fontsize=8.5, loc="upper right")
    a2.axis("off")
    fs = ps["feature_sets"]
    rows = [["rows scored", f"{ps['rows_scored']:,}"], ["fraud rows", f"{ps['fraud_rows']:,} ({ps['fraud_rate']:.2%})"],
            ["fit sample (no labels)", f"{ps['fit_sample']:,}"], ["Polars lazy wrangle", f"{ps['wrangle_seconds']:.1f} s"]]
    for k in fs:
        rows += [[f"{names[k]}: fit / score", f"{fs[k]['fit_seconds']:.1f} s / {fs[k]['score_seconds']:.1f} s"],
                 [f"{names[k]}: rows per second", f"{fs[k]['rows_per_second']:,}"]]
    rows.append(["peak process memory", f"{ps['peak_rss_mb']:,.0f} MB"])
    t = a2.table(cellText=rows, colWidths=[0.62, 0.38], loc="center", cellLoc="left", edges="horizontal")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(viz.GRID)
    a2.set_title("Scale track: one laptop, 8 CPU cores", fontsize=10, loc="left")
    fig.suptitle("PaySim big-data track: unsupervised Isolation Forest on 2.77 M TRANSFER / CASH_OUT rows",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def render_all(m, sp, cl, an, ps, summary) -> list:
    viz.apply_style()
    out = []
    S = summary["clustering"]
    chosen = {e: S[e]["k"] for e in ("B", "C")}
    main = cl["main"]
    km_labels = main["kmeans"].labels_
    cfr = pd.Series(m["is_fraud"].to_numpy()).groupby(km_labels).mean().to_dict()
    te = m.iloc[sp.test]
    y_te = te["is_fraud"].to_numpy()
    figs = {
        "unsup_01_kmeans_selection": fig_kmeans_selection(cl["sweeps"], chosen),
        "unsup_02_cluster_composition": fig_cluster_composition(cl["clusters"], S["C"]["profile"]),
        "unsup_03_embeddings": fig_embeddings(main["proj"], m, km_labels, cfr),
        "unsup_04_dbscan": fig_dbscan(main["kdist"], main["eps"], S["C"]["dbscan"]),
        "unsup_05_detector_pr": fig_detector_pr(an["scores"], an["xgb_proba"], y_te),
        "unsup_06_detector_bars": fig_detector_bars(an["results"]),
        "unsup_07_scenario_heatmap": fig_scenario_heatmap(an["per_scenario"]),
        "unsup_08_autoencoder": fig_autoencoder(an["ae_history"], an["scores"]["autoencoder"], te,
                                                an["detectors"]["autoencoder"].threshold(95)),
        "unsup_09_complementarity": fig_complementarity(an["complementarity"]),
    }
    if ps is not None:
        figs["unsup_10_paysim_if"] = fig_paysim(ps)
    for name, fig in figs.items():
        out.append(viz.save(fig, name))
        plt.close(fig)
    summary["figures"] = [p.name for p in out]
    return out
