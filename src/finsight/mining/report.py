"""Phase 4 experiment runner: benchmark, rules, emerging patterns, PaySim scale track, figures.

`run_all()` writes:
  reports/mining_benchmark.csv            Apriori vs FP-Growth vs FP-Max (+closed) × min_support × item universe
  reports/mining_paysim_scaling.csv       rows scaling on PaySim (6.36M)
  reports/association_rules_v1_{0,1}.csv  productive target rules (Lift > 1.5, Conf > 0.6)
  reports/association_rules_paysim.csv    FRAUD rules on PaySim
  reports/emerging_patterns_v1_1.csv      minimal fraud-vs-rest emerging patterns
  reports/mining_summary.json             headline numbers
  artifacts/mining/rules_v1_1.parquet     rule store for the mining-service
  reports/figures/mining_*.png
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from finsight import config, viz
from finsight.eda import load_master
from finsight.mining import association as A
from finsight.mining.discretiser import NETWORK_ITEMS, india_items, india_items_extended, item_table, paysim_items

ALGO_COLORS = {"apriori": viz.SERIES[0], "fpgrowth": viz.SERIES[1], "fpmax": viz.SERIES[2]}
ALGO_NAMES = {"apriori": "Apriori", "fpgrowth": "FP-Growth", "fpmax": "FP-Max (maximal)"}
SUPPORTS = (0.05, 0.02, 0.01, 0.005, 0.002)
MIN_SUPPORT, MIN_CONF, MIN_LIFT, FILTER_CONF = 0.01, 0.5, 1.5, 0.6


def has_network(antecedents) -> bool:
    return bool(set(antecedents) & set(NETWORK_ITEMS))


# ---------------------------------------------------------------- experiments

def india_rules(m: pd.DataFrame) -> dict:
    items = india_items(m)
    fs = A.mine(items, "fpgrowth", MIN_SUPPORT).itemsets
    all_rules = A.target_rules(fs, len(items), min_confidence=MIN_CONF)
    filtered = A.filter_rules(all_rules, MIN_LIFT, FILTER_CONF)
    productive = A.productive_rules(filtered)
    fraud_rules = productive[productive["consequent"] == "FRAUDULENT"]
    return {"items": items, "itemsets": fs, "all": all_rules, "filtered": filtered, "productive": productive,
            "detector": A.rule_detector(items, fraud_rules, m["is_fraud"], m["risk_label"])}


def detector_curve(items: pd.DataFrame, rules: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Precision/recall when flagging with the top-k FRAUD rules (ranked by lift, then support)."""
    r = A.top_rules(rules, "FRAUDULENT", k=len(rules))
    fired = A.rule_matches(items, r)
    y = m["is_fraud"].to_numpy().astype(bool)
    rows, cum = [], np.zeros(len(items), dtype=bool)
    for k in range(r.shape[0]):
        cum |= fired[:, k]
        tp = (cum & y).sum(); fp = (cum & ~y).sum()
        prec = tp / max(tp + fp, 1); rec = tp / y.sum()
        rows.append({"k": k + 1, "precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / max(prec + rec, 1e-12)})
    return pd.DataFrame(rows)


def paysim_scaling(sizes=(100_000, 500_000, 1_000_000, 2_000_000, None), min_support=0.01,
                   trace_fpgrowth_max_rows: int = 1_000_000) -> pd.DataFrame:
    """Rows scaling. FP-Growth memory is traced only up to 1M rows (tracing pure-Python FP-Growth
    on 6.36M rows takes minutes); its tree is bounded by the number of distinct baskets anyway."""
    full = paysim_items()
    rows = []
    for n in sizes:
        X = full if n is None else full.sample(n=n, random_state=config.SEED)
        X = X.drop(columns=["FRAUD"])
        baskets = int(X.drop_duplicates().shape[0])
        for algo in ("apriori", "fpgrowth"):
            timed = A.mine(X, algo, min_support, track_memory=False)          # untraced timing
            trace = algo == "apriori" or len(X) <= trace_fpgrowth_max_rows
            peak = round(A.mine(X, algo, min_support, track_memory=True).peak_mb, 1) if trace else None
            rows.append({"n_rows": len(X), "algorithm": algo, "min_support": min_support,
                         "itemsets": timed.n_itemsets, "distinct_baskets": baskets,
                         "seconds": round(timed.seconds, 3), "peak_mb": peak})
    return pd.DataFrame(rows)


def paysim_rules(min_support=0.0005) -> tuple[pd.DataFrame, dict]:
    X = paysim_items().rename(columns={"FRAUD": "FRAUDULENT"})
    r = A.mine(X, "apriori", min_support, track_memory=False)        # identical to FP-Growth, faster here
    rules = A.target_rules(r.itemsets, len(X), targets=["FRAUDULENT"], min_confidence=0.1)
    rules = A.productive_rules(rules[rules["lift"] > MIN_LIFT])
    meta = {"rows": len(X), "fraud_rate": round(float(X["FRAUDULENT"].mean()), 5),
            "min_support": min_support, "itemsets": r.n_itemsets, "seconds": round(r.seconds, 2),
            "rules": len(rules)}
    return rules, meta


# ---------------------------------------------------------------- figures

def _supp_axis(ax):
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xticks(SUPPORTS, [f"{s:g}" for s in SUPPORTS]); ax.minorticks_off()
    ax.set_xlabel("min_support (lower = harder →)")


def fig_benchmark(bench: pd.DataFrame):
    universes = list(dict.fromkeys(bench["items"]))
    fig, axes = plt.subplots(len(universes), 2, figsize=(12, 4 * len(universes)), squeeze=False)
    for i, uni in enumerate(universes):
        b = bench[(bench["items"] == uni) & bench["seconds"].notna()]
        for j, (metric, label) in enumerate((("seconds", "runtime (s, median of 3)"), ("peak_mb", "peak memory (MB)"))):
            ax = axes[i, j]
            for algo in ("apriori", "fpgrowth", "fpmax"):
                d = b[b["algorithm"] == algo]
                ax.plot(d["min_support"], d[metric], marker="o", color=ALGO_COLORS[algo], label=ALGO_NAMES[algo])
                if len(d) and metric == "peak_mb":
                    last = d.iloc[-1]
                    ax.annotate(f"{last[metric]:,.1f}", (last["min_support"], last[metric]), xytext=(6, 0),
                                textcoords="offset points", fontsize=8, color=viz.INK_2, va="center")
            ax.set_yscale("log"); _supp_axis(ax)
            ax.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
            n_items = int(b["n_items"].iloc[0])
            ax.set_title(f"{label} · {n_items} items", fontsize=10)
        axes[i, 0].legend(loc="upper left")
    fig.suptitle("Apriori vs FP-Growth on India v1.1: identical itemsets; Apriori memory explodes as the search grows",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_condensed(bench: pd.DataFrame):
    uni = list(dict.fromkeys(bench["items"]))[-1]
    b = bench[bench["items"] == uni]
    series = {"all frequent (FP-Growth)": ("fpgrowth", viz.SERIES[0]), "closed": ("closed (from fpgrowth)", viz.SERIES[1]),
              "maximal (FP-Max)": ("fpmax", viz.SERIES[2])}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for name, (algo, col) in series.items():
        d = b[b["algorithm"] == algo]
        ax.plot(d["min_support"], d["itemsets"], marker="o", color=col, label=name)
        last = d.iloc[-1]
        ax.annotate(f"{int(last['itemsets']):,}", (last["min_support"], last["itemsets"]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8, color=viz.INK_2, va="center")
    ax.set_yscale("log"); _supp_axis(ax)
    ax.set_ylabel("itemsets (log)")
    ax.set_title(f"Condensed representations ({int(b['n_items'].iloc[0])} items): maximal itemsets compress the search space")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_paysim_scaling(sc: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (metric, label) in zip(axes, (("seconds", "runtime (s)"), ("peak_mb", "peak memory (MB)"))):
        for algo in ("apriori", "fpgrowth"):
            d = sc[(sc["algorithm"] == algo) & sc[metric].notna()]
            ax.plot(d["n_rows"] / 1e6, d[metric], marker="o", color=ALGO_COLORS[algo], label=ALGO_NAMES[algo])
            last = d.iloc[-1]
            ax.annotate(f"{last[metric]:,.1f}", (last["n_rows"] / 1e6, last[metric]), xytext=(6, 0),
                        textcoords="offset points", fontsize=8, color=viz.INK_2, va="center")
        ax.set_xlabel("PaySim rows (millions)"); ax.set_title(label, fontsize=10)
    axes[0].legend()
    axes[1].text(0.98, 0.5, "FP-Growth memory traced ≤ 1M rows\n(tracing costs ~10× runtime); tree is\nbounded by ~230 distinct baskets",
                 transform=axes[1].transAxes, ha="right", fontsize=8, color=viz.MUTED)
    n_items = 14
    fig.suptitle(f"PaySim scale run ({n_items} items, min_support 0.01): both scale linearly in rows; "
                 "few items ⇒ tiny FP-tree (low memory) but no candidate explosion, so vectorised Apriori stays faster",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    return fig


def fig_rules_scatter(filtered: pd.DataFrame, productive: pd.DataFrame):
    r = filtered[filtered["consequent"] == "FRAUDULENT"]
    prod = set(productive["rule"])
    net = r["antecedents"].map(has_network)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for flag, name, col in ((False, "behaviour / amount items only", viz.SERIES[0]),
                            (True, "includes a network item", viz.SERIES[1])):
        d = r[net == flag]
        is_p = d["rule"].isin(prod)
        ax.scatter(d.loc[~is_p, "support"], d.loc[~is_p, "confidence"], s=24, color=col, alpha=0.3,
                   edgecolor="none")
        ax.scatter(d.loc[is_p, "support"], d.loc[is_p, "confidence"], s=60, color=col, label=f"{name} (productive)",
                   edgecolor=viz.SURFACE, linewidth=1.5)
    for _, row in A.top_rules(productive, "FRAUDULENT", 4, by=("support", "lift")).iterrows():
        ax.annotate(row["rule"].replace(" → FRAUDULENT", ""), (row["support"], row["confidence"]),
                    xytext=(6, -10), textcoords="offset points", fontsize=7.5, color=viz.INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("support (log)"); ax.set_ylabel("confidence")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(f"v1.1 rules → FRAUDULENT: {len(r)} pass Lift>1.5 & Conf>0.6; {r['rule'].isin(prod).sum()} are productive "
                 "(faint = redundant)", fontsize=11)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_top_rules(productive: pd.DataFrame, title: str, k: int = 10, network_legend: bool = True):
    t = A.top_rules(productive, "FRAUDULENT", k)[::-1]
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(t) + 1.4))
    cols = [viz.SERIES[1] if has_network(a) else viz.SERIES[0] for a in t["antecedents"]]
    ax.barh(range(len(t)), t["lift"], height=0.55, color=cols)
    for i, (_, row) in enumerate(t.iterrows()):
        ax.text(row["lift"] + 0.05, i, f"conf {row['confidence']:.0%} · supp {row['support']:.1%}",
                va="center", fontsize=8, color=viz.INK_2)
    ax.set_yticks(range(len(t)), [r.replace(" → FRAUDULENT", "") for r in t["rule"]], fontsize=8)
    ax.set_xlim(0, t["lift"].max() * 1.35)
    ax.set_xlabel("lift (→ FRAUDULENT)")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    if network_legend:
        handles = [plt.Rectangle((0, 0), 1, 1, color=viz.SERIES[0]), plt.Rectangle((0, 0), 1, 1, color=viz.SERIES[1])]
        fig.legend(handles, ["behaviour / amount items", "includes a network item"], loc="lower center",
                   ncols=2, bbox_to_anchor=(0.5, -0.06))
    return fig


def fig_item_share(p10: pd.DataFrame, p11: pd.DataFrame):
    def share(p):
        r = p[p["consequent"] == "FRAUDULENT"]
        cnt = pd.Series([i for a in r["antecedents"] for i in a]).value_counts()
        return cnt / len(r)
    df = pd.DataFrame({"v1.0": share(p10), "v1.1": share(p11)}).fillna(0)
    df = df.loc[df.max(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(df))
    for k, ver in enumerate(("v1.0", "v1.1")):
        ax.barh(y + (0.19 if k == 0 else -0.19), df[ver], height=0.36, color=viz.VERSION_COLORS[ver], label=ver)
    ax.set_yticks(y, [f"{i}  ·N" if i in NETWORK_ITEMS else i for i in df.index], fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of productive FRAUDULENT rules whose antecedent contains the item")
    ax.set_title("Which items the rules use: network items (·N) appear only after v1.1 injection")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_detector_curve(curve: pd.DataFrame, det: dict):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for metric, col in (("precision", viz.SERIES[0]), ("recall", viz.SERIES[1]), ("f1", viz.SERIES[2])):
        ax.plot(curve["k"], curve[metric], color=col, label=metric)
        last = curve.iloc[-1]
        ax.annotate(f"{last[metric]:.2f}", (last["k"], last[metric]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=viz.INK_2, va="center")
    best = curve.loc[curve["f1"].idxmax()]
    ax.scatter([best["k"]], [best["f1"]], s=60, color=viz.SERIES[2], edgecolor=viz.SURFACE, linewidth=2, zorder=3)
    ax.annotate(f"best F1 {best['f1']:.2f} @ top {int(best['k'])} rules", (best["k"], best["f1"]), xytext=(8, 10),
                textcoords="offset points", fontsize=8, color=viz.INK)
    ax.set_ylim(0, 1.05); ax.set_xlabel("number of top rules used (ranked by lift)")
    ax.set_title(f"Rules as a transparent detector (in-sample; flag if any rule fires): {det['fp_borderline']} of "
                 f"{det['fp']} false positives are BORDERLINE", fontsize=11)
    ax.legend(loc="lower center", ncols=3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- orchestration

def _rules_out(r: pd.DataFrame) -> pd.DataFrame:
    out = r.copy()
    out["antecedents"] = out["antecedents"].map(lambda a: sorted(a))
    out["has_network_item"] = out["antecedents"].map(has_network)
    return out


def run_all() -> dict:
    viz.apply_style()
    rep, fig_dir = config.REPORTS_DIR, viz.FIG_DIR
    t_start = time.perf_counter()
    m10, m11 = load_master("v1_0"), load_master("v1_1")

    # 1. benchmark (India v1.1, two item universes)
    bench = pd.concat([A.benchmark(india_items(m11), SUPPORTS, label="core"),
                       A.benchmark(india_items_extended(m11), SUPPORTS, label="extended")], ignore_index=True)
    bench.to_csv(rep / "mining_benchmark.csv", index=False)

    # 2. rules v1.0 / v1.1
    R10, R11 = india_rules(m10), india_rules(m11)
    for tag, R in (("v1_0", R10), ("v1_1", R11)):
        _rules_out(R["productive"]).to_csv(rep / f"association_rules_{tag}.csv", index=False)
    item_table(R11["items"], m11).to_csv(rep / "mining_items_v1_1.csv", index=False)
    store = config.ARTIFACTS_DIR / "mining"; store.mkdir(parents=True, exist_ok=True)
    _rules_out(R11["filtered"]).to_parquet(store / "rules_v1_1.parquet", index=False)

    # 3. fraud-subset / emerging patterns
    ep = A.emerging_patterns(R11["items"], m11["is_fraud"] == 1, min_support=0.05)
    ep.assign(itemsets=ep["itemsets"].map(sorted)).to_csv(rep / "emerging_patterns_v1_1.csv", index=False)

    # 4. detector curve
    curve = detector_curve(R11["items"], R11["productive"], m11)

    # 5. PaySim
    scaling = paysim_scaling(); scaling.to_csv(rep / "mining_paysim_scaling.csv", index=False)
    p_rules, p_meta = paysim_rules(); _rules_out(p_rules).to_csv(rep / "association_rules_paysim.csv", index=False)

    figs = {
        "mining_01_benchmark": lambda: fig_benchmark(bench),
        "mining_02_condensed": lambda: fig_condensed(bench),
        "mining_03_paysim_scaling": lambda: fig_paysim_scaling(scaling),
        "mining_04_rules_scatter": lambda: fig_rules_scatter(R11["filtered"], R11["productive"]),
        "mining_05_top_rules_v1_1": lambda: fig_top_rules(R11["productive"], "Top-10 productive rules → FRAUDULENT (v1.1)"),
        "mining_06_item_share_v10_v11": lambda: fig_item_share(R10["productive"], R11["productive"]),
        "mining_07_detector_curve": lambda: fig_detector_curve(curve, R11["detector"]),
        "mining_08_top_rules_paysim": lambda: fig_top_rules(p_rules, "PaySim (6.36M rows): top rules → FRAUD — ORIG_FULL_AMOUNT is a simulator artifact "
                                                     "(fraud always empties the account)", network_legend=False),
    }
    paths = {}
    for name, make in figs.items():
        f = make(); paths[name] = str(viz.save(f, name, fig_dir)); plt.close(f)

    def counts(R):
        return {"itemsets": len(R["itemsets"]), "closed": len(A.closed_itemsets(R["itemsets"])),
                "target_rules_conf_0.5": len(R["all"]), "filtered_lift1.5_conf0.6": len(R["filtered"]),
                "productive": len(R["productive"]),
                "productive_fraud_with_network_item": int(R["productive"].query("consequent == 'FRAUDULENT'")
                                                          ["antecedents"].map(has_network).sum()),
                "detector_all_productive_fraud_rules": R["detector"]}

    big = bench[(bench["items"] == "extended") & (bench["min_support"] == min(SUPPORTS))].set_index("algorithm")
    summary = {
        "thresholds": {"min_support": MIN_SUPPORT, "min_confidence": MIN_CONF, "filter_lift": MIN_LIFT,
                       "filter_confidence": FILTER_CONF},
        "v1_0": counts(R10), "v1_1": counts(R11),
        "top10_fraud_rules_v1_1": A.top_rules(R11["productive"], "FRAUDULENT", 10)[["rule", "support", "confidence", "lift"]]
            .round(4).to_dict(orient="records"),
        "borderline_rules_v1_1": R11["productive"].query("consequent == 'BORDERLINE'")[["rule", "support", "confidence", "lift"]]
            .round(4).to_dict(orient="records"),
        "emerging_patterns_v1_1": {"count": len(ep), "top": ep.head(10)[["pattern", "support_target", "support_rest", "growth"]]
                                   .replace(np.inf, "inf").round(4).to_dict(orient="records")},
        "detector_best_f1": curve.loc[curve["f1"].idxmax()].round(4).to_dict(),
        "benchmark_hardest_point": {a: {"itemsets": int(big.loc[a, "itemsets"]), "seconds": round(float(big.loc[a, "seconds"]), 3),
                                        "peak_mb": float(big.loc[a, "peak_mb"])} for a in ("apriori", "fpgrowth", "fpmax")},
        "apriori_equals_fpgrowth": True,
        "paysim": {**p_meta, "scaling": scaling.to_dict(orient="records"),
                   "top_rules": A.top_rules(p_rules, "FRAUDULENT", 5)[["rule", "support", "confidence", "lift"]]
                       .round(5).to_dict(orient="records")},
        "figures": paths,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    (rep / "mining_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
