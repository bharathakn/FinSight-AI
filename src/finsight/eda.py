"""Phase 2 — Exploratory Data Analysis (Plan §10, PROGRESS Phase 2).

Each `fig_*` function takes master tables, returns a matplotlib Figure and is pure (no I/O).
`run_all()` renders every figure to reports/figures/eda_*.png and writes reports/eda_summary.json.
The notebook 02_eda.ipynb calls the same functions, so notebook, report and pipeline agree.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from finsight import config, viz
from finsight.graph.network_features import NETWORK_FEATURES
from finsight.ml.feature_sets import BEHAVIOURAL_NUMERIC, NETWORK_NUMERIC, TRANSACTION_NUMERIC

NETWORK_SCENARIOS = {
    "SCENARIO_3_RAPID_BURST": "S3 burst", "SCENARIO_6_STRUCTURING_SPLITTING": "S6 structuring",
    "SCENARIO_7_LAYERING_NETWORK": "S7 layering", "SCENARIO_8_MULE_PASS_THROUGH": "S8 mule",
    "SCENARIO_9_FRAUD_RING_HUB": "S9 ring hub", "SCENARIO_11_SHARED_DEVICE_IP": "S11 shared device",
}


def short_scenario(s: str) -> str:
    parts = str(s).split("_")
    return f"S{parts[1]} " + " ".join(p.lower() for p in parts[2:])


# ---------------------------------------------------------------- data access

def load_master(version: str = "v1_1") -> pd.DataFrame:
    m = pd.read_parquet(config.CURATED_DIR / f"master_features_{version}.parquet")
    m["risk_label"] = pd.Categorical(m["risk_label"].astype(str), categories=viz.LABEL_ORDER)
    m["scenario"] = m["scenario_id"].astype(str).map(short_scenario)
    return m


def camouflaged_ids(directory: Path | None = None) -> set[str]:
    log = pd.read_csv(Path(directory or config.INDIA_V1_1_DIR) / "injection_log.csv")
    return set(log.loc[log["group_id"] == "camouflage", "transaction_id"])


def behaviour_flags(m: pd.DataFrame) -> pd.DataFrame:
    """Discretised behaviour items (same thresholds as Plan §6.4)."""
    return pd.DataFrame({
        "NEW_DEVICE": m["is_new_device"] == 1,
        "NEW_BENEFICIARY": m["is_new_beneficiary"] == 1,
        "RECENT_BENEFICIARY (<24h)": m["time_since_beneficiary_added_min"] < 1440,
        "FAR_FROM_HOME (>100km)": m["distance_from_home_km"] > 100,
        "HIGH_AMOUNT_SPIKE (>5×)": m["amount_ratio_vs_30d_avg"] > 5,
        "HIGH_VALUE (≥₹50k)": m["amount_inr"] >= 50_000,
        "NEAR_THRESHOLD (₹45–50k)": m["amount_inr"].between(45_000, 49_999.99),
        "ODD_HOUR (<6 or >22)": (m["hour_of_day"] < 6) | (m["hour_of_day"] > 22),
        "PASS_THROUGH": m["is_pass_through"] == 1,
        "SHARED_DEVICE (≥3 accts)": m["device_distinct_accounts"] >= 3,
    }, index=m.index)


# ---------------------------------------------------------------- tables

def single_feature_auc(m: pd.DataFrame, features: list[str], target: str = "is_fraud") -> pd.Series:
    y = m[target]
    out = {}
    for c in features:
        x = m[c].astype(float)
        if x.nunique() < 2:
            continue
        a = roc_auc_score(y, x.fillna(x.median()))
        out[c] = max(a, 1 - a)
    return pd.Series(out).sort_values(ascending=False)


def feature_ranking(m: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """RQ1: rank all numeric A/B/C features by single-feature AUC and mutual information."""
    feats = TRANSACTION_NUMERIC + BEHAVIOURAL_NUMERIC + NETWORK_NUMERIC
    auc = single_feature_auc(m, feats)
    X = m[auc.index].astype(float).fillna(0)
    mi = pd.Series(mutual_info_classif(X, m["is_fraud"], random_state=seed), index=auc.index)
    group = {**{f: "A transaction" for f in TRANSACTION_NUMERIC}, **{f: "B behavioural" for f in BEHAVIOURAL_NUMERIC},
             **{f: "C network" for f in NETWORK_NUMERIC}}
    df = pd.DataFrame({"auc": auc.round(4), "mutual_info": mi.round(4)})
    df["group"] = df.index.map(group)
    return df.sort_values("auc", ascending=False)


def flag_table(m: pd.DataFrame) -> pd.DataFrame:
    """P(flag | label) and fraud rate / lift given the flag."""
    flags = behaviour_flags(m)
    base = m["is_fraud"].mean()
    rows = []
    for f in flags.columns:
        on = flags[f]
        rate = m.loc[on, "is_fraud"].mean() if on.any() else np.nan
        row = {"flag": f, "support": round(on.mean(), 4), "fraud_rate_given_flag": round(rate, 4),
               "lift": round(rate / base, 2) if on.any() else np.nan}
        for lab in viz.LABEL_ORDER:
            row[f"share_{lab}"] = round(on[m["risk_label"] == lab].mean(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

def fig_label_overview(m: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 2.2]})
    counts = m["risk_label"].value_counts().reindex(viz.LABEL_ORDER)
    a1.barh(counts.index[::-1], counts.values[::-1], height=0.55,
            color=[viz.LABEL_COLORS[k] for k in counts.index[::-1]])
    for i, (k, v) in enumerate(zip(counts.index[::-1], counts.values[::-1])):
        a1.text(v + 80, i, f"{v:,}  ({v / len(m):.1%})", va="center", fontsize=9, color=viz.INK_2)
    a1.set_xlim(0, counts.max() * 1.45)
    a1.set_title("Risk label balance")
    a1.grid(axis="y", visible=False)

    ct = pd.crosstab(m["scenario"], m["risk_label"]).reindex(columns=viz.LABEL_ORDER, fill_value=0)
    ct = ct.drop(index=[i for i in ct.index if i.startswith("S1 ")])
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    left = np.zeros(len(ct))
    for lab in viz.LABEL_ORDER:
        if ct[lab].sum() == 0:
            continue
        a2.barh(ct.index, ct[lab], left=left, height=0.6, color=viz.LABEL_COLORS[lab], label=lab,
                edgecolor=viz.SURFACE, linewidth=1.5)
        left += ct[lab].to_numpy()
    a2.set_title("Suspicious scenarios (S1 normal = 8,200 rows omitted)")
    a2.legend(loc="lower right")
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("transactions")
    fig.tight_layout()
    return fig


def fig_amount_by_label(m: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, ch in zip(axes, ["UPI", "NEFT", "RTGS"]):
        sub = m[m["channel"] == ch]
        data = [np.log10(sub.loc[sub["risk_label"] == lab, "amount_inr"]) for lab in viz.LABEL_ORDER]
        parts = ax.violinplot(data, showmedians=True, widths=0.75)
        for body, lab in zip(parts["bodies"], viz.LABEL_ORDER):
            body.set_facecolor(viz.LABEL_COLORS[lab]); body.set_alpha(0.55); body.set_edgecolor("none")
        for k in ("cbars", "cmins", "cmaxes", "cmedians"):
            parts[k].set_color(viz.INK_2); parts[k].set_linewidth(1)
        ax.set_xticks([1, 2, 3], ["Legit", "Fraud", "Border."])
        ax.set_title(f"{ch}  (n={len(sub):,})")
        ax.grid(axis="x", visible=False)
    ticks = np.arange(1, 8)
    axes[0].set_yticks(ticks, [f"₹{10**t:,.0f}" if t < 6 else f"₹{10**t / 1e5:,.0f}L" for t in ticks])
    axes[0].set_ylabel("amount (log scale)")
    fig.suptitle("Transaction amount by risk label and channel", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_fraud_rate_time_channel(m: pd.DataFrame):
    base = m["is_fraud"].mean()
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 2.4, 1.4]})
    by_ch = m.groupby("channel", observed=True)["is_fraud"].mean().reindex(["UPI", "NEFT", "RTGS"])
    a1.bar(by_ch.index, by_ch.values, width=0.5, color=viz.SERIES[0])
    for i, v in enumerate(by_ch.values):
        a1.text(i, v + 0.004, f"{v:.1%}", ha="center", fontsize=9, color=viz.INK_2)
    a1.set_title("by channel")
    by_h = m.groupby("hour_of_day")["is_fraud"].mean()
    a2.plot(by_h.index, by_h.values, color=viz.SERIES[0], marker="o", markersize=4)
    a2.set_xticks(range(0, 24, 3)); a2.set_xlabel("hour of day (IST)")
    a2.set_title("by hour")
    dow = m.groupby("day_of_week")["is_fraud"].mean()
    a3.bar(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], dow.values, width=0.55, color=viz.SERIES[0])
    a3.set_title("by day of week")
    for ax in (a1, a2, a3):
        ax.axhline(base, color=viz.MUTED, linewidth=1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(0, max(0.25, ax.get_ylim()[1]))
    a1.set_ylabel("fraud rate")
    a3.text(6.4, base, f" base {base:.1%}", va="center", fontsize=8, color=viz.MUTED)
    fig.suptitle("Channel and weekday are flat around the 13.5% base rate; late-night hours (22–02) run higher",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_geo(m: pd.DataFrame):
    city = m.groupby("tx_city", observed=True).agg(lat=("tx_lat", "median"), lon=("tx_long", "median"),
                                                   n=("is_fraud", "size"), rate=("is_fraud", "mean"))
    city = city[city["n"] >= 20]
    state = m.groupby("tx_state", observed=True)["is_fraud"].agg(["mean", "size"])
    state = state[state["size"] >= 50].sort_values("mean")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.2, 1]})
    size = 40 + 360 * (city["n"] - city["n"].min()) / max(1, city["n"].max() - city["n"].min())
    sc = a1.scatter(city["lon"], city["lat"], s=size, c=city["rate"], cmap=viz.SEQ,
                    vmin=city["rate"].min(), vmax=city["rate"].max(), edgecolor=viz.SURFACE, linewidth=1.5)
    for name, r in city.iterrows():
        a1.annotate(name, (r["lon"], r["lat"]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=viz.INK_2)
    a1.set_xlabel("longitude"); a1.set_ylabel("latitude")
    a1.set_xlim(city["lon"].min() - 2, city["lon"].max() + 4); a1.set_ylim(city["lat"].min() - 2, city["lat"].max() + 2)
    a1.set_title(f"{len(city)} cities (bubble = volume {city['n'].min()}–{city['n'].max()}, colour = fraud rate)")
    cb = fig.colorbar(sc, ax=a1, shrink=0.7, format=plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    cb.outline.set_visible(False)
    a2.barh(state.index.astype(str), state["mean"], height=0.6, color=viz.SERIES[0])
    a2.axvline(m["is_fraud"].mean(), color=viz.MUTED, linewidth=1)
    a2.text(m["is_fraud"].mean(), -0.9, f" base {m['is_fraud'].mean():.1%}", fontsize=8, color=viz.MUTED)
    a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_title("Fraud rate by state (≥50 tx)")
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_correlation(m: pd.DataFrame):
    cols = ["is_fraud", "log_amount", "amount_ratio_vs_30d_avg", "distance_from_home_km",
            "time_since_beneficiary_added_min", "is_new_device", "is_new_beneficiary", "tx_frequency_24h",
            "time_since_last_tx_sec", "account_balance_before", "sender_historical_30d_avg_tx_amt_inr",
            "receiver_distinct_senders_24h", "sender_tx_24h", "is_pass_through", "forward_ratio",
            "device_distinct_accounts", "receiver_pagerank"]
    c = m[cols].astype(float).corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(c, cmap=viz.DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            if i != j and abs(c.iloc[i, j]) >= 0.3:
                ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(c.iloc[i, j]) > 0.6 else viz.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.7); cb.outline.set_visible(False)
    ax.set_title("Spearman correlation (|ρ| ≥ 0.3 labelled)")
    fig.tight_layout()
    return fig


def fig_feature_auc(m10: pd.DataFrame, m11: pd.DataFrame, top: int = 25):
    feats = TRANSACTION_NUMERIC + BEHAVIOURAL_NUMERIC + NETWORK_NUMERIC
    a10, a11 = single_feature_auc(m10, feats), single_feature_auc(m11, feats)
    df = pd.DataFrame({"v1.0": a10, "v1.1": a11}).fillna(0.5)
    df = df.loc[df.max(axis=1).sort_values(ascending=False).index[:top]][::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(df))
    for k, (ver, off) in enumerate((("v1.0", -0.19), ("v1.1", 0.19))):
        ax.barh(y + off, df[ver] - 0.5, left=0.5, height=0.36, color=viz.VERSION_COLORS[ver], label=ver)
    labels = [f"{f}  ·N" if f in NETWORK_FEATURES else f for f in df.index]
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.95, color=viz.MUTED, linewidth=1)
    ax.text(0.947, 0, "leak-suspect\nthreshold 0.95", fontsize=8, color=viz.MUTED, va="bottom", ha="right")
    ax.set_xlabel("single-feature ROC-AUC (max(AUC, 1−AUC))")
    ax.set_title(f"Top {top} single-feature AUCs, v1.0 vs v1.1  (·N = network feature)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_behaviour_flags(m: pd.DataFrame):
    t = flag_table(m).set_index("flag")[[f"share_{l}" for l in viz.LABEL_ORDER]][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(t))
    for k, lab in enumerate(viz.LABEL_ORDER):
        ax.barh(y + (1 - k) * 0.26, t[f"share_{lab}"], height=0.24, color=viz.LABEL_COLORS[lab], label=lab)
    ax.set_yticks(y, t.index, fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of transactions in the label group that carry the flag")
    ax.set_title("Behaviour flags by risk label — RECENT_BENEFICIARY and FAR_FROM_HOME never occur on legit rows")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


MEAN_FEATURES = {"is_new_device", "is_new_beneficiary", "is_pass_through", "chain_depth", "sender_tx_10m",
                 "sender_tx_1h", "sender_tx_24h", "receiver_distinct_senders_24h", "receiver_tx_in_24h",
                 "device_distinct_accounts", "ip_distinct_accounts", "same_community", "tx_frequency_24h"}


def _scenario_table(m: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Per-scenario summary: mean for flags/counts, median for skewed continuous features."""
    g = m.groupby("scenario")
    out = pd.DataFrame({f: (g[f].mean() if f in MEAN_FEATURES else g[f].median()) for f in feats})
    return out.loc[sorted(out.index, key=lambda s: int(s.split()[0][1:]))]


def network_flags(m: pd.DataFrame) -> pd.DataFrame:
    """Simple network red flags (used to show what the graph sees that behaviour cannot)."""
    f = pd.DataFrame({
        "receiver fan-in ≥4 (24h)": m["receiver_distinct_senders_24h"] >= 4,
        "sender burst ≥2 (24h)": m["sender_tx_24h"] >= 2,
        "pass-through": m["is_pass_through"] == 1,
        "inflow <1h before": m["sec_since_sender_last_inflow"] < 3600,
        "shared device ≥3 accts": m["device_distinct_accounts"] >= 3,
    }, index=m.index)
    f["any network flag"] = f.any(axis=1)
    return f


def fig_scenario_profile(m: pd.DataFrame):
    feats = ["amount_ratio_vs_30d_avg", "log_amount", "distance_from_home_km", "time_since_beneficiary_added_min",
             "is_new_device", "is_new_beneficiary", "tx_frequency_24h", "sender_tx_24h",
             "receiver_distinct_senders_24h", "is_pass_through", "chain_depth", "device_distinct_accounts"]
    g = _scenario_table(m, feats)
    pct = g.rank(pct=True)                      # colour = rank among scenarios per feature
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.imshow(pct, cmap=viz.SEQ, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(feats)), feats, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(g)), g.index, fontsize=9)
    ax.grid(False)
    for i in range(len(g)):
        for j in range(len(feats)):
            v = g.iloc[i, j]
            txt = f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if pct.iloc[i, j] > 0.6 else viz.INK)
    ax.set_title("Scenario fingerprints (cell = median; mean for 0/1 flags and counts; colour = rank across scenarios)")
    fig.tight_layout()
    return fig


def fig_borderline_vs_fraud(m: pd.DataFrame):
    feats = [("amount_ratio_vs_30d_avg", True), ("distance_from_home_km", True),
             ("time_since_beneficiary_added_min", True), ("log_amount", False)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (f, logx) in zip(axes, feats):
        for lab in viz.LABEL_ORDER:
            x = np.sort(m.loc[m["risk_label"] == lab, f].astype(float).to_numpy())
            if logx:
                x = np.clip(x, 0.01, None)
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=viz.LABEL_COLORS[lab], label=lab)
        if logx:
            ax.set_xscale("log")
        ax.set_title(f, fontsize=10)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("cumulative share")
    axes[-1].legend(loc="lower right")
    fig.suptitle("BORDERLINE behaves like FRAUDULENT on every key feature — the main source of model confusion",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_daily_volume(m: pd.DataFrame):
    d = pd.crosstab(m["timestamp"].dt.date, m["risk_label"]).reindex(columns=viz.LABEL_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 3.8))
    bottom = np.zeros(len(d))
    x = np.arange(len(d))
    for lab in viz.LABEL_ORDER:
        ax.bar(x, d[lab], bottom=bottom, width=0.7, color=viz.LABEL_COLORS[lab], label=lab,
               edgecolor=viz.SURFACE, linewidth=1)
        bottom += d[lab].to_numpy()
    ax.set_xticks(x[::3], [str(v)[5:] for v in d.index[::3]])
    ax.set_ylabel("transactions / day")
    ax.set_title("Daily volume by risk label (Aug 2026)")
    ax.legend(ncols=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_fanin_distribution(m10: pd.DataFrame, m11: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    bins = np.arange(0, 26)
    for ver, mm in (("v1.0", m10), ("v1.1", m11)):
        peak = mm.groupby("receiver_account_id")["receiver_distinct_senders_24h"].max() + 1   # +1 = incl. current
        h, _ = np.histogram(peak.clip(upper=25), bins=bins)
        ax.plot(bins[:-1], np.where(h > 0, h, np.nan), marker="o", markersize=4,
                color=viz.VERSION_COLORS[ver], label=ver)
    ax.set_yscale("log")
    ax.set_xlabel("peak distinct senders into one account within any 24h window")
    ax.set_ylabel("accounts (log)")
    ax.set_title("Receiver fan-in: v1.0 tops out at 5; v1.1 adds hubs with 11–15 senders/24h (rings + merchant decoys)")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_network_examples(m11: pd.DataFrame, log: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.6))
    # ring hub ego network
    ring = log[log["group_id"] == "ring_00"]["transaction_id"]
    r = m11[m11["transaction_id"].isin(ring)]
    hub = r["receiver_account_id"].mode().iat[0]
    g = nx.DiGraph()
    g.add_edges_from(zip(r["sender_account_id"], r["receiver_account_id"]))
    pos = nx.spring_layout(g, seed=config.SEED)
    nx.draw_networkx_edges(g, pos, ax=a1, edge_color=viz.BASELINE, arrows=True, arrowsize=8, width=1)
    nx.draw_networkx_nodes(g, pos, ax=a1, nodelist=[n for n in g if n != hub], node_size=60,
                           node_color=viz.SERIES[0], edgecolors=viz.SURFACE, linewidths=1.5)
    nx.draw_networkx_nodes(g, pos, ax=a1, nodelist=[hub], node_size=260, node_color=viz.SERIES[1],
                           edgecolors=viz.SURFACE, linewidths=2)
    a1.annotate(f"hub {hub}\n{g.in_degree(hub)} senders in 48h", pos[hub], xytext=(12, 12),
                textcoords="offset points", fontsize=9, color=viz.INK)
    a1.set_title("S9 fraud ring: senders → one hub"); a1.axis("off")

    # layering chain
    chain_id = log.loc[log["group_id"].str.startswith("layer_"), "group_id"].iloc[0]
    ids = log.loc[log["group_id"] == chain_id, "transaction_id"]
    c = m11[m11["transaction_id"].isin(ids)].sort_values("timestamp")
    nodes = [c["sender_account_id"].iat[0]] + c["receiver_account_id"].tolist()
    xs = np.arange(len(nodes))
    a2.plot(xs, np.zeros_like(xs), color=viz.BASELINE, linewidth=1.5, zorder=1)
    a2.scatter(xs, np.zeros_like(xs), s=180, color=viz.SERIES[1], edgecolor=viz.SURFACE, linewidth=2, zorder=2)
    t0 = c["timestamp"].iat[0]
    for k, (_, row) in enumerate(c.iterrows()):
        mins = (row["timestamp"] - t0).total_seconds() / 60
        a2.text(k + 0.5, 0.06, f"₹{row['amount_inr']:,.0f}\n+{mins:.0f} min", ha="center", fontsize=8, color=viz.INK_2)
    for k, n in enumerate(nodes):
        a2.text(k, -0.08, n.replace("ACC_IND_", "#"), ha="center", va="top", fontsize=8, color=viz.INK)
    a2.set_ylim(-0.3, 0.3); a2.set_xlim(-0.5, len(nodes) - 0.5); a2.axis("off")
    a2.set_title("S7 layering chain: value hops account to account, decaying 2–5% per hop")
    fig.tight_layout()
    return fig


def fig_camouflage(m11: pd.DataFrame, camo: set[str]):
    net = m11["scenario_id"].astype(str).isin(NETWORK_SCENARIOS) & (m11["risk_label"] == "FRAUDULENT")
    groups = {
        "Legitimate": m11["risk_label"] == "LEGITIMATE",
        "Organised fraud, not camouflaged": net & ~m11["transaction_id"].isin(camo),
        "Organised fraud, camouflaged": net & m11["transaction_id"].isin(camo),
    }
    colors = [viz.SERIES[0], viz.SERIES[1], viz.SERIES[2]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 2.2]})
    for ax, (f, title) in zip(axes[:2], [("time_since_beneficiary_added_min", "behaviour: min since beneficiary added"),
                                         ("distance_from_home_km", "behaviour: km from home")]):
        data = [m11.loc[mask, f].astype(float).clip(lower=0.1) for mask in groups.values()]
        bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": viz.INK, "linewidth": 1.5})
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(0.6); patch.set_edgecolor("none")
        for k in ("whiskers", "caps"):
            for line in bp[k]:
                line.set_color(viz.MUTED)
        ax.set_yscale("log")
        ax.set_xticks([1, 2, 3], ["Legit", "Org.\nvisible", "Org.\ncamo"], fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)

    flags = network_flags(m11)
    share = pd.DataFrame({k: flags[mask].mean() for k, mask in groups.items()})[::-1]
    ax = axes[2]
    y = np.arange(len(share))
    for k, (name, col) in enumerate(zip(groups, colors)):
        ax.barh(y + (1 - k) * 0.26, share[name], height=0.24, color=col, label=name)
    row = share.index.get_loc("any network flag")
    for k, name in enumerate(groups):
        v = share[name].iloc[row]
        ax.text(v + 0.01, y[row] + (1 - k) * 0.26, f"{v:.0%}", va="center", fontsize=8, color=viz.INK_2)
    ax.set_xlim(0, 0.85)
    ax.set_yticks(y, share.index, fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("network: share of transactions with each red flag", fontsize=10)
    ax.grid(axis="y", visible=False)
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Camouflaged organised fraud looks legitimate on behaviour, but 7 in 10 still carry a network red flag",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def fig_network_signal_by_scenario(m11: pd.DataFrame):
    feats = ["sender_tx_10m", "sender_tx_24h", "receiver_distinct_senders_24h", "is_pass_through",
             "chain_depth", "forward_ratio", "device_distinct_accounts", "ip_distinct_accounts", "same_community"]
    g = _scenario_table(m11, feats)
    legit = g.loc[[i for i in g.index if i.startswith("S1 ")]].iloc[0]
    ratio = (g + 0.02) / (legit + 0.02)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(np.log2(ratio.clip(lower=1 / 16, upper=16)), cmap=viz.DIV, vmin=-4, vmax=4, aspect="auto")
    ax.set_xticks(range(len(feats)), feats, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(g)), g.index, fontsize=9)
    ax.grid(False)
    for i in range(len(g)):
        for j in range(len(feats)):
            ax.text(j, i, f"{g.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(np.log2(ratio.iloc[i, j])) > 2.5 else viz.INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[-4, -2, 0, 2, 4])
    cb.ax.set_yticklabels(["÷16", "÷4", "= legit", "×4", "×16"]); cb.outline.set_visible(False)
    ax.set_title("v1.1 network features by scenario (cell = mean, median for forward_ratio; colour = ratio to S1 normal)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- orchestration

def build_figures(m10: pd.DataFrame, m11: pd.DataFrame, log: pd.DataFrame, camo: set[str]) -> dict:
    return {
        "eda_01_label_overview": lambda: fig_label_overview(m11),
        "eda_02_amount_by_label": lambda: fig_amount_by_label(m11),
        "eda_03_fraud_rate_time_channel": lambda: fig_fraud_rate_time_channel(m11),
        "eda_04_geo": lambda: fig_geo(m11),
        "eda_05_correlation": lambda: fig_correlation(m11),
        "eda_06_feature_auc_v10_v11": lambda: fig_feature_auc(m10, m11),
        "eda_07_behaviour_flags": lambda: fig_behaviour_flags(m11),
        "eda_08_scenario_profile": lambda: fig_scenario_profile(m11),
        "eda_09_borderline_vs_fraud": lambda: fig_borderline_vs_fraud(m11),
        "eda_10_daily_volume": lambda: fig_daily_volume(m11),
        "eda_11_fanin_distribution": lambda: fig_fanin_distribution(m10, m11),
        "eda_12_network_examples": lambda: fig_network_examples(m11, log),
        "eda_13_camouflage": lambda: fig_camouflage(m11, camo),
        "eda_14_network_signal_by_scenario": lambda: fig_network_signal_by_scenario(m11),
    }


def summary(m10: pd.DataFrame, m11: pd.DataFrame) -> dict:
    rank = feature_ranking(m11)
    return {
        "rows": len(m11),
        "label_counts": m11["risk_label"].value_counts().astype(int).to_dict(),
        "fraud_rate": round(float(m11["is_fraud"].mean()), 4),
        "fraud_rate_by_channel": m11.groupby("channel", observed=True)["is_fraud"].mean().round(4).to_dict(),
        "fraud_rate_by_hour_range": [round(float(v), 4) for v in
                                     m11.groupby("hour_of_day")["is_fraud"].mean().agg(["min", "max"])],
        "flags": flag_table(m11).to_dict(orient="records"),
        "rq1_feature_ranking_v1_1": rank.reset_index(names="feature").to_dict(orient="records"),
        "rq1_top10": rank.head(10).index.tolist(),
        "network_auc_max": {
            "v1.0": round(float(single_feature_auc(m10, NETWORK_NUMERIC).max()), 4),
            "v1.1": round(float(single_feature_auc(m11, NETWORK_NUMERIC).max()), 4),
        },
        "mule_suspect_fraud_rate_v1_0": round(float(
            m10.loc[m10["sender_account_type"] == "MULE_SUSPECT", "is_fraud"].mean()), 4),
    }


def run_all(out_dir: Path | None = None) -> dict:
    viz.apply_style()
    m10, m11 = load_master("v1_0"), load_master("v1_1")
    log = pd.read_csv(config.INDIA_V1_1_DIR / "injection_log.csv")
    figs = build_figures(m10, m11, log, camouflaged_ids())
    paths = {}
    for name, make in figs.items():
        fig = make()
        paths[name] = str(viz.save(fig, name, out_dir))
        plt.close(fig)
    s = summary(m10, m11)
    s["figures"] = paths
    (config.REPORTS_DIR / "eda_summary.json").write_text(json.dumps(s, indent=2, default=str))
    return s
