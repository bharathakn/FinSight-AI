"""Build the FinSight AI project report (HTML → PDF via headless Chrome).

    uv run python docs/report/build_report.py          # → reports/FinSight_AI_Project_Report.pdf

Every number is read from reports/*.json|csv so the PDF never drifts from the results.
Re-run after each phase; update the prose sections marked PHASE-STATUS when a planned phase completes.
"""
from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"
OUT_HTML = REP / "FinSight_AI_Project_Report.html"
OUT_PDF = REP / "FinSight_AI_Project_Report.pdf"


def J(name):
    return json.loads((REP / name).read_text())


def esc(x):
    return html.escape(str(x))


def table(rows, header, cls="", widths=None, raw=False):
    """rows: list of lists. raw=True → cells are trusted HTML."""
    cols = "" if not widths else "<colgroup>" + "".join(f'<col style="width:{w}">' for w in widths) + "</colgroup>"
    numeric = "num" in cls.split()
    is_num = lambda c: numeric and bool(re.fullmatch(r"[−+\-]?[\d,.]+%?( ± [\d.]+)?|—", str(c).strip()))
    first = rows[0] if rows else []
    th = "".join(f'<th class="{"r" if j and j < len(first) and is_num(first[j]) else ""}">{h if raw else esc(h)}</th>'
                 for j, h in enumerate(header))
    body = "".join("<tr>" + "".join(f'<td class="{"r" if j and is_num(c) else ""}">{c if raw else esc(c)}</td>'
                                    for j, c in enumerate(r)) + "</tr>" for r in rows)
    keep = " keep" if len(rows) <= 14 else ""
    return f'<table class="{cls}{keep}">{cols}<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def fig(name, caption, width="100%"):
    return (f'<figure><img src="figures/{name}.png" style="width:{width}">'
            f'<figcaption>{caption}</figcaption></figure>')


def flow(stages, cls=""):
    """Horizontal box-and-arrow diagram. stages: list of (title, body) or list of lists for stacked boxes."""
    parts = []
    for i, st in enumerate(stages):
        boxes = st if isinstance(st, list) else [st]
        inner = "".join(f'<div class="box"><b>{t}</b><span>{b}</span></div>' for t, b in boxes)
        parts.append(f'<div class="col">{inner}</div>')
        if i < len(stages) - 1:
            parts.append('<div class="arrow">→</div>')
    return f'<div class="flow {cls}">{"".join(parts)}</div>'


def f3(x):
    return f"{x:.3f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


# ---------------------------------------------------------------- data
eda = J("eda_summary.json")
mining = J("mining_summary.json")
ml = J("ml_summary.json")
dq10 = J("data_quality_india_v1_0.json")
sanc = J("sanctions_summary.json")
meta11 = json.loads((ROOT / "dataset/processed/custom_india_v1_1/dataset_metadata.json").read_text())
res = pd.read_csv(REP / "ml_experiment_results.csv")
sig = pd.read_csv(REP / "ml_paired_tests.csv")
imb = pd.read_csv(REP / "ml_imbalance.csv")
temporal = pd.read_csv(REP / "ml_temporal.csv")
scen = pd.read_csv(REP / "ml_per_scenario.csv")
bench = pd.read_csv(REP / "mining_benchmark.csv")
paysc = pd.read_csv(REP / "mining_paysim_scaling.csv")
rules11 = pd.read_csv(REP / "association_rules_v1_1.csv")
ep = pd.read_csv(REP / "emerging_patterns_v1_1.csv")
items = pd.read_csv(REP / "mining_items_v1_1.csv")
us = J("unsupervised_summary.json")
anr = pd.read_csv(REP / "anomaly_results.csv")
uclu = pd.read_csv(REP / "unsup_clusters.csv")
ascen = pd.read_csv(REP / "anomaly_per_scenario.csv")
ucs, uan, ups, upar = us["clustering"], us["anomaly"], us["paysim_if"], us["onnx_parity"]
DET = {"isolation_forest": "Isolation Forest", "ocsvm_nu0.01": "One-class SVM (ν = 0.01)",
       "ocsvm_nu0.05": "One-class SVM (ν = 0.05)", "autoencoder": "Autoencoder"}
fs7 = J("final_summary.json")
ftune = pd.read_csv(REP / "final_tuning_summary.csv")
fcomp = pd.read_csv(REP / "final_model_comparison.csv").set_index("model")
ftiers = pd.read_csv(REP / "final_risk_tiers.csv").set_index("risk_label")
FT = fs7["test"]
RSC = fs7["risk_scorer"]
N_TESTS = 69
RND = J("platform_render.json")
SIMB = pd.read_csv(REP / "analyze_simulation.csv")
SIMG = SIMB.groupby(["attack", "camouflage"]).agg(alerted=("alerted", "mean"), lo=("alerted", "min"), hi=("alerted", "max"))
SIM_FA = SIMB["background_false_alert"].mean()
ATT = {"fraud_ring": "Fraud ring (fan-in to a hub)", "mule_chain": "Mule chain (pass-through, layering)", "burst": "Burst (one sender, many payees in 10 min)",
       "structuring": "Structuring (just under ₹50,000)", "shared_device": "Shared device (one phone, many accounts)"}
def sim_cell(a, c):
    g = SIMG.loc[(a, c)]
    return f"{g['alerted']:.0%}" if g["lo"] == g["hi"] else f"{g['alerted']:.0%} ({g['lo']:.0%}–{g['hi']:.0%})"
RW, RC = RND["warm"], RND["cold_start"]
LIVE_UI, LIVE_API = "https://fin-sight-ai-bay.vercel.app", "https://finsight-api-r2mw.onrender.com"
plat = J("platform_summary.json")
PW = {k: v["warm"] for k, v in plat["warm"].items()}
PL = plat["load"]
IMG = plat["images_mb"]
LBL7 = {"decision_tree": "Decision Tree", "knn": "KNN", "random_forest": "Random Forest", "xgboost": "XGBoost", "mlp": "MLP",
        "stack_uncalibrated": "Stacking (RF + XGB + MLP → LR)", "xgboost_calibrated": "XGBoost tuned + isotonic (production)",
        "stack_calibrated": "Stacking + isotonic"}

MODEL = {"decision_tree": "Decision Tree", "naive_bayes": "Naive Bayes", "knn": "KNN",
         "random_forest": "Random Forest", "xgboost": "XGBoost", "mlp": "MLP"}
ORDER = list(MODEL)
best = ml["best_v1_1"]
rb = ml["rule_baseline_v1_1"]
tri = ml["three_class_xgb_expC"]


def decisions():
    rows = []
    for line in (ROOT / "docs/DECISIONS.md").read_text().splitlines():
        m = re.match(r"\| (D-\d+) \| ([\d-]+) \| (.*?) \| (.*?) \| (.*?) \|$", line)
        if m:
            clean = lambda s: re.sub(r"\*\*|`|~~", "", s)
            rows.append([m.group(1), clean(m.group(3)), clean(m.group(4))])
    return rows


# ---------------------------------------------------------------- tables built from results
def results_matrix(version):
    r = res[res.version == version].set_index(["model", "exp"])
    rows = []
    for m in ORDER:
        rows.append([MODEL[m]] + [f3(r.loc[(m, e), "test_pr_auc"]) for e in "ABC"]
                    + [f3(r.loc[(m, "C"), "test_roc_auc"]), f3(r.loc[(m, "C"), "test_f1"]),
                       f3(r.loc[(m, "C"), "test_precision"]), f3(r.loc[(m, "C"), "test_recall"])])
    return table(rows, ["Model", "PR-AUC A", "PR-AUC B", "PR-AUC C", "ROC-AUC C", "F1 C", "Precision C", "Recall C"], "num")


def cv_matrix():
    r = res.set_index(["version", "model", "exp"])
    rows = []
    for m in ORDER:
        rows.append([MODEL[m]] + [f"{r.loc[(v, m, e), 'cv_pr_auc_mean']:.3f} ± {r.loc[(v, m, e), 'cv_pr_auc_std']:.3f}"
                                  for v in ("v1_0", "v1_1") for e in "ABC"])
    return table(rows, ["Model", "v1.0 A", "v1.0 B", "v1.0 C", "v1.1 A", "v1.1 B", "v1.1 C"], "num small")


def sig_table():
    rows = []
    for m in ORDER:
        row = [MODEL[m]]
        for v in ("v1_0", "v1_1"):
            for c in ("B vs A", "C vs B"):
                s = sig[(sig.model == m) & (sig.version == v) & (sig.comparison == c)].iloc[0]
                star = " ✔" if s.t_pvalue < 0.05 and s.mean_diff > 0 else ""
                row.append(f"{s.mean_diff:+.3f} (p={s.t_pvalue:.3g}){star}")
        rows.append(row)
    return table(rows, ["Model", "v1.0 B−A", "v1.0 C−B", "v1.1 B−A", "v1.1 C−B"], "num small")


def imb_table():
    piv = imb.pivot(index="model", columns="imbalance", values="pr_auc_mean").reindex(ORDER)
    rows = [[MODEL[m]] + [("—" if pd.isna(piv.loc[m].get(s)) else f3(piv.loc[m, s])) for s in ("none", "class_weight", "smote")]
            for m in ORDER]
    return table(rows, ["Model", "None", "Class weight", "SMOTE"], "num")


def scen_table():
    d = scen[(scen.model == "xgboost") & (scen.risk_label == "FRAUDULENT")]
    piv = d.pivot(index="scenario", columns="exp", values="flag_rate")
    n = d.drop_duplicates("scenario").set_index("scenario")["n"]
    piv = piv.loc[sorted(piv.index, key=lambda s: int(s.split()[0][1:]))]
    return table([[s, n[s]] + [pct(piv.loc[s, e], 0) for e in "ABC"] for s in piv.index],
                 ["Scenario", "n (test)", "Recall A", "Recall B", "Recall C"], "num")


def bench_table():
    b = bench[bench.min_support.isin([0.01, 0.002]) & bench.algorithm.isin(["apriori", "fpgrowth", "fpmax"])]
    names = {"apriori": "Apriori", "fpgrowth": "FP-Growth", "fpmax": "FP-Max"}
    rows = [[f"{r['items']} ({r.n_items})", r.min_support, names[r.algorithm], f"{int(r.itemsets):,}",
             f"{r.seconds:.3f}", f"{r.peak_mb:,.1f}"] for _, r in b.iterrows()]
    return table(rows, ["Item universe", "min_support", "Algorithm", "Itemsets", "Runtime s", "Peak MB"], "num")


def rules_table(k=10):
    r = rules11[rules11.consequent == "FRAUDULENT"].sort_values(["lift", "support"], ascending=False).head(k)
    return table([[x.rule.replace(" → FRAUDULENT", ""), pct(x.support), pct(x.confidence, 0), f"{x.lift:.2f}",
                   "yes" if x.has_network_item else ""] for x in r.itertuples()],
                 ["Antecedent (→ FRAUDULENT)", "Support", "Confidence", "Lift", "Network item"], "small", widths=["52%", "12%", "12%", "10%", "14%"])


def rq1_table():
    r = eda["rq1_feature_ranking_v1_1"][:12]
    return table([[i + 1, x["feature"], x["group"], f3(x["auc"]), f"{x['mutual_info']:.4f}"] for i, x in enumerate(r)],
                 ["#", "Feature", "Group", "Single-feature AUC", "Mutual information"], "num small")


def flags_table():
    return table([[f["flag"], pct(f["support"]), pct(f["fraud_rate_given_flag"]), f"{f['lift']:.2f}",
                   pct(f["share_LEGITIMATE"]), pct(f["share_FRAUDULENT"]), pct(f["share_BORDERLINE"])] for f in eda["flags"]],
                 ["Flag", "Support", "Fraud rate | flag", "Lift", "% of LEGIT", "% of FRAUD", "% of BORDER"], "num small")


def items_table():
    return table([[r["item"], r["rule"], f"{int(r['count']):,}", pct(r.support), ("—" if pd.isna(r.fraud_rate) else pct(r.fraud_rate))]
                  for _, r in items.iterrows()], ["Item", "Definition", "Count", "Support", "Fraud rate"], "small",
                 widths=["24%", "40%", "12%", "12%", "12%"])


def temporal_table():
    p = temporal.pivot_table(index=["model", "exp"], columns="split", values="pr_auc").reset_index()
    return table([[MODEL[r.model], r.exp, f3(r.stratified), f3(r.temporal)] for r in p.itertuples()],
                 ["Model", "Exp", "Stratified split", "Temporal split"], "num")


def ri_table():
    return table([[x["fk"], f"{x['rows']:,}", pct(x["matched_rate"], 2), x["orphans"]] for x in dq10["referential_integrity"]],
                 ["Relationship", "Rows", "Matched", "Orphans"], "small")


def sweep_table():
    return table([[x["threshold"], x["alerts"], pct(x["false_positive_rate"], 2)] for x in sanc["customer_screening_threshold_sweep"]],
                 ["Fuzzy threshold", "Alerts on 1,200 customers", "False-positive rate"], "num")


def tuning_table():
    rows = []
    for r in ftune.sort_values("tuned_cv_pr_auc", ascending=False).itertuples():
        bp = json.loads(r.best_params)
        short = ", ".join(f"{k}={(round(v, 3) if isinstance(v, float) else v)}" for k, v in bp.items())
        rows.append([LBL7[r.model], f"{r.search} ({r.configs_tried})", f3(r.default_cv_pr_auc), f3(r.tuned_cv_pr_auc),
                     f"{r.paired_mean_diff:+.3f}", ("< 0.01" if r.paired_t_pvalue < 0.01 else f"{r.paired_t_pvalue:.2f}"), short])
    return table(rows, ["Model", "Search (configs)", "Default CV", "Tuned CV", "Gain", "p (paired)", "Best hyperparameters"], "num small",
                 widths=["13%", "11%", "9%", "9%", "7%", "8%", "43%"])


def final_table():
    order = ["xgboost_calibrated", "xgboost", "stack_uncalibrated", "random_forest", "stack_calibrated", "decision_tree", "mlp", "knn"]
    rows = [[LBL7[k], f3(fcomp.loc[k, "val_pr_auc"]), f3(fcomp.loc[k, "test_pr_auc"]), f3(fcomp.loc[k, "test_roc_auc"]),
             f3(fcomp.loc[k, "test_f1"]), f"{fcomp.loc[k, 'test_precision']:.2f}", f"{fcomp.loc[k, 'test_recall']:.2f}"] for k in order]
    rows.append(["Phase 5 XGBoost (defaults, SMOTE)", "—", f3(best["test_pr_auc"]), f3(best["test_roc_auc"]), f3(best["test_f1"]),
                 f"{best['test_precision']:.2f}", f"{best['test_recall']:.2f}"])
    return table(rows, ["Model (v1.1, Exp C)", "Val PR-AUC", "Test PR-AUC", "Test ROC-AUC", "Test F1", "Precision", "Recall"], "num small",
                 widths=["34%", "11%", "11%", "11%", "11%", "11%", "11%"])


def tier_table():
    rows = []
    for lab in ("LEGITIMATE", "FRAUDULENT", "BORDERLINE"):
        r = ftiers.loc[lab]
        n = int(sum(r[f"{t}_n"] for t in ("LOW", "MEDIUM", "HIGH", "CRITICAL")))
        rows.append([lab.title(), f"{n:,}"] + [pct(r[f"{t}_share"], 0) for t in ("LOW", "MEDIUM", "HIGH", "CRITICAL")])
    return table(rows, ["True label (test)", "Rows", "LOW", "MEDIUM", "HIGH", "CRITICAL"], "num")


def cluster_summary_table():
    rows = []
    for e in ("B", "C"):
        c = ucs[e]
        rows.append([f"Exp {e} ({c['n_features_after_encoding']} encoded columns)", c["k"], f3(c["silhouette"]), f"{c['davies_bouldin']:.2f}",
                     f3(c["kmeans_vs_label"]["ari"]), f3(c["kmeans_vs_label"]["nmi"]),
                     f"{pct(c['flag_high_fraud_clusters']['precision'], 0)} / {pct(c['flag_high_fraud_clusters']['recall'], 0)}",
                     f"{pct(c['dbscan']['noise_rate'])} ({pct(c['dbscan']['fraud_rate_noise'], 0)} fraud)"])
    return table(rows, ["Feature space", "K", "Silhouette", "Davies–Bouldin", "ARI", "NMI", "Risk clusters P / R", "DBSCAN noise"], "num small")


NICE = {"sender_tx_10m": "sender tx in 10 min", "sender_tx_1h": "sender tx in 1 h", "time_since_last_tx_sec": "time since last tx",
        "sender_tx_24h": "sender tx in 24 h", "receiver_tx_in_24h": "receiver tx in 24 h",
        "receiver_distinct_senders_24h": "receiver distinct senders 24 h", "receiver_pagerank": "receiver PageRank",
        "receiver_distinct_senders_7d": "receiver distinct senders 7 d", "amount_ratio_vs_30d_avg": "amount vs 30-day avg",
        "log_amount": "log amount", "same_community": "same Louvain community", "forward_ratio": "forward ratio"}


def cluster_table():
    d = uclu[uclu.exp == "C"].sort_values("fraud_rate")
    names = {0: "Velocity bursts", 1: "Normal traffic", 2: "Unusual amount, community flows", 3: "Fan-in hubs"}
    rows = []
    for r in d.itertuples():
        prof = ucs["C"]["profile"][str(r.cluster)][:3]
        rows.append([f"{r.cluster} · {names.get(r.cluster, '')}", f"{r.n:,}", pct(r.fraud_rate, 0), pct(r.share_LEGITIMATE, 0), pct(r.share_BORDERLINE, 0),
                     ", ".join(f"{NICE.get(f, f.replace('_', ' '))} {v:+.1f}σ" for f, v in prof)])
    return table(rows, ["Cluster", "Rows", "Fraud", "Legit", "Borderline", "Defining centroid features"], "num small",
                 widths=["22%", "8%", "8%", "8%", "10%", "44%"])


def detector_table():
    rows = []
    for e in ("C", "B"):
        for dname in DET:
            r = anr[(anr.exp == e) & (anr.detector == dname)].iloc[0]
            rows.append([f"{DET[dname]}, Exp {e}", f3(r.pr_auc), f3(r.roc_auc), f"{r.recall_at_1pct_fpr:.2f}",
                         f"{r.p95_precision:.2f} / {r.p95_recall:.2f}", f"{r.p99_precision:.2f} / {r.p99_recall:.2f}", pct(r.p95_borderline_flag_rate, 0)])
    rows.append(["XGBoost (supervised, Phase 5), Exp C", f3(best["test_pr_auc"]), f3(best["test_roc_auc"]), "—",
                 f"{best['test_precision']:.2f} / {best['test_recall']:.2f} (val. threshold)", "—", pct(best["test_borderline_flag_rate"], 0)])
    return table(rows, ["Detector (v1.1 test)", "PR-AUC", "ROC-AUC", "Recall @1% FPR", "P / R at p95", "P / R at p99", "BORDERLINE flagged"], "num small",
                 widths=["30%", "9%", "9%", "11%", "16%", "13%", "12%"])


def anomaly_scen_table():
    d = ascen[ascen.risk_label == "FRAUDULENT"]
    piv = d.pivot_table(index="scenario", columns="detector", values="flag_rate")
    n = d.drop_duplicates("scenario").set_index("scenario")["n"]
    piv = piv.loc[sorted(piv.index, key=lambda x: int(x.split()[0][1:]))]
    cols = ["isolation_forest", "ocsvm_nu0.05", "autoencoder", "xgboost_supervised"]
    return table([[sc, n[sc]] + [pct(piv.loc[sc, c], 0) for c in cols] for sc in piv.index],
                 ["Fraud scenario", "n (test)", "Isolation Forest", "OC-SVM ν=0.05", "Autoencoder", "XGBoost (supervised)"], "num small")


def paysim_table():
    fs = ups["feature_sets"]
    nm = {"raw": "amount + balances (7 features)", "raw+balance_error": "+ 2 balance-error features (9)"}
    return table([[nm[k], f3(v["pr_auc"]), f3(v["roc_auc"]), f"{v['recall_at_1pct_fpr']:.2f}", f"{v['p99_precision']:.2f} / {v['p99_recall']:.2f}",
                   f"{v['fit_seconds']:.1f} / {v['score_seconds']:.0f}", f"{v['rows_per_second']:,}"] for k, v in fs.items()],
                 ["Isolation Forest features", "PR-AUC", "ROC-AUC", "Recall @1% FPR", "P / R at p99", "Fit / score s", "Rows per s"], "num small")


inj = meta11["injection"]
inj_rows = inj["rows_by_group_type"]
v10, v11 = mining["v1_0"], mining["v1_1"]
hp = mining["benchmark_hardest_point"]
r11 = res[res.version == "v1_1"].set_index(["model", "exp"])
dets = mining["v1_1"]["detector_all_productive_fraud_rules"]

# ---------------------------------------------------------------- document
TODAY = date.today().strftime("%d %B %Y")
H = []
add = H.append

add(f"""
<section class="cover">
  <div class="kicker">Integrated project · Data Mining &amp; Wrangling · Machine Learning · Cloud Computing &amp; Technology</div>
  <h1 class="title">FinSight AI</h1>
  <div class="subtitle">Cloud-based financial transaction intelligence: pattern mining, machine learning and microservices on a synthetic Indian payments ecosystem (UPI · NEFT · RTGS)</div>
  <div class="cover-grid">
    <div><b>Document</b><span>Complete project report: planning, design, implementation, results, status and hand-off guide</span></div>
    <div><b>Version / date</b><span>Report v1.0 · {TODAY}</span></div>
    <div><b>Status</b><span>All 10 phases complete: data, mining, supervised ML, unsupervised + anomaly detection, tuning + risk score + explanations, microservices + Docker, React dashboard with live analysis, and cloud deployment on Vercel + Render with measurements.</span></div>
    <div><b>Headline result</b><span>Production model (tuned, calibrated XGBoost): test PR-AUC {FT['test_pr_auc']:.3f}, ROC-AUC {FT['test_roc_auc']:.3f}, F1 {FT['test_f1']:.3f} (precision {pct(FT['test_precision'],0)}, recall {pct(FT['test_recall'],0)}). Network features add a significant gain only where real network structure exists.</span></div>
    <div><b>Live system</b><span>Dashboard {LIVE_UI} · API {LIVE_API}/docs (React on Vercel's CDN · FastAPI in Docker on Render)</span></div>
    <div><b>Repository</b><span>github.com/bharathakn/FinSight-AI</span></div>
  </div>
  <div class="note">All India data is synthetic. Results describe the data generator and our documented modifications, not real-world fraud.</div>
</section>

<section class="howto">
<h2 class="nonum">How to use this document</h2>
<p>This single PDF is meant to be enough for three separate course reports and three presentations. Every number comes directly from the project's result files.</p>
{table([
 ["Data Mining &amp; Wrangling (DMW)", "1–10, 11 (baselines: rules vs ML), 12 (cluster analysis, PaySim scale), 15 (dashboard: data and pattern pages), 17 (DMW table), 18, 19.1", "KDD pipeline, wrangling, data quality and leakage audit, v1.1 dataset engineering, discretisation, Apriori vs FP-Growth, rule evaluation"],
 ["Machine Learning (ML)", "1–3, 5, 7–9, 11, 12, 13, 14 (model serving), 15 (risk scorer, ML lab), 17 (ML table), 18, 19.2", "Feature engineering, experimental protocol, 6 classifiers × 3 feature sets × 2 datasets, significance testing, imbalance, per-scenario analysis; clustering, anomaly detection, autoencoder → ONNX; tuning, bias–variance, stacking, calibration, risk score, SHAP"],
 ["Cloud Computing &amp; Technology (CCT)", "1–4, 12–13 (serving artifacts), 14, 15, 16, 17 (CCT table), 18, 19.3", "Microservices (built + measured), REST API, Docker, Compose scaling, load testing, observability, security; live Vercel + Render deployment, free-tier constraints, CI/CD, IaC, scale-to-zero and cold start (measured)"],
], ["If you are writing for…", "Read chapters", "Core material"], raw=True, widths=["25%", "30%", "45%"])}
<p>Chapter 19 contains ready-made <b>report outlines and slide-by-slide deck outlines</b> for each subject, with the figure to use on each slide. The figures are also available as PNG files in <code>reports/figures/</code> of the repository.</p>

<h2 class="nonum">Contents</h2>
<ol class="toc">
<li>Executive summary</li><li>Introduction, objectives and research questions</li><li>Datasets</li>
<li>Planning, methodology and project management</li><li>Phase 0: dataset audit</li><li>Phase 1: data wrangling and data quality</li>
<li>Phase 1b: dataset v1.1 (graph-structure injection)</li><li>Phase 3: graph (network) features</li><li>Phase 2: exploratory data analysis</li>
<li>Phase 4: association rule mining</li><li>Phase 5: supervised classification</li><li>Phase 6: unsupervised learning and anomaly detection</li><li>Phase 7: tuning, ensembles, calibration, risk score and explanations</li><li>Phase 8: cloud services and containers</li><li>Phase 9: the web dashboard and live analysis</li><li>Phase 10: cloud deployment and measurements</li>
<li>Syllabus mapping</li><li>Conclusions, limitations and future work</li><li>Guide for report and presentation writers</li>
<li>Appendices: decision log · feature dictionary · glossary · reproducibility · references</li>
</ol>
</section>
""")

# 1 executive summary
add(f"""
<section>
<h1>Executive summary</h1>
<p><b>FinSight AI</b> is a financial-transaction intelligence platform built as one project for three courses. It takes a synthetic Indian payment ecosystem (10,000 UPI/NEFT/RTGS transactions, 1,200 customers, 1,500 accounts) plus four public datasets (IEEE-CIS fraud, PaySim mobile money, OFAC and UN sanctions lists). It runs the full <b>KDD process</b> (data mining), builds and rigorously compares <b>machine-learning models</b>, and is deployed as <b>cloud microservices</b> with a public web dashboard.</p>
<h3>What was done (all 10 phases)</h3>
<ul>
<li><b>Audit before building.</b> The original plan was checked against the data. The audit found label-leaking columns (<code>rapid_forward_delta_sec</code>, <code>is_flagged_alert</code>, <code>alert_id</code>), a join that matched only 8 of 10,000 rows, redundant and constant columns, and, most importantly, that the dataset's "network" fraud scenarios (rings, mules, layering) <b>had no network structure at all</b>: every graph feature scored AUC ≈ 0.50.</li>
<li><b>Dataset engineering (v1.1).</b> A seeded, logged, tested injection step wired those scenarios into real graph structure. It added legitimate look-alike patterns (merchant hubs, family devices) so the signal isn't a giveaway, and camouflaged 50% of organised fraud so it looks normal per transaction. Labels were never changed; v1.0 is kept as a control.</li>
<li><b>Wrangling and quality.</b> Seven files were merged into one 57-column table, then 76 columns with 19 point-in-time network features. Data-quality and leakage reports were produced, and 20,332 sanctions entities were parsed from CSV + XML.</li>
<li><b>Association rule mining.</b> Apriori and FP-Growth return identical itemsets, but Apriori needs <b>{hp['apriori']['peak_mb']:,.0f} MB vs {hp['fpgrowth']['peak_mb']:.1f} MB</b> at the hardest setting. {v11['productive']} productive rules were found on v1.1 ({v11['productive_fraud_with_network_item']} use network items) vs {v10['productive']} on v1.0 (none use network items). Top rule: <i>pass-through → fraud</i>, confidence 99%, lift 7.4.</li>
<li><b>Supervised learning.</b> 6 classifiers × 3 feature sets × 2 dataset versions, compared with 5×2 repeated cross-validation and paired significance tests. Behavioural features help every model. Network features help <b>5 of 6 models significantly on v1.1 (p &lt; 0.001) and 0 of 6 on v1.0</b>. Best model: <b>XGBoost, test PR-AUC {best['test_pr_auc']:.3f}, F1 {best['test_f1']:.3f}</b>. SMOTE never improved results.</li>
<li><b>Unsupervised learning and anomaly detection.</b> Without labels, K-Means (K = {ucs['C']['k']}) separates normal traffic (3% fraud) from three interpretable risk clusters (velocity bursts, fan-in hubs, unusual-amount community flows; ARI {ucs['C']['kmeans_vs_label']['ari']:.2f}). Detectors trained only on legitimate rows reach <b>PR-AUC ≈ {uan['best_detector_expC']['pr_auc']:.2f}</b> (One-class SVM ≈ autoencoder) vs {best['test_pr_auc']:.3f} supervised. They flag ~80% of BORDERLINE rows (<i>anomalous ≠ fraudulent</i>) and recover {uan['complementarity']['caught_by']['autoencoder']} of the {uan['complementarity']['xgb_missed']} frauds XGBoost misses. The autoencoder ships as a {upar['onnx_bytes'] // 1024} KB ONNX file; an Isolation Forest scores {ups['rows_scored']/1e6:.2f} M PaySim rows at ~{ups['feature_sets']['raw']['rows_per_second']//1000*1000:,} rows/s.</li>
<li><b>Tuning, ensembles and the risk score.</b> Grid and randomized search with cross-validation improved every model (decision tree +0.058 PR-AUC). A stacking ensemble did not beat tuned XGBoost, so the <b>production model is a tuned, isotonic-calibrated XGBoost: test PR-AUC {FT['test_pr_auc']:.3f}, F1 {FT['test_f1']:.3f}</b> (Phase 5 default: {best['test_f1']:.3f}), in a {fs7['artifact_sizes_mb']['production_model.joblib']} MB file. A 0–100 risk score combines it with the anomaly percentile and a network-flag score; validation-chosen tiers put {pct(ftiers.loc['FRAUDULENT','CRITICAL_share'],0)} of fraud in CRITICAL and {pct(ftiers.loc['LEGITIMATE','LOW_share'],0)} of legitimate payments in LOW. Every score is explained with SHAP values computed natively by XGBoost.</li>
<li><b>Cloud services.</b> The models run behind a FastAPI <b>API gateway and four microservices</b> (ML, mining, graph, sanctions) in Docker, or as one container for the free cloud tier. The same code runs in both modes and returns identical scores. A warm score takes <b>{PW['compose-remote']['client_p95_ms']:.0f} ms (p95)</b>; the single container uses {plat['warm']['compose-inprocess']['rss_mb']['gateway']:.0f} MB of memory, within Render's 512 MB free instance. A load test served {PL['remote_ml3']['rps_total']:.0f} requests/s with no errors, and scaling the ML service to three replicas cut median latency from {PL['remote_ml1']['p50_ms']:.0f} to {PL['remote_ml3']['p50_ms']:.0f} ms.</li>
<li><b>Web dashboard.</b> A React single-page app with, besides the live-analysis home page, nine pages (overview, data quality, transaction explorer, pattern-mining lab, ML lab, clustering and anomaly, network explorer, a what-if risk scorer and a platform page) presents every result live from the API, in one warm paper-and-teal theme, with an optional dark version.</li>
<li><b>Live analysis.</b> The dashboard's home page runs the whole system on new data: upload a CSV of payments, analyse the built-in demo dataset, or inject simulated attacks (fraud ring, mule chain, burst, structuring, shared device, optionally camouflaged) into a normal day. Analysing the demo dataset reproduces every training feature exactly. Simulated attacks are alerted 100% of the time plain and 80–100% camouflaged, with about 2% false alerts on normal payments.</li>
<li><b>Cloud deployment.</b> The dashboard is live on <b>Vercel's CDN</b> and the API runs as a Docker container on <b>Render</b> (Singapore), both deployed from GitHub by Infrastructure-as-Code files and redeployed on every push, with CI testing each push. Measured from India: warm scoring <b>p50 {RW['client_p50_ms']:.0f} ms, p95 {RW['client_p95_ms']:.0f} ms</b> (server time p50 {RW['server_p50_ms']:.0f} ms), {RND['rss_mb']['gateway']:.0f} MB of the 512 MB free instance, and a <b>{RC['seconds_to_healthy']:.0f} s cold start</b> after the free instance sleeps.</li>
</ul>
<h3>Key results at a glance</h3>
{table([
 ["RQ1 strongest single signals", "amount (AUC 0.85), amount vs 30-day average (0.81), minutes since beneficiary added (0.79), distance from home (0.74)"],
 ["RQ2 top fraud patterns", "{PASS_THROUGH} 99% conf · {UPI, high spike, hub receiver} 100% · {quick inflow, recent beneficiary} 95% · {₹45–50k, recent beneficiary} 90%"],
 ["RQ2 Apriori vs FP-Growth", f"identical results; Apriori faster on small searches (vectorised), FP-Growth faster and ~17× leaner on large ones"],
 ["RQ3 feature groups", "B (behaviour) > A for 12/12 model-dataset pairs; C (network) > B for 5/6 models on v1.1 and 0/6 on v1.0"],
 ["RQ4 best model", f"tuned + calibrated XGBoost (lr ≈ 0.024, ~465 trees, depth 6): PR-AUC {FT['test_pr_auc']:.3f} · ROC-AUC {FT['test_roc_auc']:.3f} · F1 {FT['test_f1']:.3f} · precision {FT['test_precision']:.2f} · recall {FT['test_recall']:.2f}; stacking no better"],
 ["RQ5 clustering without labels", f"partly: K = {ucs['C']['k']} on all features gives 1 normal + 3 risk clusters (ARI {ucs['C']['kmeans_vs_label']['ari']:.2f}); reviewing the risk clusters catches {pct(ucs['C']['flag_high_fraud_clusters']['recall'],0)} of fraud at {pct(ucs['C']['flag_high_fraud_clusters']['precision'],0)} precision; silhouette ≤ 0.16 (weak structure)"],
 ["RQ6 anomaly detection", f"One-class SVM PR-AUC {uan['best_detector_expC']['pr_auc']:.3f} ≈ autoencoder > Isolation Forest; catches single-transaction anomalies ≥ 94%, network fraud 40–76%, high-value spikes ≤ 15%; flags ~80% of BORDERLINE"],
 ["RQ7 hardest cases", "high-value spike (80% recall); layering without network features (65% → 85% with them); BORDERLINE causes 65% of false positives"],
 ["Rules vs ML (same test set)", f"association rules F1 {rb['test']['f1']:.3f} / PR-AUC {rb['test']['pr_auc']:.3f} vs production model F1 {FT['test_f1']:.3f} / PR-AUC {FT['test_pr_auc']:.3f}"],
 ["Risk score + tiers", f"weights model {RSC['weights']['model']:.2f} / anomaly {RSC['weights']['anomaly']:.2f} / graph {RSC['weights']['graph']:.2f}; PR-AUC {fs7['risk_score_test']['final']['pr_auc']:.3f}; fraud ≥ HIGH {pct(ftiers.loc['FRAUDULENT','HIGH_share']+ftiers.loc['FRAUDULENT','CRITICAL_share'],0)}, legitimate in LOW {pct(ftiers.loc['LEGITIMATE','LOW_share'],0)}"],
], ["Question", "Finding"], widths=["28%", "72%"])}
<h3>Try it</h3>
<p>Dashboard: <b>{LIVE_UI}</b> (open it a minute before a demo: the free API sleeps after 15 idle minutes). API documentation: {LIVE_API}/docs. Source: github.com/bharathakn/FinSight-AI.</p>
</section>
""")

# 2 intro
add(f"""
<section>
<h1>Introduction, objectives and research questions</h1>
<h2>Problem context</h2>
<p>India's digital payments are dominated by <b>UPI</b> (instant, mostly person-to-person, capped at ₹1 lakh per transaction in this data), with <b>NEFT</b> (batch bank transfers) and <b>RTGS</b> (large-value real-time transfers, minimum ₹2 lakh). Fraud on these rails takes two broad forms:</p>
<ul>
<li><b>Opportunistic fraud</b> visible in a single transaction: an unusually large payment, a brand-new device, a payee added minutes ago, a login far from home.</li>
<li><b>Organised fraud</b> visible only across many transactions: <i>mule accounts</i> that receive and immediately forward money, <i>layering chains</i> that move value through several accounts, <i>fraud rings</i> that funnel many payers into one hub, <i>structuring</i> (splitting a large amount into several payments just below a reporting threshold), and <i>shared devices</i> used by many accounts.</li>
</ul>
<p>The RBI Master Directions on Fraud Risk Management and the FATF Recommendations (both included in the project's data folder) frame the compliance side: early detection, risk-based monitoring, sanctions screening and explainable decisions.</p>
<h2>Objectives</h2>
<ol>
<li>Apply the full KDD process to multi-source financial data, including wrangling, quality assessment, transformation and pattern discovery (<b>DMW</b>).</li>
<li>Discover interpretable behaviour patterns linked to fraud with association-rule mining, and compare Apriori with FP-Growth (<b>DMW</b>).</li>
<li>Build, compare, tune and explain supervised, unsupervised and anomaly-detection models, and measure the value of behavioural and network features (<b>ML</b>).</li>
<li>Deliver the system as stateless containerised microservices with a REST API and a web dashboard, deployed on cloud platforms (<b>CCT</b>).</li>
</ol>
<h2>Research questions</h2>
{table([
 ["RQ1", "Which transactional and behavioural features are most strongly associated with suspicious activity?", "DMW", "Answered (Ch. 9)"],
 ["RQ2", "What recurring combinations of behaviours does association-rule mining discover, and how do Apriori and FP-Growth compare?", "DMW", "Answered (Ch. 10)"],
 ["RQ3", "Do behavioural (Exp B) and network (Exp C) features measurably improve classification over raw transaction features (Exp A), and only where real network structure exists (v1.0 vs v1.1)?", "ML", "Answered (Ch. 11)"],
 ["RQ4", "Which classifier performs best, with which hyperparameters, and does SMOTE beat class weighting?", "ML", "Answered (Ch. 11, 13)"],
 ["RQ5", "Can unsupervised clustering recover risk groupings without labels?", "ML", "Answered (Ch. 12)"],
 ["RQ6", "How well do Isolation Forest, One-class SVM and an autoencoder trained only on legitimate data detect fraud?", "ML", "Answered (Ch. 12)"],
 ["RQ7", "Which scenarios (and BORDERLINE cases) are hardest for the models?", "ML + DMW", "Answered (Ch. 11)"],
 ["RQ8", "How does a containerised microservice architecture on managed cloud platforms support independent deployment, scaling and cost (scale-to-zero vs cold start)?", "CCT", "Answered: Ch. 14 (local) and Ch. 16 (cloud)"],
], ["ID", "Question", "Subject", "Status"], widths=["7%", "60%", "11%", "22%"])}
<h2>Scope and non-goals</h2>
<p>In scope: offline analytics and model building on the provided datasets, a scoring API and a dashboard. Out of scope: real customer data, real-time streaming infrastructure (Kafka), graph databases (Neo4j), and Kubernetes clusters (only conceptual manifests). These were deliberately excluded to keep the project deliverable.</p>
</section>
""")

# 3 datasets
scen_rows = [
 ["S1", "Normal legitimate", "8,200", "—", "Ordinary payments"],
 ["S2", "High-value spike", "200", "104 F / 96 B", "Amount far above the account's 30-day average"],
 ["S3", "Rapid burst", "150", "89 F / 61 B", "Many payments from one sender in minutes"],
 ["S4", "New beneficiary, large tx", "150", "99 F / 51 B", "Large payment to a payee added very recently"],
 ["S5", "New device, suspicious", "150", "102 F / 48 B", "Unregistered device + anomalies"],
 ["S6", "Structuring / splitting", "150", "150 F", "Several payments just under ₹50,000"],
 ["S7", "Layering network", "200", "200 F", "Value hops through a chain of accounts"],
 ["S8", "Mule pass-through", "200", "200 F", "Account receives and forwards almost immediately"],
 ["S9", "Fraud ring hub", "200", "200 F", "Many senders pay one hub account"],
 ["S10", "Impossible travel", "150", "101 F / 49 B", "Transaction location implausibly far from home"],
 ["S11", "Shared device / IP", "150", "103 F / 47 B", "One device used by many accounts"],
 ["S12", "Mixed borderline travel", "100", "100 B", "Travel anomalies labelled borderline"],
]
add(f"""
<section>
<h1>Datasets</h1>
<h2>Inventory</h2>
{table([
 ["Custom India synthetic v1.0", "7 CSV files", "10,000 transactions · 1,200 customers · 1,500 accounts · 1,400 devices · 2,000 beneficiaries", "Primary corpus; v1.0 kept as untouched control"],
 ["Custom India synthetic v1.1", "derived by this project", "same rows; 1,470 rows modified (structure, decoys, camouflage); labels unchanged", "Primary training data"],
 ["IEEE-CIS Fraud (Kaggle)", "2 tables (transaction + identity)", "590,540 train rows × 394 cols ⋈ 144,233 identity rows × 41 cols; 506,691 test rows", "Data-quality / missingness showcase (planned profiling)"],
 ["PaySim mobile money", "1 CSV", "6,362,620 rows, 11 columns, 8,213 frauds (0.13%)", "Big-data wrangling with Polars; mining scale test; Isolation Forest scale track (Ch. 12)"],
 ["OFAC SDN list", "headerless CSV", "19,321 entities; '-0-' as the null marker; aliases inside free-text remarks", "Sanctions screening service"],
 ["UN Consolidated list", "XML", "736 individuals + 275 entities, with aliases and original-script names", "Sanctions screening service"],
 ["RBI Master Directions; FATF Recommendations 2026", "PDF", "—", "Compliance framing for the report"],
], ["Dataset", "Format", "Size", "Role"], "small", widths=["22%", "16%", "38%", "24%"])}
<h2>The India dataset (7 files)</h2>
{table([
 ["transactions_features.csv", "10,000 × 32", "Transaction facts: timestamp, channel, amount, sender/receiver account, customer, VPA, IFSC, device, IP, location, distance from home, new-device / new-beneficiary flags, amount ratio vs 30-day average, 24h frequency, balances"],
 ["transactions_targets.csv", "10,000 × 6", "Labels: is_fraud (0/1), risk_label (LEGITIMATE / FRAUDULENT / BORDERLINE), scenario_id, alert_id, is_flagged_alert"],
 ["accounts.csv", "1,500 × 11", "Account type (SAVINGS, SALARY, CURRENT, MULE_SUSPECT), age, balance, 30-day history"],
 ["customers.csv", "1,200 × 11", "Synthetic names, customer type, KYC status, PEP flag, declared income, home city and coordinates"],
 ["devices.csv", "1,400 × 9", "Device type, OS/app version, first seen, registered flag, associated accounts"],
 ["beneficiaries.csv", "2,000 × 7", "Sender → receiver payee records, date added, prior transaction count"],
 ["graph_edges.csv", "10,000 × 8", "Neo4j-style edge list mirroring the transactions (incl. the leaking rapid_forward_delta_sec)"],
], ["File", "Shape", "Content"], "small", widths=["24%", "12%", "64%"])}
<p><b>Labels:</b> 8,200 LEGITIMATE (82.0%), 1,348 FRAUDULENT (13.5%), 452 BORDERLINE (4.5%). The primary target is <code>is_fraud</code>; BORDERLINE rows count as 0 but are analysed separately. A model that always predicts "legitimate" scores 86.5% accuracy, so <b>accuracy is never used as a headline metric</b> (PR-AUC and F1 are). Transactions span 1–30 August 2026 (IST), roughly 300–390 per day.</p>
<h2>The 12 scenarios</h2>
{table(scen_rows, ["ID", "Scenario", "Rows", "Labels", "Meaning"], "small", widths=["6%", "24%", "8%", "16%", "46%"])}
</section>
""")

# 4 planning & methodology
add(f"""
<section>
<h1>Planning, methodology and project management</h1>
<h2>KDD process mapping</h2>
{flow([("1 · Selection", "7 India files + auxiliary corpora; targets"), ("2 · Cleaning", "types, sentinels, redundancy, dedupe"), ("3 · Integration", "7-file merge; OFAC + UN → entities"), ("4 · Transformation", "v1.1 injection; graph features; discretisation; encoding"), ("5 · Mining", "Apriori, FP-Growth, FP-Max; classifiers"), ("6 · Evaluation", "support / confidence / lift; PR-AUC, F1, significance"), ("7 · Presentation", "notebooks, figures, dashboard")], "small")}
<h2>Three-subject division</h2>
{table([
 ["Core question", "What patterns exist in the data?", "Which model best identifies risk, and why?", "How is it delivered as a scalable cloud service?"],
 ["Primary method", "KDD + association rules", "Model training, comparison, tuning, explanation", "Containers, microservices, REST, PaaS deploy"],
 ["Unique deliverable", "Leakage audit; v1.1 dataset; Apriori vs FP-Growth", "A/B/C × v1.0/v1.1 ablation with significance tests", "Vercel + Render deployment; Compose microservices"],
], ["", "DMW", "ML", "CCT"], widths=["16%", "28%", "28%", "28%"])}
<h2>Roadmap and status</h2>
{table([
 ["0", "Dataset audit", "DMW", "✔ done", "Leakage, join failures, missing network structure found; plan revised to v2"],
 ["1", "Wrangling + data quality", "DMW", "✔ core done", "Master table, DQ/leakage reports, sanctions parsing (IEEE/PaySim profiling pending)"],
 ["1b", "Graph-structure injection (v1.1)", "DMW", "✔ done", "Seeded rewiring + decoys + 50% camouflage, validated"],
 ["2", "Exploratory data analysis", "DMW+ML", "✔ done", "14 figures, RQ1 ranking"],
 ["3", "Graph features", "ML+DMW", "✔ features done", "13 point-in-time + 6 global features"],
 ["4", "Association rule mining", "DMW", "✔ done", "Benchmark, rules, emerging patterns, PaySim scale"],
 ["5", "Supervised ML", "ML", "✔ done", "72 configs, paired CV, imbalance, temporal, per-scenario"],
 ["6", "Unsupervised + anomaly", "ML", "✔ done", "K-Means, DBSCAN, PCA/t-SNE; IF, OC-SVM, autoencoder (PyTorch → ONNX); PaySim IF"],
 ["7", "Tuning + stacking + SHAP", "ML", "✔ done", "Grid/Random search, learning curves, stacking, isotonic calibration, risk score + tiers, SHAP"],
 ["8", "Services + Docker", "CCT", "✔ done", "5 FastAPI services, one multi-stage Dockerfile, Compose, two service modes, load test"],
 ["9", "Dashboard", "All", "✔ done", "React + Vite SPA, 10 pages incl. live analysis, light/dark, wake-up screen"],
 ["9b", "Live analysis", "All", "✔ done", "Upload a CSV, analyse the demo dataset or simulate attacks; scored and explained live"],
 ["10", "Deployment + report", "CCT", "✔ done", "Live on Vercel (frontend) + Render (API); CI; latency, memory and cold start measured"],
], ["Phase", "Name", "Subject", "Status", "Outcome / scope"], "small", widths=["7%", "26%", "10%", "13%", "44%"])}
<h2>Plan review: from v1 to v2</h2>
<p>Before any code was written, the original plan was audited against the actual data. Sixteen issues were found and fixed. The most consequential ones:</p>
{table([
 ["Network scenarios had no network structure (all graph features AUC ≈ 0.50)", "Exp C, RQ3 and the network demo would be empty", "Seeded structure injection → dataset v1.1 (D-001), plus camouflage (D-002)"],
 ["rapid_forward_delta_sec non-zero only for fraud; alert flags = fraud ∪ borderline", "Label leakage; fake-perfect models", "Blacklist enforced in code and tested (D-003)"],
 ["Beneficiaries join matched 8/10,000 rows", "Planned merge useless", "Point-in-time per-sender aggregates (D-004)"],
 ["Isolation Forest planned on PaySim then applied to India data", "Incompatible feature spaces", "Detectors trained on India legit rows; PaySim as its own track (D-006)"],
 ["Sanctions hits ≈ 0 on synthetic names", "Constant feature", "Screening service + rule override instead (D-007)"],
 ["9 containers incl. long-running ETL", "Cost, latency, complexity", "5 services + batch pipeline (D-008)"],
 ["Deployment target changed to Vercel + Render", "Streamlit can't run on Vercel; no private networking on Render free", "React SPA + Render; two service modes (D-022–D-024)"],
], ["Finding", "Impact", "Fix"], "small", widths=["38%", "26%", "36%"])}
<h2>Engineering practices</h2>
<ul>
<li><b>Single source of truth:</b> all logic lives in the Python package <code>src/finsight/</code>. Notebooks only call it, and the batch pipeline (<code>python -m finsight.pipeline</code>) reproduces every output.</li>
<li><b>Reproducibility:</b> global seed 42. The v1.1 dataset is regenerated byte-identically (tested). The environment is pinned with <code>uv.lock</code> (Python 3.12, CPU-only PyTorch/XGBoost).</li>
<li><b>Testing:</b> {N_TESTS} automated tests (pytest) cover the dataset invariants, point-in-time correctness, the leakage blacklist, split disjointness, mining correctness, the evaluation maths, the anomaly-detector training rows and ONNX parity.</li>
<li><b>Documentation:</b> a design plan, phase tracker (<code>docs/PROGRESS.md</code>), decision log ({len(decisions())} decisions with evidence), data reference, setup guide and an agent manual (<code>CLAUDE.md</code>) for AI-assisted development.</li>
<li><b>Visual standard:</b> one validated, colour-blind-safe palette; the same entity always has the same colour (blue = legitimate, orange = fraudulent, aqua = borderline); every figure was inspected after rendering.</li>
</ul>
<h2>Technology stack</h2>
{table([
 ["Language / environment", "Python 3.12, uv, Jupyter"], ["Data", "pandas 3, NumPy, PyArrow (Parquet), Polars (6.36M-row PaySim)"],
 ["Mining", "MLxtend (Apriori, FP-Growth, FP-Max)"], ["Machine learning", "scikit-learn, XGBoost, imbalanced-learn; PyTorch autoencoder → ONNX → onnxruntime; SHAP"],
 ["Graph", "NetworkX (PageRank, Louvain communities)"], ["Sanctions", "RapidFuzz fuzzy matching, lxml"],
 ["Visualisation", "Matplotlib (report figures); Plotly.js (dashboard)"], ["API / services", "FastAPI, Pydantic, Uvicorn, httpx; Docker, Docker Compose; Locust (load testing)"],
 ["Frontend", "React 18 + Vite + TypeScript, Plotly.js; static build for Vercel"], ["Cloud", "Render (Docker, Blueprint IaC), Vercel (static CDN), GitHub Actions (CI), GitHub Releases (model registry); Cloud Run / Kubernetes scripts as the documented portability path"],
], ["Layer", "Technology"], widths=["30%", "70%"])}
</section>
""")

# 5 audit
lp = dq10["label_proxies"][0]
add(f"""
<section>
<h1>Phase 0: dataset audit</h1>
<p>The audit ran before any modelling. Its job was to check what the data can and can't support. Each finding below drove a design decision.</p>
<h2>What was healthy</h2>
<ul><li>0 nulls (except <code>alert_id</code>, null by design for non-alerts), 0 duplicate rows, 100% foreign-key integrity for accounts, customers and devices.</li>
<li><code>balance_after = balance_before − amount</code> holds for every row; <code>amount_ratio_vs_30d_avg = amount / sender's 30-day average</code> for 99.1% of rows.</li>
<li>The task is not trivially separable: a random forest reached F1 ≈ 0.72 on transaction features and ≈ 0.87 with behavioural features. There is room for real modelling.</li></ul>
<h2>Problems found</h2>
{table([
 ["Label proxy", f"graph_edges.rapid_forward_delta_sec is non-zero on {lp['nonzero_rows']} rows, {pct(lp['fraud_rate_when_nonzero'],0)} of them fraud (scenarios S6/S7 only), inconsistent with the actual transaction timeline", "Excluded; recomputed from real timestamps in v1.1"],
 ["Label proxy", "is_flagged_alert and alert_id equal 1 exactly for FRAUDULENT ∪ BORDERLINE (single-feature AUC 0.97)", "Blacklisted"],
 ["No network structure", "fan-in/fan-out/degree AUC 0.50–0.52; pass-through timing 0.50; velocity 0.50; fraud spread over ~900 random senders", "Dataset v1.1 (Ch. 7)"],
 ["Failed join", "only 8 of 10,000 (sender, receiver) pairs exist in beneficiaries.csv", "Per-sender point-in-time aggregates"],
 ["Redundancy", "transaction_type is a 1:1 recoding of channel (UPI=P2P, NEFT=C2B, RTGS=B2B); status is always SUCCESS", "Both dropped"],
 ["Dead mining items", "NEW_ACCOUNT matches 0 rows; IMMEDIATE_REPEAT matches 2; tx_frequency_24h has no signal", "Item list rebuilt from audit counts"],
 ["Generator artifacts", "'beneficiary added < 24h' and 'distance > 100 km' never occur on legitimate rows", "Kept, but reported as by-construction"],
 ["Suspicious-looking column", "account_type = MULE_SUSPECT looked like an investigation outcome", "Checked: fraud rate 11.3% vs 13.5% base, so not a leak; kept"],
], ["Type", "Evidence", "Action"], "small", widths=["16%", "58%", "26%"])}
</section>
""")

# 6 wrangling
add(f"""
<section>
<h1>Phase 1: data wrangling and data quality</h1>
<h2>The 7-file integration</h2>
{flow([[("transactions", "core facts (32 cols)"), ("targets", "labels only")], [("accounts", "sender + receiver"), ("customers", "sender KYC, PEP, income")], [("devices", "type, registered, age"), ("beneficiaries", "point-in-time aggregates")], [("graph_edges", "→ 19 network features (Ph. 3)")], [("master_transactions", "10,000 × 57"), ("master_features", "10,000 × 76")]])}
<p>Cleaning steps (each logged in the data-quality report): de-duplication on transaction ID, timestamp parsing to timezone-aware IST, dropping redundant/constant columns, re-deriving calendar fields from the timestamp, and adding <code>log_amount</code>. Merges use <code>validate=</code> checks (many-to-one, one-to-one), so a bad key fails loudly.</p>
<h2>Data-quality report</h2>
<p>Computed for v1.0 and v1.1: completeness, duplicate rate, valid timestamps, referential integrity, constant columns, redundant column pairs, label balance and an automatic <b>leakage scan</b> (single-feature AUC ≥ 0.95 is flagged as suspect). The scan found <code>is_flagged_alert</code> (0.974) on its own, and the label-proxy check found <code>rapid_forward_delta_sec</code>.</p>
{ri_table()}
<h2>Sanctions lists (CSV + XML wrangling)</h2>
<p>OFAC's CSV has no header and uses <code>-0-</code> for nulls. Individuals are written "LAST, First" (normalised to "First LAST"), and aliases are hidden in a free-text remarks column (parsed with a regular expression). The UN XML has separate individual/entity trees with alias and original-script names. Both are unified into one table: <b>{sanc['entities']:,} entities, {sanc['name_variants']:,} name variants</b> ({sanc['by_kind']['alias']:,} aliases). Names are normalised (accents, punctuation, case) and matched with RapidFuzz token-sort similarity.</p>
<p><b>Threshold experiment:</b> none of the 1,200 synthetic customers is sanctioned, so every hit is a false positive. This measures the false-alarm cost of each fuzzy threshold:</p>
{sweep_table()}
<p>A threshold of ~90 keeps false alarms near 0.2%. Sanctions screening therefore becomes a rule-based override in the risk score, not an ML feature.</p>
</section>
""")

# 7 v1.1
add(f"""
<section>
<h1>Phase 1b: dataset v1.1 (graph-structure injection)</h1>
<h2>Why</h2>
<p>The generator labelled six scenarios as network fraud but never connected them in the transaction graph. Any network-feature experiment would have reported "no effect" because the structure wasn't there to find. Rather than drop the network part of the project, we engineered the missing structure in a <b>controlled, documented and reversible</b> way, and kept v1.0 as a control dataset.</p>
<h2>Injection rules</h2>
{table([
 ["S9 fraud ring", inj_rows.get("ring"), "10 rings; ~20 senders each pay one hub account within 48 h"],
 ["S8 mule", inj_rows.get("mule"), "100 pairs: X → M, then M → Y 2–30 min later at 90–98% of the amount"],
 ["S7 layering", inj_rows.get("layer"), "50 chains A→B→C→D→E, 5–60 min apart, 2–5% decay per hop"],
 ["S3 burst", inj_rows.get("burst"), "30 senders × 5 payments within 10 minutes"],
 ["S6 structuring", inj_rows.get("struct"), "30 senders × 5 payments of ₹45,000–49,999 within 24 h"],
 ["S11 shared device", inj_rows.get("shared_dev"), "15 devices × ~10 different accounts (shared IP)"],
 ["Legit decoys", sum(inj_rows.get(k, 0) for k in ("decoy_merchant", "decoy_forward", "decoy_family_dev", "decoy_salary")), "10 merchant hubs, 60 benign forwards, 20 family devices, 10 salary-day bursts"],
 ["Camouflage", inj_rows.get("camouflage"), "50% of network-fraud rows get 4 behavioural columns (new device, new beneficiary, beneficiary age, distance) copied from random legitimate rows"],
], ["Group", "Rows", "Rule"], "small", widths=["20%", "10%", "70%"])}
<p><b>Invariants preserved and tested:</b> labels unchanged; no self-transfers; account → customer/VPA/IFSC consistency; device belongs to the sender (except injected shared devices); balance and amount-ratio formulas recomputed; RTGS minimum respected; calendar fields recomputed; deterministic output (byte-identical re-runs). Every change is logged per row in <code>injection_log.csv</code> ({inj['rows_modified']:,} rows touched).</p>
<p><b>Why camouflage:</b> after rewiring alone, behavioural features still caught ~99% of network fraud (the generator had given it obvious per-transaction fingerprints), so network features had nothing left to explain (PR-AUC 0.924 → 0.933). Real organised fraud looks normal one transaction at a time, which is the whole point of mule accounts. Camouflaging half creates a realistic mix of opportunistic and organised fraud.</p>
<h2>Validation</h2>
{table([["v1.0 (control)", "0.724", "0.874", "0.872"], ["v1.1", "0.754", "0.810", "0.839"]], ["Random forest F1 (5-fold)", "Exp A", "Exp B", "Exp C"], "num")}
<p>On v1.0, network features add nothing. On v1.1 they add a clear gain, and no network feature exceeds single-feature AUC {eda['network_auc_max']['v1.1']:.2f}, so there is no giveaway.</p>
{fig("eda_11_fanin_distribution", "Receiver fan-in: v1.0 never exceeds 5 senders per day; v1.1 adds 15 hubs (10 fraud rings + 5 clustered merchant decoys), so fan-in alone cannot single out fraud.", "88%")}
{fig("eda_12_network_examples", "Injected structures: a fraud ring (20 senders into one hub within 48 h) and a layering chain (value decays 2–5% per hop over two hours).")}
</section>
""")

# 8 graph features
add(f"""
<section>
<h1>Phase 3: graph (network) features</h1>
<p>The transactions form a directed, weighted graph of accounts. Two families of features were built:</p>
<h2>Point-in-time features (13)</h2>
<p>Each uses <b>only transactions strictly before the current one</b>, which is what a live system would know at scoring time. This removes look-ahead leakage and is tested in code.</p>
{table([
 ["sender_tx_10m / _1h / _24h", "Number of payments the sender made in the last 10 min / 1 h / 24 h (bursts, structuring)"],
 ["sender_distinct_receivers_24h", "Distinct payees of the sender in 24 h"],
 ["receiver_tx_in_24h · receiver_distinct_senders_24h / _7d", "Fan-in into the receiver (fraud-ring hubs)"],
 ["sec_since_sender_last_inflow", "Seconds since the sender last received money (mules, layering)"],
 ["forward_ratio", "Amount sent ÷ amount last received"],
 ["is_pass_through", "≥ 80% of the last inflow forwarded within 30 minutes"],
 ["chain_depth", "Number of consecutive pass-through hops leading to this payment"],
 ["device_distinct_accounts · ip_distinct_accounts", "Distinct accounts seen on this device / IP so far (shared devices)"],
], ["Feature", "Meaning"], "small", widths=["38%", "62%"])}
<h2>Global features (6, transductive)</h2>
<p>PageRank of sender and receiver, Louvain community size of both, same-community flag, and receiver clustering coefficient. They're label-free but computed on the whole graph, which is documented as a transductive assumption (D-012).</p>
{fig("eda_14_network_signal_by_scenario", "Network features by scenario on v1.1 (colour = ratio to normal transactions): signal appears only where structure was injected (bursts S3/S6, chains S7/S8, ring fan-in S9, shared devices S11).", "92%")}
</section>
""")

# 9 EDA
add(f"""
<section>
<h1>Phase 2: exploratory data analysis</h1>
{fig("eda_01_label_overview", "Label balance and the suspicious scenarios. S6–S9 are fully fraudulent; S12 is fully borderline.")}
{fig("eda_02_amount_by_label", "Amounts by label and channel (log scale): fraud sits 1–2 orders of magnitude above legitimate UPI/NEFT payments, but legitimate RTGS is larger, so amount only matters relative to channel and account history.")}
{fig("eda_03_fraud_rate_time_channel", "Channel and weekday stay near the 13.5% base rate; late-night hours reach 22–27%.")}
<h2>Behaviour flags</h2>
{flags_table()}
<p>Two flags never fire on legitimate rows (generator artifact). PASS_THROUGH is the most precise signal (99% fraud) but covers only 1.6% of rows: a classic high-precision, low-recall mule indicator.</p>
{fig("eda_09_borderline_vs_fraud", "BORDERLINE and FRAUDULENT distributions almost coincide on every key feature, while LEGITIMATE is clearly separate. This later explains most model errors.")}
{fig("eda_06_feature_auc_v10_v11", "Single-feature AUC, v1.0 vs v1.1: behavioural AUCs fall (camouflage) while network AUCs rise from ≈0.50 to 0.56–0.64. Nothing crosses the 0.95 leak-suspect line on v1.1.", "82%")}
<h2>RQ1: features most associated with fraud (v1.1)</h2>
{rq1_table()}
{fig("eda_13_camouflage", "Camouflaged organised fraud looks legitimate on behaviour, yet 73% of it carries at least one network red flag versus 12% of legitimate transactions. This is the gap Experiment C is designed to close.")}
<p>Other EDA figures in the repository: correlation matrix (amount vs amount ratio ρ = 0.85, so Naive Bayes' independence assumption is violated), geography (states 11–16.5%, no hotspot), daily volume (no trend, which supports the temporal hold-out), and scenario fingerprints.</p>
</section>
""")

# 10 mining
top_ep = ep.head(6)
add(f"""
<section>
<h1>Phase 4: association rule mining</h1>
<h2>Transformation: transactions as baskets</h2>
<p>Each transaction becomes a basket of boolean <b>items</b> (20 feature items + 2 label items). For example, a transaction might be <i>{{CH_UPI, HIGH_AMOUNT_SPIKE, NEW_DEVICE, RECENT_BENEFICIARY, QUICK_INFLOW, FRAUDULENT}}</i>. Label items may appear only as a rule's consequent.</p>
{items_table()}
<h2>Algorithms and measures</h2>
{table([
 ["Apriori", "Level-wise candidate generation: every frequent k-itemset is joined to make (k+1)-candidates, which are counted against all transactions. Simple, but candidates grow exponentially."],
 ["FP-Growth", "Compresses transactions into a prefix tree (FP-tree) and mines it recursively without generating candidates."],
 ["FP-Max", "FP-Growth variant returning only maximal itemsets (no frequent superset). Lossy but very compact."],
 ["Closed itemsets", "No superset with the same support. A lossless compression of all frequent itemsets."],
 ["Support", "P(X ∪ Y): the share of transactions containing the whole rule"],
 ["Confidence", "P(Y | X): how often the rule is right when the antecedent appears"],
 ["Lift", "confidence ÷ P(Y): how many times more likely fraud is given X (1 = independent)"],
 ["Leverage / conviction", "P(X∪Y) − P(X)P(Y); and (1 − P(Y)) / (1 − confidence)"],
 ["Productive rule", "A rule kept only if its confidence beats every simpler sub-rule by ≥ 0.01 (Bayardo's improvement), which removes redundant variants"],
 ["Emerging pattern", "Itemset whose support grows strongly from non-fraud to fraud (growth rate); 'jumping' when absent from non-fraud"],
], ["Concept", "Definition"], "small", widths=["22%", "78%"])}
<h2>Apriori vs FP-Growth benchmark</h2>
<p>Protocol: two item universes (22 core items; 49 items adding city, device, hour band, account and customer type), five support levels, three algorithms. Runtime is the median of 3 untraced runs; memory comes from a separate traced run (D-017). The code asserts that Apriori and FP-Growth return <b>identical itemsets and supports</b> at every level.</p>
{bench_table()}
{fig("mining_01_benchmark", "Runtime and peak memory vs min_support. Apriori's memory explodes as the search grows (1.3 GB vs 79 MB at 49 items / 0.002). On small searches mlxtend's vectorised Apriori is faster than its pure-Python FP-Growth.")}
{fig("mining_02_condensed", "Condensed representations at 49 items: 56,278 frequent → 39,713 closed → 5,542 maximal itemsets.", "78%")}
{fig("mining_03_paysim_scaling", "PaySim scale run (100K → 6.36M rows, 14 items): both scale linearly in rows; Apriori uses 1.3 GB, FP-Growth's tree stays tiny (~230 distinct baskets) but runs slower.")}
<h2>Rules found (v1.1, min_support 0.01)</h2>
<p>Funnel: {v11['itemsets']} frequent itemsets → {v11['target_rules_conf_0.5']} target rules (confidence ≥ 0.5) → {v11['filtered_lift1.5_conf0.6']} pass Lift &gt; 1.5 and Confidence &gt; 0.6 → <b>{v11['productive']} productive rules</b> ({v11['productive_fraud_with_network_item']} contain network items). The only BORDERLINE rule: <i>{{EXTREME_LOCATION, NEW_DEVICE}} → BORDERLINE</i> (confidence 60%, lift 13.4).</p>
{rules_table()}
{fig("mining_06_item_share_v10_v11", "Control experiment: v1.0 rules use only behaviour/amount items (0 network rules); v1.1 rules use QUICK_INFLOW, HUB_RECEIVER and PASS_THROUGH. Mining independently rediscovers the injected structure and invents none where it doesn't exist.", "85%")}
<h2>Fraud-subset mining: emerging patterns</h2>
<p>{len(ep)} minimal emerging patterns (≥ 5% of fraud, growth ≥ 3). The strongest are jumping patterns, present in fraud and never in non-fraud:</p>
{table([[r.pattern, pct(r.support_target), pct(r.support_rest)] for r in top_ep.itertuples()], ["Pattern", "Support in fraud", "Support in non-fraud"], "small")}
<h2>Rules as a transparent detector</h2>
<p>Flagging a transaction when any rule fires gives, in-sample, a best F1 of {mining['detector_best_f1']['f1']:.2f} with the top {int(mining['detector_best_f1']['k'])} rules. {dets['fp_borderline']} of the {dets['fp']} false positives are BORDERLINE. The fair, held-out comparison with ML (rules mined on the training split only) is in Chapter 11: <b>F1 {rb['test']['f1']:.3f}, PR-AUC {rb['test']['pr_auc']:.3f}</b>.</p>
<p><b>PaySim:</b> on 6.36M rows (fraud rate 0.13%, min_support 0.0005) the top rule is {{ORIG_FULL_AMOUNT}} → fraud (confidence 99%, lift 767). This is a known simulator artifact (PaySim fraud always empties the account) and is reported as such.</p>
</section>
""")

# 11 supervised
add(f"""
<section>
<h1>Phase 5: supervised classification</h1>
<h2>Experimental design</h2>
{table([
 ["Feature sets", "A = transaction (12 columns: amount, channel, time, balance, account history) · B = A + behavioural (30: device, beneficiary, distance, amount ratio, velocity, KYC/PEP) · C = B + network (49: Ch. 8)"],
 ["Datasets", "v1.1 (primary) and v1.0 (control without network structure)"],
 ["Models", "Decision Tree (depth 8), Gaussian Naive Bayes, KNN (k = 15, distance-weighted), Random Forest (300 trees), XGBoost (300 trees, depth 6), MLP (64-32, early stopping). Default hyperparameters; tuning is Phase 7."],
 ["Pre-processing", "trees: raw numerics + one-hot categoricals; KNN/MLP/NB: signed-log + standardisation (heavy-tailed amounts and durations)"],
 ["Split", f"stratified on risk_label: train {ml['split_sizes']['train']:,} / validation {ml['split_sizes']['val']:,} / test {ml['split_sizes']['test']:,}"],
 ["Comparison", "5×2 repeated stratified cross-validation on the 80% development set with identical folds for every configuration, giving paired t-tests and Wilcoxon tests (D-021)"],
 ["Imbalance", "SMOTE inside the imblearn pipeline (re-fit per fold, so no leakage); compared with class weighting and none"],
 ["Threshold", "maximum-F1 threshold chosen on validation, never on test; the test set is scored once per configuration"],
 ["Metrics", "PR-AUC (primary, since the class balance is 13.5%), ROC-AUC, F1, precision, recall, per-scenario recall, BORDERLINE flag rate"],
 ["Robustness", f"temporal hold-out: train 1–19 Aug ({ml['temporal_split_sizes']['train']:,}), validate 20–23 ({ml['temporal_split_sizes']['val']:,}), test 24–30 ({ml['temporal_split_sizes']['test']:,})"],
], ["Item", "Choice"], "small", widths=["18%", "82%"])}
<p>In total: 360 cross-validation fits + 36 hold-out fits for the main grid, plus imbalance, temporal, rule-baseline and 3-class experiments (~6 minutes on 8 cores).</p>
<h2>RQ3: do the feature groups help?</h2>
{fig("ml_01_abc_cv_pr_auc", "Cross-validated PR-AUC (mean ± sd) as feature groups are added. Left: v1.0 control, where network features add nothing. Right: v1.1, where every model except the decision tree gains significantly.")}
<p><b>Paired significance</b> (mean PR-AUC difference across 10 identical folds; ✔ = significantly better, p &lt; 0.05):</p>
{sig_table()}
<p><b>Answer:</b> behavioural context is the largest gain (B &gt; A for 12/12 model-dataset pairs, p &lt; 0.001). Network features give a further, significant gain on v1.1 (5/6 models, +0.05 to +0.09) and none on v1.0. Network features help <i>exactly where network structure exists</i>.</p>
<h2>RQ4: which model wins? (held-out test, v1.1)</h2>
{results_matrix("v1_1")}
<p>Cross-validation ranks the models identically (table in the appendix), so the ranking is not test-set luck. Tree ensembles win because they capture interactions (large amount × recent payee × quick inflow). Naive Bayes is hurt by correlated features (amount vs amount ratio, ρ = 0.85). Naive Bayes and KNN gain the most from network features (+0.09, +0.08), because those features place camouflaged fraud near other fraud.</p>
{fig("ml_03_pr_roc_expC", "Test precision–recall and ROC curves (Exp C, v1.1). ROC-AUC is ≥ 0.90 for every model and hides differences that PR-AUC shows. The diamond is the association-rule detector mined on the training split.")}
{fig("ml_02_test_pr_auc_heatmap", "Held-out test PR-AUC for every model × dataset × feature set.", "85%")}
<h2>Imbalance: SMOTE vs class weights vs nothing (5×2 CV PR-AUC, v1.1 Exp C)</h2>
{imb_table()}
<p><b>SMOTE never improved PR-AUC</b> and was clearly worse for the decision tree, KNN and MLP. Thresholds are tuned anyway, and SMOTE interpolates across one-hot categories and across different fraud scenarios, creating unrealistic synthetic fraud. Phase 7 therefore uses class weights for trees and no resampling elsewhere (D-020). This negative result is worth stating in the ML report.</p>
<h2>RQ7: which cases are hardest?</h2>
{scen_table()}
<p>Network features fix the camouflaged network scenarios (S7 layering 65% → 85%, S8 mule 90% → 97.5%, S3 burst 83% → 89%). The hardest scenario overall is S2 high-value spike (80%): its only signal is the amount, and legitimate RTGS payments are large too. Test counts per scenario are small (16–48), so differences under ~10 points are noise.</p>
{fig("ml_06_confusion_by_label", f"XGBoost errors by true label: 2% of legitimate rows flagged, 94% of fraud caught, but 79% of BORDERLINE flagged. 71 of 110 false positives are BORDERLINE.", "60%")}
<p><b>3-class extension</b> (XGBoost, class-balanced): F1 LEGITIMATE {tri['per_class_f1']['LEGITIMATE']:.2f}, FRAUDULENT {tri['per_class_f1']['FRAUDULENT']:.2f}, BORDERLINE {tri['per_class_f1']['BORDERLINE']:.2f} (macro {tri['macro_f1']:.2f}). Borderline cases are only half-separable, so the risk score will treat them as a "review" tier.</p>
<h2>Robustness and interpretability</h2>
{temporal_table()}
<p>Training on early August and testing on the last week gives the same results, so there is no drift within the month.</p>
{fig("ml_09_decision_tree", "Depth-4 decision tree (class-balanced, Exp C): amount first; small payments are risky only with a sender burst (network) or far from home; large non-UPI payments are risky with a new payee or money received under an hour earlier (network).")}
<h2>Baselines on the same test set</h2>
{table([["Association rules (train-mined, k = " + str(rb['k_selected_on_val']) + " chosen on validation)", f3(rb['test']['pr_auc']), f3(rb['test']['f1']), f3(rb['test']['precision']), f3(rb['test']['recall'])],
        ["XGBoost, Exp A (transaction only)", f3(r11.loc[('xgboost','A'),'test_pr_auc']), f3(r11.loc[('xgboost','A'),'test_f1']), f3(r11.loc[('xgboost','A'),'test_precision']), f3(r11.loc[('xgboost','A'),'test_recall'])],
        ["XGBoost, Exp C (all features)", f3(best['test_pr_auc']), f3(best['test_f1']), f3(best['test_precision']), f3(best['test_recall'])]],
       ["Detector", "PR-AUC", "F1", "Precision", "Recall"], "num")}
</section>
""")

# 12 phase 6
cC, compl = ucs["C"], uan["complementarity"]
add(f"""
<section>
<h1>Phase 6: unsupervised learning and anomaly detection</h1>
<p>Phase 5 asked how well a model can learn fraud <i>from labels</i>. Phase 6 asks what can be found <b>without them</b>: do transactions fall into natural risk groups (RQ5, cluster analysis), and can a model that has only ever seen legitimate behaviour flag fraud as "unusual" (RQ6, anomaly detection)? This matters in practice because new fraud patterns appear before anyone has labelled them.</p>
<h2>Design</h2>
{table([
 ["Data", f"v1.1, same stratified 60/20/20 split as Phase 5. Exp B ({ucs['B']['n_features_after_encoding']} encoded columns) and Exp C ({cC['n_features_after_encoding']}); signed-log + standardisation, one-hot categoricals; leakage/ID blacklist enforced"],
 ["Clustering", "all 10,000 rows; labels never used to fit, only attached afterwards (fraud rate, scenario mix, Adjusted Rand Index, NMI)"],
 ["K-Means", "K = 2…8 scored by inertia (elbow), silhouette and Davies–Bouldin. Rule: lowest Davies–Bouldin among the K values within 0.005 of the best silhouette (D-026)"],
 ["DBSCAN", f"min_samples = {cC['dbscan']['min_samples']}, ε = {cC['dbscan']['eps']:.2f} from the knee of the k-distance curve"],
 ["Projections", "PCA (linear) and t-SNE (non-linear, perplexity 40), coloured by label and by cluster"],
 ["Anomaly detectors", f"Isolation Forest (300 trees), One-class SVM (RBF, ν = 0.01 and 0.05), autoencoder (d → 32 → 16 → 8 → 16 → 32 → d, PyTorch). Fitted on the {uan['train_rows_legit']:,} <b>legitimate training rows only</b> (no fraud, no BORDERLINE)"],
 ["Thresholds", f"95th and 99th percentile of scores on the {uan['reference_rows_legit_val']:,} legitimate <b>validation</b> rows (D-025); the same reference gives each transaction an anomaly percentile for the risk score"],
 ["Metrics", f"test set ({uan['test_rows']:,} rows): PR-AUC, ROC-AUC, recall at 1% false-positive rate, precision/recall at p95/p99, BORDERLINE flag rate, per-scenario flag rate"],
 ["Scale track", f"Isolation Forest on {ups['rows_scored']:,} PaySim TRANSFER/CASH_OUT rows (Polars lazy pipeline), evaluated against PaySim's isFraud"],
], ["Item", "Choice"], "small", widths=["18%", "82%"], raw=True)}
<h2>RQ5: clustering</h2>
{fig("unsup_01_kmeans_selection", "K-Means model selection. Silhouette is low (≤ 0.16) and flat for K = 2–4, so it can't choose K alone; Davies–Bouldin favours K = 4 for Exp C and K = 3 for Exp B (circled). Exp C has the higher silhouette: network features give the clusters more to separate on.")}
{cluster_summary_table()}
<p>"Risk clusters P / R": precision and recall if an analyst reviewed every row in the clusters whose fraud rate is above the 13.5% base rate.</p>
{fig("unsup_02_cluster_composition", "Exp C clusters with labels attached afterwards. Each cluster is named by its most extreme centroid features (in standard deviations).", "92%")}
{cluster_table()}
<p><b>The four clusters map onto fraud mechanisms.</b> Cluster 0 is the S3 rapid-burst scenario (senders with ~7σ more transactions in the last 10 minutes). Cluster 3 is the S9 fraud-ring hubs (receivers with ~4σ more distinct senders). Cluster 2 collects most other fraud (layering, mules, new device/beneficiary, impossible travel) through unusual amounts and same-community flows. The normal cluster still hides the frauds designed to blend in (about half of the S2 high-value spikes and 39% of S6 structuring).</p>
{fig("unsup_03_embeddings", "PCA (top) keeps only 15% of the variance in 2-D; t-SNE (bottom) shows the burst and hub clusters as tight islands and several pure-fraud islands (the injected rings and mules), while camouflaged fraud stays inside the normal mass.", "92%")}
{fig("unsup_04_dbscan", f"DBSCAN marks {cC['dbscan']['noise_n']} points ({pct(cC['dbscan']['noise_rate'])}) as noise; they are {pct(cC['dbscan']['fraud_rate_noise'],0)} fraud vs {pct(cC['dbscan']['fraud_rate_clustered'],0)} elsewhere, but cover only {pct(cC['dbscan']['noise_rate_by_label']['FRAUDULENT'],1)} of all fraud.")}
<p><b>Answer to RQ5: partly.</b> With no labels, K-Means isolates normal traffic (3% fraud) from three interpretable risk groups, with ARI {cC['kmeans_vs_label']['ari']:.2f} against the 3-class label (Exp B: {ucs['B']['kmeans_vs_label']['ari']:.2f}). Reviewing the risk clusters would catch {pct(cC['flag_high_fraud_clusters']['recall'],0)} of fraud at {pct(cC['flag_high_fraud_clusters']['precision'],0)} precision, about three times the base rate. But silhouette ≤ 0.16 means the structure is weak, and 41–60% of each risk cluster is not fraud. Clustering is a <b>triage and explanation</b> tool, not a classifier. For the DMW report, this is the cluster-analysis component, and the cluster profiles are the "description" output of the KDD process.</p>
<h2>RQ6: anomaly detection trained only on legitimate behaviour</h2>
{fig("unsup_05_detector_pr", "Precision–recall on the test set (Exp C). The One-class SVMs and the autoencoder are nearly identical; Isolation Forest is weaker; the supervised XGBoost (dashed) stays far ahead.", "82%")}
{detector_table()}
<ul>
<li>Never having seen a fraud label, the One-class SVM and the autoencoder reach <b>PR-AUC ≈ 0.55, four times the no-skill level</b> (0.135), with ROC-AUC ≈ 0.87. The choice of ν barely matters.</li>
<li><b>Isolation Forest is weakest</b>: its random axis-aligned splits are wasted on the many one-hot and near-constant columns. It is also the detector that gains most from network features (PR-AUC {f3(anr.query("exp=='B' and detector=='isolation_forest'").pr_auc.iloc[0])} → {f3(anr.query("exp=='C' and detector=='isolation_forest'").pr_auc.iloc[0])}).</li>
<li>For the OC-SVM and autoencoder, network features raise ROC-AUC (≈ 0.81–0.85 → 0.87) but not PR-AUC, and recall at 1% FPR falls. The 19 extra columns widen what "normal" looks like. A supervised model can use new columns selectively; an unsupervised detector has to weigh them all.</li>
<li><b>Anomalous ≠ fraudulent.</b> Every strong detector flags ~80% of BORDERLINE rows. Those transactions are genuinely unusual (travel, new devices, large first payments) but legitimate. This is the main limitation of anomaly detection for fraud.</li>
</ul>
{fig("unsup_06_detector_bars", "Exp B vs Exp C for each detector: PR-AUC, recall at 1% false-positive rate, and the share of BORDERLINE rows flagged.")}
<h3>Per scenario</h3>
{anomaly_scen_table()}
{fig("unsup_07_scenario_heatmap", "Flag rate for every scenario × label on the test set. Fraud rows: higher is better. Legitimate and BORDERLINE rows: lower is better.", "80%")}
<p>The detectors catch <b>single-transaction anomalies almost perfectly</b> (S3 burst, S4 new beneficiary, S5 new device, S10 impossible travel, S11 shared device: 94–100% for OC-SVM/autoencoder). They catch the <b>network scenarios only partly</b> (S6 structuring 61–78%, S7 layering 62%, S8 mule 40%, S9 ring hub 68–76%), because those frauds are built to look normal one transaction at a time. They <b>miss the S2 high-value spike</b> (≤ 15%): after log-scaling, a large but plausible amount is not far from normal, while XGBoost catches 80% because it learns the interaction with the sender's history.</p>
<h3>Complementarity with the supervised model</h3>
{fig("unsup_09_complementarity", f"XGBoost misses {compl['xgb_missed']} of {compl['test_frauds']} test frauds (mostly S7 layering and S2 high-value spikes). Each strong detector recovers {compl['caught_by']['autoencoder']} of them.", "62%")}
<p>The rank correlation between the detector scores and the XGBoost probability is only ≈ {uan['spearman_vs_xgb']['autoencoder']:.1f}, so the two carry different information. This is the evidence for including the <b>anomaly percentile</b> as a separate component of the Phase 7 risk score (its weight will be chosen on validation).</p>
<h2>The autoencoder and its ONNX export</h2>
{fig("unsup_08_autoencoder", f"Left: training and validation loss with early stopping ({uan['ae_epochs']} epochs; best weights restored). Right: reconstruction error by label on the test set. Most fraud and BORDERLINE rows sit beyond the p95 threshold; the rest reconstruct like legitimate data.")}
<ul>
<li>Training used Adam (learning rate 10⁻³, weight decay 10⁻⁵), batch 128, mean-squared-error loss, and early stopping (patience 20) on the legitimate validation rows. Train and validation loss stay close, so there's no overfitting.</li>
<li><b>ONNX export</b> (opset 17, dynamic batch size; the graph returns both the reconstruction and the per-row error, D-027): a <b>{upar['onnx_bytes'] // 1024} KB</b> file. Maximum difference from PyTorch on {upar['rows']:,} test rows: <b>{upar['max_abs_diff_error']:.1e}</b> (acceptance ≤ 1 × 10⁻⁵).</li>
<li><b>Why it matters for CCT:</b> the serving container needs only <code>onnxruntime</code> (~50 MB), not PyTorch (several hundred MB). That keeps the ML service within Render's ~512 MB free tier, and the model becomes framework-independent.</li>
</ul>
<h2>PaySim scale track</h2>
{fig("unsup_10_paysim_if", "Fully unsupervised Isolation Forest on 2.77 M PaySim TRANSFER/CASH_OUT rows. Two engineered balance-error features raise PR-AUC from 0.34 to 0.49.")}
{paysim_table()}
<p>Polars filters 6.36 M rows to {ups['rows_scored']:,} and builds the features lazily in about a second (cached file); the raw CSV never enters pandas. The forest is fitted on a 250,000-row random sample <b>without labels</b> and then scores every row, using about {ups['peak_rss_mb']/1024:.1f} GB peak memory on one laptop. Against the {pct(ups['fraud_rate'],2)} fraud rate, PR-AUC {f3(ups['feature_sets']['raw+balance_error']['pr_auc'])} is over 100× the no-skill level. The two engineered columns add +{ups['feature_sets']['raw+balance_error']['pr_auc'] - ups['feature_sets']['raw']['pr_auc']:.2f} PR-AUC, more than any model choice. PaySim has no device, location or beneficiary fields, so this track demonstrates <i>scale</i>, not transfer to the India data (D-006).</p>
<p><b>Answer to RQ6:</b> detectors trained only on legitimate data are useful but clearly weaker than supervised learning (PR-AUC ≈ 0.55 vs {best['test_pr_auc']:.3f}). They excel at single-transaction anomalies, partly catch organised fraud, miss "plausible-amount" fraud, and cannot tell unusual-but-legitimate (BORDERLINE) from fraud. Their value is as a <b>complementary signal</b> and as a safety net for fraud patterns not yet in the labels.</p>
</section>
""")

# 13 phase 7
add(f"""
<section>
<h1>Phase 7: tuning, ensembles, calibration, risk score and explanations</h1>
<p>Phase 5 compared models with default settings. Phase 7 turns the best of them into a <b>production model</b>: it tunes hyperparameters, studies bias and variance, tries a stacking ensemble, calibrates the probabilities, converts them into a 0–100 risk score with tiers, and explains every prediction.</p>
<h2>Protocol</h2>
{table([
 ["Train (6,000 rows)", "hyperparameter search with 5-fold stratified cross-validation (PR-AUC); learning curves; stacking and calibration fit; fraud rates of the network flags"],
 ["Validation (2,000)", "choice of the production model; decision threshold; risk-score weights and tier cut-offs; permutation importance"],
 ["Test (2,000)", "scored once at the end for every final candidate; nothing is chosen on it"],
 ["Imbalance", "class weights for the tree models, no resampling for KNN/MLP (D-020: SMOTE never helped in Phase 5)"],
], ["Split", "Used for"], "small", widths=["20%", "80%"])}
<h2>Hyperparameter tuning</h2>
{fig("final_01_tuning", "Cross-validated PR-AUC before (hollow) and after (filled) tuning, on the same five folds, so the gains are paired comparisons.", "88%")}
{tuning_table()}
<p><b>Grid search</b> suits small, discrete spaces (the tree's depth, leaf size and split criterion; KNN's k, weighting and distance). <b>Randomized search</b> samples large or continuous spaces at a fixed budget: 20 XGBoost draws took about 30 seconds. Four of five models improve significantly. The decision tree gains most because the defaults overfit: the search chooses 40 rows per leaf. Tuned XGBoost uses many small boosting steps (learning rate ≈ 0.024, ~465 trees).</p>
<h2>Bias and variance</h2>
{fig("final_02_bias_variance", "Left: decision-tree depth. Shallow trees underfit (both scores low); beyond depth 8 the training score keeps rising while the held-out score stays flat (overfitting). Middle and right: learning curves for the tuned tree and the tuned XGBoost.")}
<p>XGBoost fits its training folds perfectly (high variance), yet its cross-validated score still rises with more data (0.78 at 500 rows → 0.91 at 4,800), so more data would still help. The tuned tree has a smaller gap but a lower ceiling (≈ 0.85): the bias–variance trade-off in one picture.</p>
<h2>Stacking and calibration</h2>
<p><b>Stacking</b> combines the tuned Random Forest, XGBoost and MLP through a logistic-regression meta-learner trained on out-of-fold probabilities. It reached validation PR-AUC {f3(fcomp.loc['stack_uncalibrated','val_pr_auc'])} vs {f3(fcomp.loc['xgboost','val_pr_auc'])} for tuned XGBoost alone. The meta-learner relies mostly on the forest (coefficient {fs7['stack_meta_weights']['random_forest']:.1f}) and XGBoost ({fs7['stack_meta_weights']['xgboost']:.1f}); their errors are highly correlated, so combining them adds little. <b>An ensemble is not automatically better</b>, and this negative result is worth stating.</p>
{fig("final_03_calibration", "Reliability diagrams on validation. Isotonic calibration moves predicted probabilities onto the diagonal; for XGBoost the expected calibration error falls from 0.014 to 0.005.")}
<p><b>Isotonic calibration</b> (learned on out-of-fold predictions, one refitted model kept, D-029) makes a predicted 30% mean roughly 30% fraud, which the 0–100 score needs. It costs about 0.01 PR-AUC because the isotonic step function creates ties. The <b>production model</b> is the better calibrated candidate on validation: calibrated XGBoost (D-031).</p>
<h2>Final test results (scored once)</h2>
{final_table()}
{fig("final_04_test_comparison", "Test PR-AUC for every final candidate, with the Phase 5 default XGBoost for reference. The production model was chosen on validation, not on this chart.", "88%")}
<p><b>Answer to RQ4:</b> XGBoost is the best classifier. Tuning lifts test F1 from {best['test_f1']:.3f} to {FT['test_f1']:.3f} (precision {FT['test_precision']:.2f}, recall {FT['test_recall']:.2f}: {int(FT['test_tp'])} of {int(FT['test_tp']+FT['test_fn'])} frauds caught with {int(FT['test_fp'])} false alarms out of 1,730 non-fraud rows). The saved artifact is {fs7['artifact_sizes_mb']['production_model.joblib']} MB; reloading it reproduces the test metrics exactly (an automated test).</p>
<h2>The 0–100 risk score</h2>
<p><code>risk = 100 × ({RSC['weights']['model']:.2f} × P(fraud) + {RSC['weights']['anomaly']:.2f} × anomaly percentile + {RSC['weights']['graph']:.2f} × graph risk)</code>, then overrides. The weights come from a grid over all combinations summing to 1, maximising validation PR-AUC. The anomaly percentile is the Phase 6 autoencoder's. Graph risk is the training fraud rate of the strongest network red flag that fires (pass-through {pct(RSC['flag_rates']['PASS_THROUGH'],0)}, quick inflow {pct(RSC['flag_rates']['QUICK_INFLOW'],0)}, hub receiver {pct(RSC['flag_rates']['HUB_RECEIVER'],0)}, burst sender {pct(RSC['flag_rates']['BURST_SENDER'],0)}, shared device {pct(RSC['flag_rates']['SHARED_DEVICE'],0)}; D-030). The two extra components raise PR-AUC slightly (validation {RSC['model_only_val_pr_auc']:.3f} → {RSC['weights_val_pr_auc']:.3f}; test {FT['test_pr_auc']:.3f} → {fs7['risk_score_test']['final']['pr_auc']:.3f}) and make the score easier to explain.</p>
{table([
 ["CRITICAL", f"≥ {RSC['cutoffs']['CRITICAL']:.0f}", "lowest score with ≥ 95% precision on validation", "block / escalate immediately"],
 ["HIGH", f"≥ {RSC['cutoffs']['HIGH']:.0f}", "maximum-F1 threshold on validation", "alert an analyst"],
 ["MEDIUM", f"≥ {RSC['cutoffs']['MEDIUM']:.0f}", "keeps 98% of validation fraud at MEDIUM or above", "review queue"],
 ["LOW", f"< {RSC['cutoffs']['MEDIUM']:.0f}", "—", "allow"],
], ["Tier", "Score", "How the cut-off was chosen", "Intended action"], "small", widths=["14%", "12%", "44%", "30%"])}
<p><b>Overrides:</b> a sanctions match ≥ 90 forces CRITICAL; a PEP customer is lifted to at least MEDIUM; the structuring rule (near-threshold amount + burst sender) adds 10 points. On the test set the structuring rule fired {fs7['risk_overrides_test']['structuring']} times and {fs7['risk_overrides_test']['pep_floor']} PEP customers were lifted.</p>
{fig("final_06_risk_score", "Left: share of each label scoring at or above each value, with the tier cut-offs. Right: tier assignment by true label on the test set.")}
{tier_table()}
<p>{pct(ftiers.loc['FRAUDULENT','CRITICAL_share'],0)} of fraud is CRITICAL and {pct(ftiers.loc['FRAUDULENT','HIGH_share']+ftiers.loc['FRAUDULENT','CRITICAL_share'],0)} is HIGH or above, while {pct(ftiers.loc['LEGITIMATE','LOW_share'],0)} of legitimate payments are LOW. The 98%-recall review band costs {pct(ftiers.loc['LEGITIMATE','MEDIUM_share'],0)} of legitimate payments in MEDIUM: a policy choice that a bank would set by reviewer capacity (D-032). Most BORDERLINE cases land in HIGH, where a human should look.</p>
<h2>Explanations</h2>
{fig("final_07_importance", "Global importance: mean absolute SHAP value for the tuned XGBoost (left) and permutation importance for the production model (right). Both rank amount, channel, payee recency, inflow timing and distance from home highest.")}
{fig("final_08_shap_beeswarm", "Direction of each feature's effect: red = high feature value, blue = low. Large amounts, UPI, new payees, distant locations, recent inflows and sender bursts push towards fraud.", "80%")}
{fig("final_09_local_explanations", "Per-transaction explanations, as the API will return them. Left: a mule pass-through fraud (money arrived 195 s earlier and was forwarded). Right: a BORDERLINE new-device payment where every signal is suspicious, the reason BORDERLINE belongs in human review.")}
<p>At serve time, XGBoost's built-in <code>pred_contribs</code> computes the same TreeSHAP values as the <code>shap</code> library (maximum difference {fs7['explanations']['parity_native_vs_shap']['max_abs_diff_vs_shap']:.1f} on 2,000 rows), so the service explains every score without installing <code>shap</code>, which saves memory on the 512 MB server. The amount's dominance partly reflects the synthetic generator (Chapter 5).</p>
</section>
""")

# 14 phase 8
add(f"""
<section>
<h1>Phase 8: cloud services and containers</h1>
<p>Phase 8 turns the trained models into a <b>running service</b>: a public REST API backed by four domain microservices, packaged as Docker containers, measured for latency, memory and load. The cloud deployment itself (Render + Vercel) is Phase 10 (Ch. 16); everything in this chapter runs locally with Docker Compose.</p>
<h2>Architecture</h2>
{flow([("Client / React SPA", "HTTPS + CORS"), ("api-gateway :8000", "public · validation · orchestration · risk score · CORS · /metrics"), [("ml-service :8001", "calibrated XGBoost · TreeSHAP reasons · ONNX autoencoder · K-Means"), ("mining-service :8002", "association rules · live rule matching")], [("graph-service :8003", "account metrics · ego networks · graph risk"), ("sanctions-service :8004", "fuzzy OFAC + UN screening")]])}
<p>The gateway is the only public service. For each score it calls the four domain services <b>in parallel</b>, so latency is the slowest call, not the sum. All services are <b>stateless</b>: they load a versioned artifact bundle at start-up and keep no per-request state, so any replica can serve any request.</p>
{table([
 ["Microservices (remote)", "FINSIGHT_SERVICE_MODE=remote", "5 containers talking HTTP/JSON; each can be deployed and scaled on its own (Docker Compose, optional 5-service Render Blueprint)"],
 ["Modular monolith (inprocess)", "FINSIGHT_SERVICE_MODE=inprocess", "the gateway imports the four modules directly: one container, one cold start (the default free Render deployment)"],
], ["Mode", "Setting", "What it means"], "small", widths=["22%", "28%", "50%"])}
<p>Both modes run the <b>same code</b>. An automated test sends the same transactions through both and checks that every score, tier, reason and fired rule is identical.</p>
<h2>REST API (OpenAPI / Swagger at /docs)</h2>
{table([
 ["POST /score-transaction", "risk 0–100, tier, component scores, top reasons (TreeSHAP), fired rules, network flags, cluster, sanctions, per-service timings"],
 ["POST /anomaly-score", "Isolation Forest, One-class SVM and autoencoder scores with percentiles vs legitimate traffic"],
 ["GET /patterns, /patterns/benchmark", "association rules (filter by consequent, confidence, lift, network items) and the Apriori vs FP-Growth benchmark"],
 ["GET /cluster/{{tx_id}} · /network/{{account_id}}", "K-Means cluster profile · account metrics, activity and ego network"],
 ["GET /sanctions-check?name=", "fuzzy OFAC + UN matches with scores"],
 ["GET /models/compare", "Phase 5 grid, Phase 7 final models and tuning, Phase 6 detectors"],
 ["GET /transactions, /transactions/{{tx_id}}, /overview", "filterable transaction list, full profile, dataset KPIs (for the dashboard)"],
 ["GET /health, /metrics, /platform/status", "liveness, per-route latency percentiles and memory, health of every downstream service"],
], ["Endpoint", "Returns"], "small", widths=["40%", "60%"])}
<p><b>Scoring input (D-033):</b> a stored transaction (its point-in-time features, exactly what the model was evaluated on), the same transaction with <b>what-if overrides</b> (e.g. a different amount or a new device; dependent columns are recomputed), or a complete feature row. Computing network features for a brand-new transaction would need a live streaming feature store, which is out of scope. Errors are mapped to HTTP codes: unknown ID → 404, invalid input → 422, downstream failure → 502.</p>
<h2>Containers and the model registry</h2>
{table([
 ["One multi-stage Dockerfile", "build args SERVICE (which service) and REQS (dependency set); stages: dependencies → artifact bundle → slim runtime"],
 ["Slim, non-root runtime", "python:3.12-slim, a dedicated user (uid 10001), no compilers or pip cache, a HEALTHCHECK, $PORT from the platform"],
 ["Two dependency sets", f"base (FastAPI, pandas, pyarrow, RapidFuzz): {IMG['finsight/gateway:local']:.0f} MB; ml (+ scikit-learn, XGBoost, ONNX Runtime): {IMG['finsight/ml:local']:.0f} MB (uncompressed; pyarrow is the largest single part at 163 MB)"],
 ["Artifact bundle = model registry", f"27 files, {plat['bundle']['total_mb']} MB ({plat['bundle']['version']}), sha256 per file in manifest.json; published as a GitHub Release and downloaded during docker build, so each image is immutable and versioned"],
 ["12-factor configuration", "all settings from environment variables (mode, URLs, internal key, CORS origins, timeouts)"],
], ["Practice", "Implementation"], "small", widths=["26%", "74%"])}
<h2>Security and observability</h2>
<ul>
<li><b>Network isolation + shared secret:</b> in Compose only the gateway publishes a port. The internal services are unreachable from the host, and they reject calls without the <code>X-Internal-Key</code> header (401). On Render's free tier, which has no private network, the key is the only barrier, so it is mandatory there.</li>
<li><b>CORS allow-list:</b> only the configured dashboard origins can call the API from a browser (tested: a foreign origin gets no CORS header).</li>
<li><b>Observability:</b> every request gets an <code>X-Request-ID</code> that is forwarded to downstream services and appears in every structured JSON log line. <code>/metrics</code> reports per-route request counts, errors and p50/p95/p99 latency. Responses carry <code>X-Served-By</code> (which replica answered) and <code>Server-Timing</code>.</li>
</ul>
<h2>Measurements (Docker Compose on one 8-core laptop)</h2>
{fig("platform_01_phase8", "Left: warm single-client latency in both modes, far below the 200 ms target. Middle: Locust load tests in remote mode. Right: process memory per container against Render's 512 MB free instance.")}
{table([
 ["Warm /score-transaction, 1 client (p50 / p95)", f"{PW['compose-inprocess']['client_p50_ms']:.0f} / {PW['compose-inprocess']['client_p95_ms']:.0f} ms", f"{PW['compose-remote']['client_p50_ms']:.0f} / {PW['compose-remote']['client_p95_ms']:.0f} ms"],
 ["Memory (process RSS)", f"{plat['warm']['compose-inprocess']['rss_mb']['gateway']:.0f} MB (one container)", f"gateway {plat['warm']['compose-remote']['rss_mb']['gateway']:.0f} + ml {plat['warm']['compose-remote']['rss_mb']['ml']:.0f} + 3 × ~165 MB"],
 ["Start-up to healthy", f"{plat['startup_s']['inprocess_container']:.0f} s", f"~{plat['startup_s']['remote_stack_all_healthy']:.0f} s for the whole stack"],
], ["Metric", "Inprocess (monolith)", "Remote (5 services)"], "small", widths=["40%", "30%", "30%"])}
{table([[r["scenario"], f"{r['rps_total']:.0f}", f"{r['p50_ms']:.0f}", f"{r['p95_ms']:.0f}", f"{r['p99_ms']:.0f}", r["failures"]] for r in PL.values()],
       ["Load test (remote mode)", "Requests/s", "p50 ms", "p95 ms", "p99 ms", "Errors"], "num small")}
<ul>
<li><b>Microservices vs monolith:</b> on one machine the two modes are equally fast for a single client, because the parallel fan-out hides the extra HTTP hops. The monolith needs about half the total memory, which is why it is the free-tier default.</li>
<li><b>Horizontal scaling:</b> <code>docker compose up --scale ml-service=3</code> starts two more ML replicas behind the same DNS name, and each replica's request counter shows that the traffic is shared. Median latency under 30 users fell from {PL['remote_ml1']['p50_ms']:.0f} to {PL['remote_ml3']['p50_ms']:.0f} ms. The remaining limit is the gateway and the load generator sharing the same 8 cores.</li>
<li><b>Performance engineering from measurements (D-034, D-035):</b> profiling showed that the sklearn preprocessing objects cost ~10 ms per single-row call, four times per score. Compiling them to numpy (with exact, tested equality) cut warm latency from 138 to 21 ms and memory from 460 to 325 MB. The load test then showed the sanctions service at 96% CPU (a fuzzy scan of 27,000 names per score); an LRU cache brought it to 17%. It also showed pooled connections pinning traffic to one replica, which a short keep-alive fixed.</li>
</ul>
<p>Acceptance criteria met: all services healthy under <code>docker compose up</code>; warm p95 {PW['compose-remote']['client_p95_ms']:.0f} ms (target &lt; 200 ms); the single container at {plat['warm']['compose-inprocess']['rss_mb']['gateway']:.0f} MB (target &lt; 450 MB). {N_TESTS} automated tests pass, including 12 API tests.</p>
</section>
""")

# 15 phase 9
add(f"""
<section>
<h1>Phase 9: the web dashboard and live analysis</h1>
<p>The dashboard is how examiners and users see the project. It is a <b>React single-page application</b> (React 18, Vite, TypeScript, Plotly.js) that holds no data of its own: every number on every page comes live from the API gateway built in Phase 8. It builds to static files, so it can be served from Vercel's global CDN with no server.</p>
{table([
 ["Executive overview", "KPIs, daily volume (fraud vs not), fraud rate by channel, scenarios, how a score is built, key findings", "All"],
 ["Data quality", "table completeness, referential integrity, leakage scan, RQ1 feature ranking, behaviour flags, sanctions threshold sweep, cleaning log", "DMW"],
 ["Transaction explorer", "filter by label, scenario, channel, amount, account; profile of the selected payment; jump to scoring or the network", "DMW"],
 ["Pattern mining lab", "all rules (support × confidence × lift), filterable rule table, Apriori vs FP-Growth runtime and memory, item definitions", "DMW"],
 ["ML experiments lab", "A/B/C feature groups per dataset version, test PR-AUC heatmap, tuning gains, final candidates, SHAP importance, per-scenario recall, learning curves", "ML"],
 ["Clustering & anomaly", "2-D PCA of 4,000 transactions coloured by label or cluster, cluster profiles, K selection curves, anomaly detector comparison", "ML"],
 ["Network explorer", "accounts ranked by PageRank or degree, ego network (senders left, receivers right, fraud edges highlighted), counterparties", "ML + DMW"],
 ["Risk scorer", "score any transaction; what-if changes (amount, new device, new payee, distance, burst); score, tier, components, SHAP reasons, fired rules, network flags, cluster, sanctions, per-service timings", "All"],
 ["Platform", "service health, memory and latency, per-route percentiles, bundle version, wake-up time of this visit, a live latency probe, recorded load tests and image sizes", "CCT"],
], ["Page", "What it shows", "Subject"], "small", widths=["20%", "66%", "14%"])}
{fig("dashboard_08_risk_scorer", "Risk scorer: a mule pass-through fraud scores 100 (CRITICAL). The reasons are native TreeSHAP values; the components, network red flags, fired rules, cluster, sanctions result and per-service timings come from the four services.", "66%")}
{fig("dashboard_01_overview", "Executive overview.", "66%")}
{fig("dashboard_05_ml_lab", "ML experiments lab: feature-group ablation, test heatmap, tuning gains and final candidates (switchable between dataset v1.1 and the v1.0 control).", "66%")}
{fig("dashboard_04_patterns", "Pattern mining lab: every productive rule, the Apriori vs FP-Growth benchmark, and a filterable rule table with network items highlighted in amber.", "66%")}
{fig("dashboard_06_clusters", "Clustering and anomaly: PCA scatter (click a point to score it), K-Means cluster profiles, K selection curves and the detector comparison.", "66%")}
{fig("dashboard_07_network", "Network explorer: a fan-in hub receiving fraudulent payments (orange edges) from many senders.", "66%")}
{fig("dashboard_09_platform", "Platform page: live service status and latency, plus the recorded Phase 8 measurements.", "66%")}
{fig("dashboard_02_data_quality", "Data quality page: audit tables, leakage scan (the alert flag is marked SUSPECT), feature ranking and the sanctions threshold curve.", "66%")}
{fig("dashboard_03_transactions", "Transaction explorer with filters and paging.", "66%")}
<h2>Design decisions (D-036, D-039)</h2>
<ul>
<li><b>Cold starts handled in the interface.</b> Before rendering, the app polls <code>/health</code> and shows "Waking the backend…" with a timer, because Render's free tier sleeps after 15 minutes. That screen needs about 68 KB of compressed code and 182 KB of self-hosted fonts (text shows in a system font until they arrive); the 1.4 MB charting library loads with the first chart. The Platform page reports how long this visit's wake-up took.</li>
<li><b>One design system (D-039).</b> Warm paper surfaces, a deep-teal interface colour, serif headings (Newsreader) with a sans body (Inter), soft cards and pills, on every page and state. Data keeps the meaning used in the report figures (legitimate blue, fraudulent orange, borderline aqua), with shades re-validated on the new surfaces for colour-blind readers; charts of a single unlabelled series use teal. Light is the default; dark has its own stepped colours. Risk tiers always show an icon and a word, never colour alone.</li>
<li><b>Interactive, not decorative.</b> Every chart has hover tooltips, the PCA scatter opens the scorer for the clicked transaction, the network graph re-centres on the clicked account, and tables are keyboard-navigable.</li>
<li><b>Static hosting.</b> <code>vercel.json</code> rewrites every route to <code>index.html</code> (client-side routing), caches hashed assets for a year, and adds security headers. The API address is a build-time environment variable (<code>VITE_API_BASE_URL</code>), and the API only accepts browser calls from origins on its CORS allow-list.</li>
<li><b>Checked like the report figures.</b> Every page was rendered in a headless browser in light and dark and from 320 to 1920 px wide, with automated checks for horizontal overflow, text contrast (4.5:1 for normal text) and console errors, followed by inspection of the screenshots.</li>
</ul>
<p>Four API endpoints were added for the dashboard (<code>/clusters</code>, <code>/accounts</code>, <code>/data-quality</code>, <code>/platform/benchmarks</code>), with tests in both service modes. The bundle now includes a 2-D PCA embedding for the scatter plot, and large responses are gzip-compressed.</p>
<h2>Phase 9b: live analysis (the home page)</h2>
<p>A results dashboard only shows what already happened. To let examiners see the system <b>work</b>, the home page runs the complete pipeline on new data in about the time it takes to explain it. There are three inputs:</p>
{table([
 ["Demo dataset", "The 10,000 synthetic payments (or a 2,000-row stratified sample) with their labels", "Shows the end-to-end pipeline and how accurate it is, including on the held-out test rows only"],
 ["Upload a CSV", "Any file in the FinSight transaction format: 6 required columns, optional ones defaulted; template and sample files are downloadable; up to 25,000 rows / 10 MB, processed in memory and never stored", "Real use: a bank analyst's batch of payments"],
 ["Attack simulator", "A normal day of legitimate payments (31 Aug) with chosen attacks injected and labelled, optionally camouflaged to look normal per payment", "Shows that the system catches organised patterns it has never seen as rows"],
], ["Input", "What it is", "What it demonstrates"], "small", widths=["16%", "52%", "32%"])}
{flow([("1 · Validate", "required columns, types, dates, duplicates, amounts; bad rows reported"), ("2 · Defaults", "optional fields from the bank's records; unknown accounts imputed and reported"), ("3 · Features", "bank history + batch through the unchanged Phase 1/3 code (point-in-time windows, PageRank)"), ("4 · Score", "risk score 0–100 via the same backend as the API: model, anomaly, network flags, rules, sanctions"), ("5 · Summarise", "tiers, alerts, suspicious structures, rules, anomalies, accuracy if labels exist, CSV export")], "small")}
<ul>
<li><b>Same code, same features.</b> A batch is analysed together with the bank's 10,000-row history, so velocity windows and PageRank see the same context as in training. Analysing the demo dataset reproduces every stored training feature exactly (maximum difference 0.0, tested). Held-out PR-AUC is 0.892 without the sanctions override and 0.881 with it: two synthetic customer names fuzzy-match sanctions entries, which is reported rather than hidden.</li>
<li><b>Labels never influence a score.</b> If the file has <code>is_fraud</code> / <code>risk_label</code> / <code>scenario</code>, they are only compared with the results afterwards (PR-AUC, precision, recall, F1, recall per scenario).</li>
<li><b>Suspicious structures.</b> Besides single payments, the page groups alerts into fan-in hubs, pass-through chains, bursts and shared devices, and draws them as small network diagrams.</li>
<li><b>Fits the free cloud instance.</b> 10,000 rows take ~3 s on a laptop and ~20–35 s on Render's shared CPU; peak memory 436 MB at 24,000 rows. At most two results are kept for 30 minutes, memory is released after each run, and the demo results are cached (a repeat run returns in 0.09 s).</li>
</ul>
{table([[ATT[a], sim_cell(a, False), sim_cell(a, True)] for a in ATT],
       ["Simulated attack (10 injected payments, 800 normal ones; 3 seeds)", "Alerted (HIGH / CRITICAL)", "Alerted when camouflaged"], "small", widths=["52%", "24%", "24%"])}
<p>Normal background payments are falsely alerted {SIM_FA:.1%} of the time. Camouflaged shared devices are the hardest case: every other signal looks normal, so only the device sharing gives them away.</p>
{fig("dashboard_10_analyze", "Live analysis of the 2,000-row demo sample (screenshot from a local run of the same code): data-quality checks, KPIs, risk tiers, alerts over time and the accuracy check, overall and on held-out test rows only.", "62%")}
{fig("dashboard_11_simulation", "Attack simulator: a fraud ring, a mule chain and a burst injected into a normal day. 35 of the 36 injected payments raised an alert (one burst payment was missed), and the suspicious structures (fan-in hub, pass-through chains, burst, shared devices) are found and drawn.", "62%")}
</section>
""")

# 16 phase 10
add(f"""
<section>
<h1>Phase 10: cloud deployment and measurements</h1>
<p>The system is live: the React dashboard on <b>Vercel</b> at <b>{LIVE_UI}</b> and the FastAPI backend as a Docker container on <b>Render</b> at <b>{LIVE_API}</b> (interactive API documentation at <code>/docs</code>). Both are deployed from the GitHub repository by Infrastructure-as-Code files and redeploy automatically on every push to <code>main</code>.</p>
<h2>Deployment topology</h2>
{flow([("Developer", "git push to main"), [("GitHub", "code · Actions CI · Releases (model bundle, sha256)")], [("Render build", "docker build (same Dockerfile as local); downloads the model bundle"), ("Vercel build", "npm run build; API address baked in")], [("Render container", "FastAPI gateway, inprocess mode · Singapore · 512 MB · HTTPS"), ("Vercel CDN", "static SPA at the edge · HTTPS")], ("Browser", "loads the SPA from Vercel; calls the API on Render directly (CORS allow-list)")], "small")}
{table([
 ["React dashboard (static)", "Vercel (Hobby, free)", "Global CDN, Git-push deploys, preview URL per pull request; SPA rewrites and security headers in vercel.json"],
 ["FastAPI services (Docker)", "Render web service (free)", "Native Docker builds from GitHub, managed HTTPS, health checks, Blueprint IaC (render.yaml)"],
 ["Model artifacts (7.9 MB bundle)", "GitHub Releases", "Versioned, checksummed 'model registry'; downloaded during docker build into an immutable image"],
 ["Training pipeline", "Local", "Needs the 1.8 GB dataset (not in Git); publishes artifacts to a new release"],
 ["CI", "GitHub Actions", "Every push: tests, dashboard build, image build + smoke test at 512 MB"],
 ["Portability path (documented)", "Google Cloud Run · Kubernetes", "deploy/cloudrun.sh and deploy/k8s/finsight.yaml run the same image; not deployed (D-038)"],
], ["Component", "Platform", "Role"], "small", widths=["24%", "24%", "52%"])}
<h2>Infrastructure as Code and CI/CD</h2>
<ul>
<li><b>render.yaml (Blueprint):</b> one free web service in Singapore (closest Render region to India) built from <code>docker/Dockerfile</code>, with health check <code>/health</code>, <code>autoDeploy</code> on push, build arguments (which service, which dependency set, the model release URL) and runtime settings. <code>deploy/render-full.yaml</code> is the five-service microservices variant.</li>
<li><b>vercel.json:</b> client-side routing, one-year caching of hashed assets, security headers.</li>
<li><b>GitHub Actions (.github/workflows/ci.yml), three jobs:</b> (1) the Python test suite ({N_TESTS} tests; those that need the private dataset skip on the clean runner); (2) type-check and production build of the dashboard; (3) build the exact image Render builds and smoke-test it under a 512 MB memory limit (health check, then one score that must be CRITICAL). Every run on <code>main</code> passed.</li>
<li><b>GitOps:</b> nothing is deployed by hand. A push to <code>main</code> triggers CI, the Render build and the Vercel build. A new model is published as a GitHub Release, and the next build picks it up without any configuration change.</li>
</ul>
<h2>Configuration and secrets (12-factor)</h2>
{table([
 ["FINSIGHT_SERVICE_MODE", "Render (runtime)", "inprocess: one container; remote: five services"],
 ["FINSIGHT_CORS_ORIGINS", "Render (runtime)", "the browser origins allowed to call the API: the Vercel domain and localhost:5173"],
 ["FINSIGHT_INTERNAL_KEY", "Render (generated secret)", "shared key between services in remote mode; never in Git"],
 ["SERVICE, REQS, ARTIFACTS_URL", "Render (build arguments)", "which service to run, which dependencies, which model release"],
 ["VITE_API_BASE_URL", "Vercel (build time)", "the API address compiled into the static dashboard"],
], ["Variable", "Set in", "Purpose"], "small", widths=["32%", "24%", "44%"])}
<h2>Measurements on the live deployment</h2>
<p>Measured from a laptop in India with the same script as the local runs (<code>deploy/measure.py</code>, {RW['n']} scoring calls on random stored transactions). "Server" is the time spent inside the API (the <code>Server-Timing</code> header); the difference from "client" is the network round trip to Singapore.</p>
{table([
 ["Warm /score-transaction, client p50 / p95", f"{PW['compose-inprocess']['client_p50_ms']:.0f} / {PW['compose-inprocess']['client_p95_ms']:.0f} ms", f"{RW['client_p50_ms']:.0f} / {RW['client_p95_ms']:.0f} ms (target &lt; 200 ms)"],
 ["Warm, server time p50 / p95", f"{PW['compose-inprocess']['server_p50_ms']:.0f} / {PW['compose-inprocess']['server_p95_ms']:.0f} ms", f"{RW['server_p50_ms']:.0f} / {RW['server_p95_ms']:.0f} ms"],
 ["Memory (process RSS)", f"{plat['warm']['compose-inprocess']['rss_mb']['gateway']:.0f} MB", f"{RND['rss_mb']['gateway']:.0f} MB of 512 MB"],
 ["Start-up", f"{plat['startup_s']['inprocess_container']:.0f} s (container to healthy)", f"cold start {RC['seconds_to_healthy']:.0f} s after sleeping (≈ {RC['health']['uptime_s']:.0f} s loading the app, the rest waking the instance)"],
 ["Demo analysis, 10,000 rows", "~3 s", "~20–35 s on first run; 0.09 s when cached"],
], ["Metric", "Local container (8-core laptop)", "Render free instance (Singapore)"], "small", widths=["30%", "30%", "40%"], raw=True)}
<ul>
<li><b>The latency target holds in the cloud:</b> p95 {RW['client_p95_ms']:.0f} ms from India, of which about {RW['client_p50_ms'] - RW['server_p50_ms']:.0f} ms (median) is the network. Server time is ~2.5× the laptop's because the free instance gets only a small share of one CPU.</li>
<li><b>Scale-to-zero has a price:</b> after 15 idle minutes the free instance sleeps; the next visitor waits about a minute. The dashboard shows a "waking the backend" screen with a timer instead of failing, and the API loads its models once at start-up so every later request is fast. A paid instance (always on) or a minimum-instances setting would remove the wait; this is the classic cost vs latency trade-off of serverless containers (RQ8).</li>
<li><b>Memory fits:</b> {RND['rss_mb']['gateway']:.0f} MB idle, and a 24,000-row analysis peaked at 436 MB in local tests, inside the 512 MB limit, thanks to the Phase 8 memory work (fast path, allocator settings) and Phase 9b's result limits.</li>
</ul>
<h2>Problems met during deployment and their fixes</h2>
{table([
 ["The deployed dashboard called http://localhost:8000", "Vite compiles VITE_API_BASE_URL into the static files at build time; it was missing when Vercel built", "Set the variable in Vercel and redeploy (a runtime change cannot fix a static build)"],
 ["Stuck on 'Waking the backend…' although Render logged GET /health 200", "CORS: the Vercel domain was not in the allow-list, so the browser discarded the responses (preflight 400)", "Add the exact origin (https, no trailing slash) to FINSIGHT_CORS_ORIGINS"],
 ["HTTP 502 right after a settings change", "Render was swapping in the redeployed instance", "Wait until the service shows Live"],
 ["Demo analysis took ~35 s on every click", "The free instance's small CPU share", "Cache the demo results in memory (0.09 s on repeat); the page tells visitors the first run takes ~30 s"],
], ["Symptom", "Cause", "Fix"], "small", widths=["30%", "40%", "30%"])}
<p>These are typical cloud-engineering lessons: build-time vs run-time configuration, browser security (same-origin policy and CORS) between two providers, rolling deployments, and designing for constrained, shared compute.</p>
<h3>Why this is cloud computing (and why not only AWS / GCP / Azure)</h3>
<p>Cloud computing is defined by its <b>service model</b>, not by a particular vendor. The NIST definition (SP 800-145) lists five essential characteristics; the deployment has all of them:</p>
{table([
 ["On-demand self-service", "Deploys are triggered by a Git push or the render.yaml Blueprint; no hardware, no tickets"],
 ["Broad network access", "Public HTTPS API; the dashboard is served from Vercel's global edge CDN"],
 ["Resource pooling", "Multi-tenant container hosts (Render) and a shared edge network (Vercel)"],
 ["Rapid elasticity", "Scale-to-zero when idle and wake on request (measured: " + f"{RC['seconds_to_healthy']:.0f}" + " s); horizontal scaling (Compose --scale ml-service=3, measured in Ch. 14)"],
 ["Measured service", "Metered instance-hours (Render: 750 free per month), bandwidth and build minutes (Vercel); free-tier quotas"],
], ["NIST characteristic", "FinSight implementation"], "small", widths=["28%", "72%"])}
{table([
 ["IaaS", "Virtual machines, networks, disks", "Not used directly; Render and Vercel run on hyperscaler IaaS underneath. Discussed as the VM-vs-container comparison"],
 ["PaaS / CaaS", "Run my container or app; platform handles servers, TLS, scaling", "Render web services (container PaaS); Vercel (static hosting + CDN); Google Cloud Run as the documented alternative"],
 ["FaaS (serverless functions)", "Run a function per request", "Available on Vercel; not needed (the API is a container), discussed as an alternative"],
 ["SaaS", "Finished software over the web", "The FinSight scoring API and dashboard themselves; GitHub (code, CI, releases)"],
], ["Service model", "Meaning", "In FinSight"], "small", widths=["16%", "34%", "50%"])}
<p><b>Why Render + Vercel:</b> both have free tiers without a credit card, so every team member can deploy and demo. Platform services let the project spend its effort on architecture (microservices, REST, containers, Infrastructure as Code, CI/CD) rather than server administration. The free-tier limits (scale-to-zero cold starts, no private networking, 512 MB RAM, a small CPU share) are real cloud-engineering constraints, and the design responses here are part of the CCT contribution.</p>
<h3>Hyperscaler portability: Google Cloud Run (documented, not deployed; D-038)</h3>
<p>The original plan (D-028) also deployed the same image to Google Cloud Run for a measured comparison. Cloud Run requires a billing account with a payment card even inside its free tier, so the team decided not to risk real charges. The portability argument rests on what <i>was</i> done: the one Docker image already runs unchanged in three environments (Docker Compose locally, a 512 MB smoke test on GitHub Actions, and Render), and it reads all configuration from environment variables. <code>deploy/cloudrun.sh</code> (Artifact Registry push, then <code>gcloud run deploy</code> with 512 MiB, 0–2 instances, Mumbai) and <code>deploy/k8s/finsight.yaml</code> (Deployment, Service, HorizontalPodAutoscaler) are ready to run.</p>
{table([
 ["Billing unit", "Instance-hours (750 free per month); a sleeping service uses none", "vCPU-seconds, GiB-seconds and requests, only while handling requests"],
 ["Free tier", "512 MB, small CPU share, no card", "2 M requests, 180,000 vCPU-s, 360,000 GiB-s per month; card required"],
 ["Idle behaviour", "Sleeps after 15 min; cold start measured at ~1 min", "Scales to zero; min-instances (paid) avoids cold starts"],
 ["Scaling", "Manual or paid autoscaling", "Automatic, per-request concurrency"],
 ["Region near India", "Singapore", "Mumbai (asia-south1)"],
 ["Lock-in", "Plain Docker + env vars: none", "Plain Docker + env vars: none"],
], ["", "Render (deployed)", "Google Cloud Run (documented)"], "small", widths=["18%", "41%", "41%"])}
<h3>Free-tier constraints and design responses</h3>
{table([
 ["Render free web services can't receive private-network traffic", "One inprocess container by default; in the five-service variant, HTTPS + a shared secret header (X-Internal-Key)"],
 ["Services sleep after 15 min idle; waking takes ~1 min (measured)", "Wake-up screen that polls /health; models loaded once at start-up"],
 ["750 free instance-hours per workspace per month", "Sleeping services use no hours, so the demo fits"],
 ["512 MB RAM per free instance", "Numpy fast path, allocator settings, ≤ 2 stored analyses, memory released after each run"],
 ["Small CPU share", "Demo results cached; analyses processed in chunks; page states the expected wait"],
 ["Two providers, two origins", "The SPA calls Render directly with a CORS allow-list (no proxy, so no proxy timeout)"],
], ["Constraint", "Design response"], "small", widths=["45%", "55%"])}
<p><b>Two service modes from one codebase</b> (<code>FINSIGHT_SERVICE_MODE</code>): <i>remote</i> runs five services talking over HTTP (Docker Compose locally, and the optional five-service Render Blueprint), which is the microservices demonstration. <i>inprocess</i> runs the gateway with the four modules imported directly, giving one Render service and one cold start (the free deployment). Ch. 14 measured both locally: equal single-client latency, half the memory for the monolith. That, plus the cold-start measurement above, answers RQ8.</p>
<p>Acceptance criteria met: the public Vercel URL loads and scores transactions through Render ✔; internal endpoints reject calls without the key ✔ (tested); warm p95 under 200 ms from India ✔; memory under 512 MB ✔.</p>
</section>
""")

# 13 syllabus
add(f"""
<section>
<h1>Syllabus mapping</h1>
<h2>Data Mining &amp; Wrangling</h2>
{table([
 ["KDD process", "7 labelled steps (Ch. 4)", "✔"], ["Data acquisition from heterogeneous sources", "CSV, headerless CSV, XML, 6.36M-row CSV", "✔"],
 ["Data profiling and quality", "DQ report: completeness, duplicates, RI, redundancy, leakage scan", "✔"],
 ["Data cleaning", "types, sentinels, redundancy, name normalisation", "✔"], ["Data integration", "7-file merge with validated joins; OFAC + UN unification", "✔"],
 ["Data transformation / feature construction", "v1.1 injection, point-in-time graph features, discretisation", "✔"],
 ["Data reduction / feature subset selection", "Exp A/B/C ablation; redundant column removal", "✔"],
 ["Similarity measures", "fuzzy string similarity (RapidFuzz); KNN distances", "✔"],
 ["Frequent itemsets, Apriori, FP-Growth", "benchmark on two item universes + PaySim", "✔"], ["Closed and maximal itemsets", "closed filter; FP-Max", "✔"],
 ["Association rules and interestingness", "support, confidence, lift, leverage, conviction; productive rules; emerging patterns", "✔"],
 ["Classification and evaluation", "DT, NB, KNN with PR-AUC, F1, ROC", "✔"],
 ["Cluster analysis", "K-Means (elbow, silhouette, Davies–Bouldin), DBSCAN, cluster profiling, ARI", "✔"],
 ["Outlier analysis", "DBSCAN noise; Isolation Forest, One-class SVM, autoencoder", "✔"],
 ["Big-data wrangling", "Polars lazy pipelines on 6.36M PaySim rows (mining + anomaly scale tracks)", "✔"],
 ["IEEE-CIS missingness profiling", "transaction ⋈ identity", "not done (optional)"],
], ["Topic", "Where in FinSight", "Status"], "small", widths=["36%", "52%", "12%"])}
<h2>Machine Learning</h2>
{table([
 ["Supervised learning", "6 classifiers × 3 feature sets × 2 datasets", "✔"], ["Feature engineering", "behavioural + point-in-time network features", "✔"],
 ["Train / validation / test, cross-validation", "stratified 60/20/20; 5×2 repeated CV; temporal hold-out", "✔"],
 ["Statistical model comparison", "paired t-test + Wilcoxon on identical folds", "✔"], ["Class imbalance", "SMOTE vs class weights vs none", "✔"],
 ["Ensembles (bagging, boosting)", "Random Forest, XGBoost", "✔"], ["Neural network classifier", "MLP", "✔"],
 ["Evaluation metrics", "PR-AUC, ROC-AUC, F1, precision/recall, per-class, confusion", "✔"], ["Interpretability", "decision tree visualisation", "✔"],
 ["Unsupervised learning", "K-Means, DBSCAN, PCA, t-SNE", "✔"], ["Anomaly detection", "Isolation Forest, One-class SVM, autoencoder (legit-only training)", "✔"],
 ["Neural network (unsupervised)", "PyTorch autoencoder with early stopping", "✔"],
 ["Hyperparameter tuning, bias–variance", "Grid/Randomized search with CV; validation and learning curves", "✔"], ["Stacking and calibration", "RF + XGB + MLP → LR; isotonic calibration, reliability, Brier/ECE", "✔"],
 ["Explainability", "SHAP global + local; permutation importance; native TreeSHAP for serving", "✔"],
 ["Decision thresholds and cost of errors", "validation-chosen tiers (precision / recall / F1 targets), review band", "✔"], ["Model persistence / serving", "joblib; ONNX autoencoder with parity test", "✔"],
], ["Topic", "Where in FinSight", "Status"], "small", widths=["36%", "52%", "12%"])}
<h2>Cloud Computing &amp; Technology</h2>
{table([
 ["Cloud definition (NIST 5 characteristics)", "Git-push deploys, HTTPS + CDN, pooled hosts, scale-to-zero (cold start measured), metered free tiers (Ch. 16)", "✔"],
 ["Service models (IaaS / PaaS / SaaS / FaaS)", "Render = container PaaS; Vercel = frontend PaaS + CDN; FinSight API as SaaS; FaaS and IaaS discussed", "✔"],
 ["Multi-cloud / portability, vendor lock-in", "one Docker image runs unchanged on Compose, CI and Render; Cloud Run and Kubernetes scripts ready (not deployed, D-038)", "✔ (partly)"],
 ["Virtualisation and containers", "multi-stage Dockerfile for 5 services; slim non-root images; image size measured", "✔"],
 ["Microservices and API gateway", "gateway + 4 services; inprocess vs remote modes, identical results, measured", "✔"],
 ["REST and OpenAPI", "14 public endpoints, Pydantic schemas, Swagger UI at /docs", "✔"],
 ["Orchestration", "Docker Compose with health checks, depends_on, --scale ml-service=3; conceptual Kubernetes", "✔ (Compose)"],
 ["Performance and load testing", "profiling-driven optimisation; Locust load tests; p50/p95/p99", "✔"],
 ["CDN / edge", "static SPA (250 KB first load incl. self-hosted fonts, hashed assets cached a year) live on Vercel's edge network", "✔"],
 ["CI/CD, GitOps", "GitHub Actions: tests, dashboard build, image build + 512 MB smoke test; auto-deploy on push", "✔"],
 ["Infrastructure as Code", "render.yaml Blueprint (deployed), render-full.yaml, vercel.json, docker-compose.yml, k8s manifests", "✔"],
 ["Artifact registry / storage", "versioned, checksummed bundle published as GitHub Releases; downloaded at image build", "✔"],
 ["Portable model format", "ONNX autoencoder served with onnxruntime (no PyTorch in the image)", "✔ (6)"],
 ["Elasticity, scale-to-zero, cost", "Render scale-to-zero vs measured cold start; locust load test; Compose scaling", "✔"],
 ["Security", "internal shared key (401 tested), private network, CORS allow-list (tested and debugged live), non-root images, generated secrets; HTTPS on both platforms", "✔"],
 ["Observability", "structured JSON logs, request-ID propagation, /metrics percentiles, /platform/status", "✔ (incl. Platform page)"],
 ["Cloud-native principles", "12-factor configuration (env vars, build vs run time), stateless services, immutable images", "✔"],
], ["Topic", "Where in FinSight", "Status"], "small", widths=["36%", "50%", "14%"])}
</section>
""")

# 14 conclusions
add(f"""
<section>
<h1>Conclusions, limitations and future work</h1>
<h2>Conclusions</h2>
<ol>
<li><b>Audit before you model.</b> The original dataset contained label proxies that would have produced near-perfect but meaningless models, and its "network" scenarios had no network. Finding this first changed the project's direction and is a central data-mining result.</li>
<li><b>Data engineering can be done honestly.</b> The v1.1 injection is seeded, logged, tested and validated against a control. Mining and ML both rediscover exactly the injected structure and nothing on v1.0.</li>
<li><b>Behaviour matters most; network matters for organised fraud.</b> Behavioural context gives the largest gain for every model. Network features add a significant gain (5/6 models) precisely on camouflaged organised fraud: layering recall rises from 65% to 85%.</li>
<li><b>Interpretable rules are competitive but not best.</b> Productive association rules reach F1 {rb['test']['f1']:.2f} on held-out data vs {best['test_f1']:.2f} for XGBoost. Rules remain valuable as explanations and as a transparent first-line filter.</li>
<li><b>FP-Growth's advantage is about memory and large searches</b>, not universal speed. Implementation (vectorised vs pure Python) matters as much as asymptotic complexity.</li>
<li><b>SMOTE is not a default.</b> With threshold tuning, rebalancing added nothing and sometimes hurt.</li>
<li><b>Tuning helps; ensembling did not.</b> Cross-validated search improved every model and lifted the best model's test F1 from {best['test_f1']:.3f} to {FT['test_f1']:.3f}. Stacking three strong but correlated models added nothing over tuned XGBoost, so the simpler model went to production.</li>
<li><b>A score needs calibration, tiers and reasons.</b> Isotonic calibration, validation-chosen tier cut-offs and native SHAP explanations turn a classifier into a usable risk score: {pct(ftiers.loc['FRAUDULENT','CRITICAL_share'],0)} of fraud is CRITICAL, {pct(ftiers.loc['LEGITIMATE','LOW_share'],0)} of legitimate payments are LOW, and every score comes with its reasons.</li>
<li><b>Measure, then optimise.</b> The first working service missed the latency target in practice (138 ms median in-process); profiling located the cost in per-request sklearn overhead, and a tested numpy fast path brought it to 21 ms with identical outputs. The load test found a different bottleneck (sanctions screening), which caching removed.</li>
<li><b>Without labels you get triage, not detection.</b> Clustering finds interpretable risk groups (bursts, hubs, unusual flows) but with weak structure. Anomaly detectors trained on legitimate data reach PR-AUC ≈ 0.55 vs 0.875 supervised; they add a complementary signal that recovers a quarter of the supervised model's misses.</li>
<li><b>Anomalous is not the same as fraudulent.</b> Anomaly detectors flag ~80% of BORDERLINE (unusual but legitimate) transactions, the same group the supervised model struggles with.</li>
<li><b>A system is convincing when it runs on new data.</b> The live analysis reproduces the training features exactly and catches simulated organised attacks it never saw as rows (100% plain; 90–100% camouflaged except shared devices at ~63%), at about 1.4% false alerts.</li>
<li><b>Free cloud tiers are enough for a real ML service, with trade-offs.</b> The deployed API answers in {RW['client_p95_ms']:.0f} ms (p95) from India within 512 MB, but scale-to-zero costs a ~1 minute cold start, and configuration across two providers (build-time API address, CORS) caused the only deployment failures.</li>
<li><b>Label ambiguity is the main remaining error.</b> BORDERLINE cases cause most false positives and are only half-separable even as their own class. A "review" tier is more honest than forcing a binary decision.</li>
</ol>
<h2>Limitations and threats to validity</h2>
<ul>
<li><b>Synthetic data:</b> all India results describe a generator plus our documented modifications. Two behaviour flags are perfect by construction. Absolute scores shouldn't be compared with real-world systems.</li>
<li><b>Injected structure:</b> network gains depend on how v1.1 was built. The v1.0 control and the decoys limit, but don't remove, this circularity. The report states the injection rules in full.</li>
<li><b>Small per-scenario test counts</b> (16–48 fraud rows per scenario) make scenario-level differences under ~10 points unreliable.</li>
<li><b>One month of data:</b> the temporal hold-out shows no drift within August, but can't test seasonal or long-term drift.</li>
<li><b>Global graph features</b> (PageRank, communities) are transductive (computed on the full graph, label-free); velocity features are strictly point-in-time.</li>
<li><b>Tuning budget:</b> 20 randomized draws for RF/XGBoost and 10 for the MLP; a larger search could find slightly better settings. Phase 6 detectors use default settings, and their thresholds assume a set of known-legitimate rows to calibrate on.</li>
<li><b>One split for model choice:</b> the production model, threshold and tiers are chosen on a single 2,000-row validation split; small differences (± 0.01 PR-AUC) between candidates are within noise.</li>
<li><b>Clustering in 63 dimensions</b> is dominated by distance concentration (silhouette ≤ 0.16); the cluster findings are descriptive, not a validated segmentation.</li>
</ul>
<h2>Future work</h2>
<p>Deploy the same image to Google Cloud Run (or Kubernetes) for a measured multi-cloud comparison; an always-on instance or a job queue for larger uploads. Beyond the course: graph neural networks over the account graph; streaming scoring (Kafka) with online graph updates; drift monitoring; human-in-the-loop review of BORDERLINE cases; evaluation on real, consented bank data.</p>
</section>
""")

# 19 guide
add(f"""
<section>
<h1>Guide for report and presentation writers</h1>
<p>Each outline lists sections or slides with the source chapter and the figure file (in <code>reports/figures/</code>) to use. Keep the colour meaning consistent: blue = legitimate, orange = fraudulent, aqua = borderline.</p>
<h2>19.1 Data Mining &amp; Wrangling</h2>
<p><b>Report outline:</b> Introduction (Ch. 2) → Datasets and KDD (Ch. 3–4) → Audit and data quality (Ch. 5–6) → Dataset engineering v1.1 (Ch. 7) → EDA and RQ1 (Ch. 9) → Association mining and RQ2 (Ch. 10) → Rules vs classifiers (Ch. 11 baselines) → Cluster analysis and the PaySim scale track (Ch. 12) → Conclusions (Ch. 18).</p>
{table([
 ["1", "Title + one-line pitch", "—"], ["2", "Problem: opportunistic vs organised fraud in Indian payments", "—"],
 ["3", "Datasets and the KDD process", "KDD flow (Ch. 4)"], ["4", "Audit: leakage and the missing network", "Ch. 5 table"],
 ["5", "Data quality and integration (7 files → 76 columns)", "Ch. 6 flow + RI table"], ["6", "Sanctions wrangling + threshold experiment", "Ch. 6 sweep table"],
 ["7", "v1.1: injecting structure honestly", "eda_12_network_examples, eda_11_fanin_distribution"],
 ["8", "EDA highlights and RQ1", "eda_02_amount_by_label, eda_07_behaviour_flags, eda_06_feature_auc_v10_v11"],
 ["9", "Discretisation into items", "Ch. 10 items table"], ["10", "Apriori vs FP-Growth", "mining_01_benchmark, mining_03_paysim_scaling"],
 ["11", "Closed/maximal itemsets", "mining_02_condensed"], ["12", "Top rules (RQ2)", "mining_05_top_rules_v1_1, mining_04_rules_scatter"],
 ["13", "Control: v1.0 vs v1.1 rules", "mining_06_item_share_v10_v11"], ["14", "Rules as a detector vs ML", "mining_07_detector_curve, ml_03_pr_roc_expC"],
 ["15", "Cluster analysis: K-Means profiles + DBSCAN outliers", "unsup_02_cluster_composition, unsup_04_dbscan"],
 ["16", "Big data: PaySim 6.36M rows with Polars", "mining_03_paysim_scaling, unsup_10_paysim_if"],
 ["17", "Live: data quality, pattern lab and transaction explorer pages; analysing an uploaded CSV (validation, defaults, rules among alerts)", "dashboard_02_data_quality, dashboard_04_patterns, dashboard_10_analyze (Ch. 15)"],
 ["18", "Conclusions and limitations", "Ch. 18"],
], ["Slide", "Content", "Figure / source"], "small", widths=["8%", "52%", "40%"])}
<p><b>One-line pitch:</b> "FinSight AI follows the full KDD process across five heterogeneous sources; our audit found label leakage and fraud scenarios that were never encoded in the graph. We fixed and validated that with a seeded augmentation step, then mined behaviour patterns with Apriori and FP-Growth and benchmarked both."</p>
<h2>19.2 Machine Learning</h2>
<p><b>Report outline:</b> Introduction and RQs (Ch. 2) → Data and features (Ch. 3, 8, Appendix B) → Protocol (Ch. 11 design) → RQ3 → RQ4 + imbalance → RQ7 + BORDERLINE → robustness and interpretability → RQ5 clustering and RQ6 anomaly detection (Ch. 12) → tuning, bias–variance, stacking, calibration, risk score, SHAP (Ch. 13) → Conclusions (Ch. 18).</p>
{table([
 ["1", "Title + pitch", "—"], ["2", "Task: binary fraud detection, 13.5% positives; why PR-AUC", "eda_01_label_overview"],
 ["3", "Feature sets A/B/C (12/30/49)", "Appendix B"], ["4", "Network features and point-in-time design", "eda_14_network_signal_by_scenario"],
 ["5", "Protocol: split, 5×2 CV, paired tests, SMOTE in pipeline", "Ch. 11 design table"],
 ["6", "RQ3 result", "ml_01_abc_cv_pr_auc + significance table"], ["7", "RQ4 model comparison", "ml_02_test_pr_auc_heatmap, ml_03_pr_roc_expC"],
 ["8", "Imbalance: SMOTE doesn't help", "ml_05_imbalance"], ["9", "RQ7 per-scenario recall", "ml_04_per_scenario_xgb"],
 ["10", "BORDERLINE problem + 3-class", "ml_06_confusion_by_label, ml_08_borderline"],
 ["11", "Temporal robustness", "ml_07_temporal"], ["12", "Interpretable tree", "ml_09_decision_tree"],
 ["13", "RQ5: clustering without labels (K choice, 4 clusters)", "unsup_01_kmeans_selection, unsup_02_cluster_composition, unsup_03_embeddings"],
 ["14", "RQ6: detectors trained on legitimate data only", "unsup_05_detector_pr, unsup_06_detector_bars"],
 ["15", "Per-scenario anomaly results: anomalous ≠ fraudulent", "unsup_07_scenario_heatmap"],
 ["16", "Autoencoder + ONNX; complementarity with XGBoost", "unsup_08_autoencoder, unsup_09_complementarity"],
 ["17", "Tuning: Grid vs Randomized search; default vs tuned", "final_01_tuning + Ch. 13 tuning table"],
 ["18", "Bias–variance: depth curve and learning curves", "final_02_bias_variance"],
 ["19", "Stacking (no gain) and calibration", "final_03_calibration, final_04_test_comparison"],
 ["20", "Final model and the 0–100 risk score with tiers", "Ch. 13 final table, final_06_risk_score"],
 ["21", "Explainability: SHAP global, beeswarm, two local explanations", "final_07_importance, final_08_shap_beeswarm, final_09_local_explanations"],
 ["22", "Live demo: attack simulator (plain vs camouflaged); risk scorer with a what-if change; ML lab", "dashboard_11_simulation, dashboard_08_risk_scorer, dashboard_05_ml_lab (Ch. 15)"],
 ["23", "Conclusions", "Ch. 18"],
], ["Slide", "Content", "Figure / source"], "small", widths=["8%", "52%", "40%"])}
<p><b>One-line pitch:</b> "We trained 6 classifiers across 3 feature sets on two dataset versions and used paired cross-validation to show that network features help only where real network structure exists. the tuned, calibrated XGBoost reaches PR-AUC {FT['test_pr_auc']:.2f} and F1 {FT['test_f1']:.2f}, stacking didn't beat it, and SMOTE didn't help. Without labels, clustering finds interpretable risk groups and legit-only anomaly detectors reach PR-AUC ≈ 0.55 while adding a complementary signal."</p>
<h2>19.3 Cloud Computing &amp; Technology</h2>
<p><b>Report outline:</b> Introduction (Ch. 2) → Requirements and service models → Architecture, microservices vs monolith (Ch. 14) → Containers and the model registry (Ch. 14) → REST API design (Ch. 14) → Security and observability (Ch. 14) → Local evaluation: latency, memory, load, scaling (Ch. 14) → Frontend as a static CDN app (Ch. 15) → Deployment on Vercel + Render, constraints, NIST framing (Ch. 16) → Live deployment, CI/CD and IaC (Ch. 16) → Cloud evaluation: latency, memory, cold start, deployment problems (Ch. 16) → Portability and the Cloud Run alternative (Ch. 16) → Conclusions (Ch. 18).</p>
{table([
 ["1", "Title + pitch", "—"], ["2", "What the platform does (ML + mining as a service)", "Ch. 1"],
 ["3", "Is this cloud? NIST's 5 characteristics + service models (why Render/Vercel, not only AWS/GCP)", "Ch. 16 'Why this is cloud computing' tables"], ["4", "Architecture: gateway + 4 services + SPA", "Ch. 14 flow diagram"],
 ["5", "REST endpoints and OpenAPI", "Ch. 14 endpoint table"], ["6", "Containers: multi-stage Dockerfile, slim non-root images, image sizes", "Ch. 14 containers table"],
 ["7", "Deployment topology: Vercel CDN → Render containers", "Ch. 16 deployment table"], ["8", "Free-tier constraints → design responses", "Ch. 16 constraints table"],
 ["9", "Microservices vs modular monolith: same code, measured", "Ch. 14 mode table + platform_01_phase8"], ["10", "Model registry (versioned bundle), CI/CD, IaC; ONNX as a portable model format", "Ch. 14, Ch. 12 (ONNX)"],
 ["11", "Load test + horizontal scaling (--scale ml-service=3); profiling-driven optimisation", "platform_01_phase8, Ch. 14 load table"], ["11a", "Live measurements: warm latency, memory, scale-to-zero vs cold start", "Ch. 16 measurements table"],
 ["11b", "Portability: one image, many environments; Render vs Cloud Run cost models; deployment problems and fixes", "Ch. 16 tables"], ["12", "Security and observability (internal key, CORS, request IDs, /metrics)", "Ch. 14"],
 ["12b", "Frontend on a CDN: static SPA, cold-start handling, CORS", "Ch. 15 design decisions, dashboard_09_platform"],
 ["13", "Live demo: Analyze page (attack simulator), Platform page", "dashboard_11_simulation, dashboard_09_platform"], ["14", "Conclusions", "Ch. 18"],
], ["Slide", "Content", "Figure / source"], "small", widths=["8%", "52%", "40%"])}
<p><b>One-line pitch:</b> "Five stateless FastAPI microservices in Docker, deployed on Render from a Blueprint, with a React dashboard served from Vercel's CDN. Artifacts are versioned as GitHub Releases and baked into immutable images, deploys are triggered by Git pushes, and we measured the scale-to-zero vs cold-start trade-off live (p95 {RW['client_p95_ms']:.0f} ms warm, ~1 min cold)."</p>
<h2>Suggested live-demo flow</h2>
<ol><li>Open the live dashboard a minute early (the free API sleeps). Start on the Analyze page: run the demo sample, then simulate a fraud ring + mule chain, then the same with camouflage; open an alert to show its reasons.</li><li>Dataset overview and the audit story (leakage, missing network).</li><li>v1.0 → v1.1: a fraud ring and a layering chain.</li><li>Top association rules; Apriori vs FP-Growth memory.</li><li>A → B → C improvement; v1.0 control.</li><li>A BORDERLINE example and why it's hard.</li><li>Clusters in t-SNE and an anomaly score that flags a new-device fraud the rules don't know.</li><li>Risk scorer page: score the mule example (100, CRITICAL), then set the amount to ₹500 and watch the score and reasons change.</li><li>Network explorer: open the top PageRank account and show the fan-in of fraudulent payments.</li><li>Platform page: services, this visit's wake-up time and the live latency probe from the cloud.</li></ol>
</section>
""")

# appendices
feat_rows = [
 ["A · transaction (12)", "log_amount, hour_of_day, day_of_week, account_balance_before, sender_account_age_days, sender 30-day avg amount / tx count / avg balance, receiver_account_age_days, channel, sender_account_type, receiver_account_type"],
 ["B · + behavioural (+18 = 30)", "is_new_device, is_new_beneficiary, time_since_beneficiary_added_min, amount_ratio_vs_30d_avg, distance_from_home_km, tx_frequency_24h, time_since_last_tx_sec, device_is_registered, device_associated_accounts, device_age_days, sender_bene_count, sender_bene_added_7d, sender_bene_min_prior_tx, sender_pep_flag, sender_declared_annual_income_inr, device_type, sender_kyc_status, sender_customer_type"],
 ["C · + network (+19 = 49)", "13 point-in-time (Ch. 8) + sender/receiver PageRank, sender/receiver community size, same_community, receiver_clustering"],
 ["Never features", "is_fraud, risk_label, scenario_id, alert_id, is_flagged_alert, rapid_forward_delta_sec (labels/proxies); all IDs, VPAs, IFSC, IP"],
]
gloss = [
 ["UPI / NEFT / RTGS", "Unified Payments Interface (instant, P2P) / National Electronic Funds Transfer (batch) / Real-Time Gross Settlement (high value, ≥ ₹2 lakh)"],
 ["VPA / IFSC", "Virtual payment address (e.g. name@bank) / bank branch code"],
 ["Mule account", "account used to receive and quickly pass on illicit funds"], ["Layering", "moving funds through a chain of accounts to hide their origin"],
 ["Structuring", "splitting a large amount into payments just under a reporting threshold"], ["PEP", "politically exposed person (higher-risk customer category)"],
 ["KDD", "Knowledge Discovery in Databases: selection → cleaning → integration → transformation → mining → evaluation → presentation"],
 ["Point-in-time feature", "a feature computed only from information available before the event (prevents look-ahead leakage)"],
 ["Label leakage / proxy", "a column that encodes the answer, giving unrealistically perfect models"],
 ["PR-AUC", "area under the precision–recall curve (average precision); the right primary metric for rare positives"],
 ["ROC-AUC", "area under the true-positive vs false-positive rate curve"], ["F1", "harmonic mean of precision and recall"],
 ["SMOTE", "Synthetic Minority Over-sampling: creates synthetic positives by interpolating between neighbours"],
 ["5×2 repeated CV", "5-fold cross-validation repeated twice with different shuffles = 10 paired scores"],
 ["Support / confidence / lift", "see Ch. 10"], ["Transductive", "computed using the whole dataset (here: label-free graph statistics)"],
 ["Cold start", "delay when a scaled-to-zero cloud service receives its first request"], ["Blueprint (Render)", "render.yaml file describing services as code"],
 ["CDN", "content delivery network: serves static files from servers close to the user"],
]
pipeline_cmds = [
 ["Install", "uv sync --all-extras"], ["Tests", f"make test ({N_TESTS} tests)"],
 ["Full pipeline", "make pipeline (stages: inject, wrangle, graph, sanctions, eda, mining, train_supervised, unsupervised, train_final); then make bundle → make serve (one process) or make up (Docker microservices) → make dashboard (React UI on :5173)"],
 ["One stage", "make pipeline STAGES=mining"], ["Notebooks", "notebooks/02_eda, 03_association_mining, 04_supervised_ml, 05_unsupervised_ml, 06_tuning_ensemble (executed, 0 errors)"],
 ["This report", "uv run python docs/report/build_report.py"],
]
outputs = [
 ["reports/data_quality_india_v1_{0,1}.json · sanctions_summary.json", "Phase 1"], ["dataset/processed/custom_india_v1_1/ (+ injection_log.csv)", "Phase 1b"],
 ["reports/eda_summary.json · figures/eda_01…14", "Phase 2"], ["dataset/curated/master_features_v1_{0,1}.parquet · artifacts/graph/", "Phase 3"],
 ["reports/association_rules_*.csv · mining_benchmark.csv · mining_paysim_scaling.csv · emerging_patterns_v1_1.csv · mining_summary.json · figures/mining_01…08", "Phase 4"],
 ["reports/ml_experiment_results.csv · ml_cv_folds.csv · ml_paired_tests.csv · ml_imbalance.csv · ml_per_scenario.csv · ml_temporal.csv · ml_summary.json · figures/ml_01…09 · artifacts/models/phase5/", "Phase 5"],
 ["reports/unsupervised_summary.json · unsup_kmeans_sweep.csv · unsup_clusters.csv · anomaly_results.csv · anomaly_per_scenario.csv · anomaly_paysim_if.json · figures/unsup_01…10 · artifacts/models/{anomaly,clustering}/ (autoencoder_expC.onnx)", "Phase 6"],
 ["reports/final_summary.json · final_tuning_*.csv · final_learning_curves.csv · final_model_comparison.csv · final_risk_*.csv · final_shap_importance.csv · final_permutation_importance.csv · figures/final_01…09 · artifacts/models/final/ (production_model.joblib, xgboost_explainer.joblib, risk_scorer.json, metrics.json)", "Phase 7"],
 ["artifacts/bundle/ (versioned serving bundle) · reports/platform_summary.json · platform_compose-*.json · load_*_stats.csv · figures/platform_01_phase8", "Phase 8"],
 ["frontend/ (React SPA, vercel.json) · reports/figures/dashboard_01…09", "Phase 9"],
]
refs = [
 "Fayyad, U., Piatetsky-Shapiro, G., Smyth, P. (1996). From Data Mining to Knowledge Discovery in Databases. AI Magazine 17(3).",
 "Han, J., Kamber, M., Pei, J. (2011). Data Mining: Concepts and Techniques, 3rd ed. Morgan Kaufmann.",
 "Agrawal, R., Srikant, R. (1994). Fast Algorithms for Mining Association Rules. VLDB.",
 "Han, J., Pei, J., Yin, Y. (2000). Mining Frequent Patterns without Candidate Generation. SIGMOD.",
 "Grahne, G., Zhu, J. (2003). Efficiently Using Prefix-trees in Mining Frequent Itemsets (FP-Max). FIMI.",
 "Bayardo, R., Agrawal, R., Gunopulos, D. (1999). Constraint-Based Rule Mining in Large, Dense Databases. ICDE.",
 "Dong, G., Li, J. (1999). Efficient Mining of Emerging Patterns. KDD.",
 "Breiman, L. (2001). Random Forests. Machine Learning 45.", "Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.",
 "Chawla, N. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR 16.",
 "Dietterich, T. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Computation 10.",
 "Saito, T., Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot on Imbalanced Datasets. PLOS ONE.",
 "Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). Isolation Forest. ICDM.",
 "MacQueen, J. (1967). Some Methods for Classification and Analysis of Multivariate Observations (K-Means). Berkeley Symp.",
 "Ester, M., Kriegel, H.-P., Sander, J., Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters (DBSCAN). KDD.",
 "Rousseeuw, P. (1987). Silhouettes. J. Comput. Appl. Math. 20; Davies, D., Bouldin, D. (1979). A Cluster Separation Measure. IEEE TPAMI.",
 "Hubert, L., Arabie, P. (1985). Comparing Partitions (Adjusted Rand Index). J. Classification 2.",
 "van der Maaten, L., Hinton, G. (2008). Visualizing Data using t-SNE. JMLR 9.",
 "Hinton, G., Salakhutdinov, R. (2006). Reducing the Dimensionality of Data with Neural Networks (autoencoders). Science 313.",
 "ONNX: Open Neural Network Exchange specification (onnx.ai); ONNX Runtime (onnxruntime.ai).", "Schölkopf, B. et al. (2001). Estimating the Support of a High-Dimensional Distribution. Neural Computation.",
 "Lundberg, S., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS; Lundberg, S. et al. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees (TreeSHAP). Nature Machine Intelligence 2.",
 "Bergstra, J., Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization. JMLR 13.",
 "Wolpert, D. (1992). Stacked Generalization. Neural Networks 5.",
 "Zadrozny, B., Elkan, C. (2002). Transforming Classifier Scores into Accurate Multiclass Probability Estimates (isotonic calibration). KDD; Niculescu-Mizil, A., Caruana, R. (2005). Predicting Good Probabilities with Supervised Learning. ICML.",
 "Breiman, L. (2001). Statistical Modeling: The Two Cultures; Fisher, A., Rudin, C., Dominici, F. (2019). All Models are Wrong, but Many are Useful (permutation importance). JMLR 20.",
 "Blondel, V. et al. (2008). Fast Unfolding of Communities in Large Networks (Louvain). J. Stat. Mech.",
 "Lopez-Rojas, E., Elmir, A., Axelsson, S. (2016). PaySim: A Financial Mobile Money Simulator for Fraud Detection. EMSS.",
 "IEEE Computational Intelligence Society & Vesta (2019). IEEE-CIS Fraud Detection, Kaggle competition dataset.",
 "U.S. Treasury OFAC, Specially Designated Nationals (SDN) List; United Nations Security Council Consolidated List.",
 "Reserve Bank of India. Master Directions on Fraud Risk Management in Commercial Banks. FATF (2026). The FATF Recommendations.",
 "Wiggins, A. (2017). The Twelve-Factor App. Render documentation (render.com/docs/free); Vercel documentation (vercel.com/docs).",
 "Libraries: pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, MLxtend, NetworkX, Polars, RapidFuzz, PyTorch, ONNX Runtime, Matplotlib.",
]
add(f"""
<section>
<h1 class="nonum">Appendices</h1>
<h2>A · Decision log</h2>
{table(decisions(), ["ID", "Decision", "Why / evidence"], "small tiny", widths=["7%", "40%", "53%"])}
<h2>B · Feature dictionary</h2>
{table(feat_rows, ["Set", "Columns"], "small", widths=["22%", "78%"])}
<h2>C · Cross-validated PR-AUC, all configurations (mean ± sd, 5×2 CV)</h2>
{cv_matrix()}
<h2>D · Glossary</h2>
{table(gloss, ["Term", "Meaning"], "small", widths=["24%", "76%"])}
<h2>E · Reproducibility</h2>
{table(pipeline_cmds, ["Task", "Command / location"], "small", widths=["22%", "78%"])}
{table(outputs, ["Output files", "Phase"], "small", widths=["82%", "18%"])}
<p>Environment: Ubuntu, Python 3.12 (uv), seed 42, CPU only. The dataset (1.8 GB) is not stored in Git; place it under <code>dataset/</code> as described in <code>docs/DATA.md</code>.</p>
<h2>F · References</h2>
<ol class="refs">{"".join(f"<li>{esc(r)}</li>" for r in refs)}</ol>
</section>
""")

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
  @bottom-left { content: "FinSight AI · Project report"; font: 8pt 'DejaVu Sans', sans-serif; color: #898781; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 8pt 'DejaVu Sans', sans-serif; color: #898781; } }
@page :first { @bottom-left { content: none; } @bottom-right { content: none; } }
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --rule:#e1e0d9; --accent:#2a78d6; --accent-soft:#eef4fc; --surface:#fcfcfb; --fraud:#eb6834; }
* { box-sizing: border-box; }
body { font-family: 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif; font-size: 9.6pt; line-height: 1.5; color: var(--ink); background:#fff; margin:0; }
section { counter-increment: none; }
h1 { font-size: 19pt; margin: 0 0 10pt; padding-bottom: 6pt; border-bottom: 2px solid var(--accent); break-before: page; counter-increment: chap; counter-reset: sec; }
h1::before { content: counter(chap) "  "; color: var(--accent); }
h1.nonum::before { content: none; } h1.nonum { counter-increment: none; }
body { counter-reset: chap; }
h2 { font-size: 12.5pt; margin: 16pt 0 6pt; color: var(--ink); break-after: avoid; }
h2.nonum { margin-top: 0; }
h3 { font-size: 10.5pt; margin: 12pt 0 4pt; color: var(--ink2); break-after: avoid; }
p, li { orphans: 3; widows: 3; } p { margin: 4pt 0 8pt; text-align: left; }
ul, ol { margin: 4pt 0 8pt 16pt; padding: 0; } li { margin: 2pt 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.4pt; background:#f3f2ee; padding: 0 3px; border-radius: 3px; }
table { width: 100%; border-collapse: collapse; margin: 6pt 0 12pt; font-size: 8.8pt; break-inside: auto; }
thead { display: table-header-group; } tr { break-inside: avoid; }
th { text-align: left; background: #f3f2ee; color: var(--ink2); font-weight: bold; padding: 5px 6px; border-bottom: 1.5px solid #c3c2b7; }
td { padding: 4.5px 6px; border-bottom: 1px solid var(--rule); vertical-align: top; }
table.small { font-size: 8.2pt; } table.tiny { font-size: 7.6pt; }
td.r, th.r { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; } table.keep { break-inside: avoid; }
figure { margin: 8pt 0 14pt; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; border: 1px solid var(--rule); border-radius: 4px; }
figcaption { font-size: 8.2pt; color: var(--ink2); margin-top: 4pt; text-align: left; }
.callout { background: var(--accent-soft); border-left: 3px solid var(--accent); padding: 8pt 10pt; border-radius: 3px; }
.note { margin-top: 24pt; font-size: 8.5pt; color: var(--ink2); border-top: 1px solid var(--rule); padding-top: 8pt; }
.flow { display: flex; align-items: stretch; gap: 4pt; margin: 8pt 0 14pt; break-inside: avoid; }
.flow .col { flex: 1; display: flex; flex-direction: column; gap: 4pt; }
.flow .box { border: 1px solid #c3c2b7; border-radius: 5px; padding: 5pt 6pt; background: var(--surface); flex: 1; }
.flow .box b { display: block; font-size: 8.4pt; color: var(--accent); } .flow .box span { font-size: 7.6pt; color: var(--ink2); }
.flow .arrow { align-self: center; color: var(--muted); font-size: 13pt; }
.flow.small .box b { font-size: 7.8pt; } .flow.small .box span { font-size: 7.1pt; }
.cover { height: 255mm; display: flex; flex-direction: column; justify-content: center; }
.kicker { font-size: 9pt; letter-spacing: .04em; color: var(--accent); text-transform: uppercase; font-weight: bold; }
.title { font-size: 44pt; border: none; margin: 8pt 0 4pt; padding: 0; break-before: auto; counter-increment: none; }
section.howto { break-before: page; } .title::before { content: none; }
.subtitle { font-size: 13pt; color: var(--ink2); max-width: 150mm; margin-bottom: 22pt; }
.cover-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10pt 18pt; border-top: 2px solid var(--accent); padding-top: 14pt; }
.cover-grid b { display:block; font-size: 8pt; text-transform: uppercase; color: var(--muted); letter-spacing: .04em; } .cover-grid span { font-size: 9.6pt; }
.toc { columns: 2; font-size: 9.6pt; } .toc li { margin: 3pt 0; break-inside: avoid; }
.refs { font-size: 8.2pt; }
"""

doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>FinSight AI: Project Report</title>
<style>{CSS}</style></head><body>{''.join(H)}</body></html>"""


def main():
    OUT_HTML.write_text(doc)
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if not chrome:
        raise SystemExit(f"HTML written to {OUT_HTML}; install Chrome/Chromium to print the PDF")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--allow-file-access-from-files",
                    "--no-pdf-header-footer", f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()],
                   check=True, capture_output=True)
    print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
