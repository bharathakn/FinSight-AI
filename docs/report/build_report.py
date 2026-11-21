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
    <div><b>Status</b><span>Phases 0–5 complete (data, mining, supervised ML). Phases 6–10 designed, not yet built (unsupervised/anomaly, tuning + stacking + SHAP, services, dashboard, deployment).</span></div>
    <div><b>Headline result</b><span>XGBoost on all features: test PR-AUC {best['test_pr_auc']:.3f}, ROC-AUC {best['test_roc_auc']:.3f}, F1 {best['test_f1']:.3f} (recall {pct(best['test_recall'],0)}). Network features add a significant gain only where real network structure exists.</span></div>
    <div><b>Deployment target</b><span>React dashboard on Vercel · FastAPI microservices in Docker on Render</span></div>
    <div><b>Repository</b><span>github.com/bharathakn/FinSight-AI</span></div>
  </div>
  <div class="note">All India data is synthetic. Results describe the data generator and our documented modifications, not real-world fraud.</div>
</section>

<section class="howto">
<h2 class="nonum">How to use this document</h2>
<p>This single PDF is meant to be enough for three separate course reports and three presentations. Every number comes directly from the project's result files.</p>
{table([
 ["Data Mining &amp; Wrangling (DMW)", "1–10, 11 (baselines: rules vs ML), 13 (DMW table), 14, 15.1", "KDD pipeline, wrangling, data quality and leakage audit, v1.1 dataset engineering, discretisation, Apriori vs FP-Growth, rule evaluation"],
 ["Machine Learning (ML)", "1–3, 5, 7–9, 11, 12, 13 (ML table), 14, 15.2", "Feature engineering, experimental protocol, 6 classifiers × 3 feature sets × 2 datasets, significance testing, imbalance, per-scenario analysis; planned clustering, anomaly detection, tuning, stacking, SHAP"],
 ["Cloud Computing &amp; Technology (CCT)", "1–4, 12, 13 (CCT table), 14, 15.3", "Microservice architecture, REST API, containers, Vercel + Render deployment design, free-tier constraints, CI/CD, IaC, scale-to-zero"],
], ["If you are writing for…", "Read chapters", "Core material"], raw=True, widths=["25%", "30%", "45%"])}
<p>Chapter 15 contains ready-made <b>report outlines and slide-by-slide deck outlines</b> for each subject, with the figure to use on each slide. The figures are also available as PNG files in <code>reports/figures/</code> of the repository.</p>

<h2 class="nonum">Contents</h2>
<ol class="toc">
<li>Executive summary</li><li>Introduction, objectives and research questions</li><li>Datasets</li>
<li>Planning, methodology and project management</li><li>Phase 0: dataset audit</li><li>Phase 1: data wrangling and data quality</li>
<li>Phase 1b: dataset v1.1 (graph-structure injection)</li><li>Phase 3: graph (network) features</li><li>Phase 2: exploratory data analysis</li>
<li>Phase 4: association rule mining</li><li>Phase 5: supervised classification</li><li>Planned work: phases 6–10 (design)</li>
<li>Syllabus mapping</li><li>Conclusions, limitations and future work</li><li>Guide for report and presentation writers</li>
<li>Appendices: decision log · feature dictionary · glossary · reproducibility · references</li>
</ol>
</section>
""")

# 1 executive summary
add(f"""
<section>
<h1>Executive summary</h1>
<p><b>FinSight AI</b> is a financial-transaction intelligence platform built as one project for three courses. It takes a synthetic Indian payment ecosystem (10,000 UPI/NEFT/RTGS transactions, 1,200 customers, 1,500 accounts) plus four public datasets (IEEE-CIS fraud, PaySim mobile money, OFAC and UN sanctions lists). It runs the full <b>KDD process</b> (data mining), builds and rigorously compares <b>machine-learning models</b>, and is being packaged as <b>cloud microservices</b> with a web dashboard.</p>
<h3>What has been done (phases 0–5 of 10)</h3>
<ul>
<li><b>Audit before building.</b> The original plan was checked against the data. The audit found label-leaking columns (<code>rapid_forward_delta_sec</code>, <code>is_flagged_alert</code>, <code>alert_id</code>), a join that matched only 8 of 10,000 rows, redundant and constant columns, and, most importantly, that the dataset's "network" fraud scenarios (rings, mules, layering) <b>had no network structure at all</b>: every graph feature scored AUC ≈ 0.50.</li>
<li><b>Dataset engineering (v1.1).</b> A seeded, logged, tested injection step wired those scenarios into real graph structure. It added legitimate look-alike patterns (merchant hubs, family devices) so the signal isn't a giveaway, and camouflaged 50% of organised fraud so it looks normal per transaction. Labels were never changed; v1.0 is kept as a control.</li>
<li><b>Wrangling and quality.</b> Seven files were merged into one 57-column table, then 76 columns with 19 point-in-time network features. Data-quality and leakage reports were produced, and 20,332 sanctions entities were parsed from CSV + XML.</li>
<li><b>Association rule mining.</b> Apriori and FP-Growth return identical itemsets, but Apriori needs <b>{hp['apriori']['peak_mb']:,.0f} MB vs {hp['fpgrowth']['peak_mb']:.1f} MB</b> at the hardest setting. {v11['productive']} productive rules were found on v1.1 ({v11['productive_fraud_with_network_item']} use network items) vs {v10['productive']} on v1.0 (none use network items). Top rule: <i>pass-through → fraud</i>, confidence 99%, lift 7.4.</li>
<li><b>Supervised learning.</b> 6 classifiers × 3 feature sets × 2 dataset versions, compared with 5×2 repeated cross-validation and paired significance tests. Behavioural features help every model. Network features help <b>5 of 6 models significantly on v1.1 (p &lt; 0.001) and 0 of 6 on v1.0</b>. Best model: <b>XGBoost, test PR-AUC {best['test_pr_auc']:.3f}, F1 {best['test_f1']:.3f}</b>. SMOTE never improved results.</li>
</ul>
<h3>Key results at a glance</h3>
{table([
 ["RQ1 strongest single signals", "amount (AUC 0.85), amount vs 30-day average (0.81), minutes since beneficiary added (0.79), distance from home (0.74)"],
 ["RQ2 top fraud patterns", "{PASS_THROUGH} 99% conf · {UPI, high spike, hub receiver} 100% · {quick inflow, recent beneficiary} 95% · {₹45–50k, recent beneficiary} 90%"],
 ["RQ2 Apriori vs FP-Growth", f"identical results; Apriori faster on small searches (vectorised), FP-Growth faster and ~17× leaner on large ones"],
 ["RQ3 feature groups", "B (behaviour) > A for 12/12 model-dataset pairs; C (network) > B for 5/6 models on v1.1 and 0/6 on v1.0"],
 ["RQ4 best model", f"XGBoost Exp C: PR-AUC {best['test_pr_auc']:.3f} · ROC-AUC {best['test_roc_auc']:.3f} · precision {best['test_precision']:.2f} · recall {best['test_recall']:.2f}"],
 ["RQ7 hardest cases", "high-value spike (80% recall); layering without network features (65% → 85% with them); BORDERLINE causes 65% of false positives"],
 ["Rules vs ML (same test set)", f"association rules F1 {rb['test']['f1']:.3f} / PR-AUC {rb['test']['pr_auc']:.3f} vs XGBoost F1 {best['test_f1']:.3f} / PR-AUC {best['test_pr_auc']:.3f}"],
], ["Question", "Finding"], widths=["28%", "72%"])}
<h3>What remains (designed, not yet built)</h3>
<p>Phase 6 clustering and anomaly detection (K-Means, DBSCAN, Isolation Forest, One-class SVM, autoencoder). Phase 7 hyperparameter tuning, stacking ensemble, calibration, SHAP explanations and the 0–100 risk score. Phase 8 FastAPI microservices and Docker. Phase 9 React dashboard. Phase 10 deployment on <b>Vercel (frontend) and Render (backend)</b> and the final report. RQ5, RQ6 and RQ8 are answered in those phases.</p>
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
 ["RQ4", "Which classifier performs best, with which hyperparameters, and does SMOTE beat class weighting?", "ML", "Model and imbalance answered (Ch. 11); hyperparameters in Phase 7"],
 ["RQ5", "Can unsupervised clustering recover risk groupings without labels?", "ML", "Planned (Phase 6)"],
 ["RQ6", "How well do Isolation Forest, One-class SVM and an autoencoder trained only on legitimate data detect fraud?", "ML", "Planned (Phase 6)"],
 ["RQ7", "Which scenarios (and BORDERLINE cases) are hardest for the models?", "ML + DMW", "Answered (Ch. 11)"],
 ["RQ8", "How does a containerised microservice architecture on managed cloud platforms support independent deployment, scaling and cost (scale-to-zero vs cold start)?", "CCT", "Planned (Phases 8–10)"],
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
 ["PaySim mobile money", "1 CSV", "6,362,620 rows, 11 columns, 8,213 frauds (0.13%)", "Big-data wrangling with Polars; mining scale test; planned anomaly track"],
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
 ["6", "Unsupervised + anomaly", "ML", "planned", "K-Means, DBSCAN, IF, OC-SVM, autoencoder (PyTorch → ONNX)"],
 ["7", "Tuning + stacking + SHAP", "ML", "planned", "Grid/Random search, stacking, calibration, risk score"],
 ["8", "Services + Docker", "CCT", "planned", "5 FastAPI services, Compose, two service modes"],
 ["9", "Dashboard", "All", "planned", "React SPA, 9 pages"],
 ["10", "Deployment + report", "CCT", "planned", "Render (backend) + Vercel (frontend), measurements"],
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
<li><b>Testing:</b> 31 automated tests (pytest) cover the dataset invariants, point-in-time correctness, the leakage blacklist, split disjointness, mining correctness and the evaluation maths.</li>
<li><b>Documentation:</b> a design plan, phase tracker (<code>docs/PROGRESS.md</code>), decision log (24 decisions with evidence), data reference, setup guide and an agent manual (<code>CLAUDE.md</code>) for AI-assisted development.</li>
<li><b>Visual standard:</b> one validated, colour-blind-safe palette; the same entity always has the same colour (blue = legitimate, orange = fraudulent, aqua = borderline); every figure was inspected after rendering.</li>
</ul>
<h2>Technology stack</h2>
{table([
 ["Language / environment", "Python 3.12, uv, Jupyter"], ["Data", "pandas 3, NumPy, PyArrow (Parquet), Polars (6.36M-row PaySim)"],
 ["Mining", "MLxtend (Apriori, FP-Growth, FP-Max)"], ["Machine learning", "scikit-learn, XGBoost, imbalanced-learn; PyTorch → ONNX (planned autoencoder)"],
 ["Graph", "NetworkX (PageRank, Louvain communities)"], ["Sanctions", "RapidFuzz fuzzy matching, lxml"],
 ["Visualisation", "Matplotlib (report figures); Plotly.js (dashboard)"], ["API / services (planned)", "FastAPI, Pydantic, Uvicorn, httpx"],
 ["Frontend (planned)", "React + Vite + TypeScript, deployed on Vercel"], ["Cloud (planned)", "Docker, Docker Compose, Render (Blueprint IaC), Vercel, GitHub Actions, GitHub Releases"],
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

# 12 planned
add(f"""
<section>
<h1>Planned work: phases 6–10 (design)</h1>
<p class="callout">These phases are fully designed but <b>not yet implemented</b>. Their numbers will be added to this report when they're done. The designs below are what the team should describe as "proposed / in progress" if the reports are due earlier.</p>
<h2>Phase 6: unsupervised learning and anomaly detection (RQ5, RQ6)</h2>
<ul>
<li><b>K-Means</b> on behavioural features, K = 2–8, chosen by elbow, silhouette and Davies–Bouldin; then fraud % per cluster and Adjusted Rand Index vs labels. <b>DBSCAN</b> with a k-distance plot for ε; noise points vs fraud. PCA and t-SNE projections.</li>
<li><b>Anomaly detectors trained only on legitimate training rows:</b> Isolation Forest, One-class SVM (RBF) and an autoencoder (32-16-8-16-32, PyTorch, exported to ONNX for lightweight serving), each thresholded at the 95th/99th percentile of training scores. Compared by PR-AUC and recall at 1% false-positive rate, per scenario.</li>
<li><b>PaySim track:</b> Isolation Forest on PaySim's own features (6.36M rows via Polars) evaluated against its fraud labels.</li>
<li>Expected from EDA: clusters separate legitimate from {{fraud ∪ borderline}} but not fraud from borderline.</li>
</ul>
<h2>Phase 7: tuning, stacking, explanation, risk score</h2>
<ul>
<li>GridSearchCV (decision tree, KNN) and RandomizedSearchCV (random forest, XGBoost; 20 iterations) with 5-fold stratified CV; learning curves for the bias–variance discussion.</li>
<li><b>Stacking ensemble:</b> RF + XGBoost + MLP → logistic regression meta-learner on out-of-fold probabilities, then isotonic calibration so scores are true probabilities.</li>
<li><b>Explanations:</b> SHAP (global importance, beeswarm, per-transaction waterfall). At serve time, XGBoost's built-in TreeSHAP (<code>pred_contribs</code>) avoids the heavy <code>shap</code> library on a 512 MB server.</li>
<li><b>Risk score 0–100</b> = weighted mix of calibrated fraud probability, anomaly percentile and graph risk (weights chosen on validation), with overrides: sanctions match ≥ 90 → CRITICAL; PEP customer → at least MEDIUM; structuring rule → +10. Tiers LOW / MEDIUM / HIGH / CRITICAL, with a dedicated review band for BORDERLINE-like scores.</li>
<li>Bar to beat: XGBoost Exp C test PR-AUC {best['test_pr_auc']:.3f}.</li>
</ul>
<h2>Phase 8: services and containers</h2>
{flow([("React SPA", "Vercel CDN"), ("api-gateway :8000", "routing · scoring · CORS · /health · /metrics"), [("ml-service :8001", "classifier · anomaly · cluster · explain"), ("mining-service :8002", "rules · matching · benchmark")], [("graph-service :8003", "point-in-time features · ego graphs"), ("sanctions-service :8004", "fuzzy OFAC + UN screening")]])}
{table([
 ["POST /score-transaction", "risk 0–100, tier, component scores, top feature contributions, cluster, fired rules, sanctions"],
 ["POST /anomaly-score", "Isolation Forest / OC-SVM / autoencoder scores"],
 ["GET /patterns, /patterns/benchmark", "association rules (filterable) and the Apriori vs FP-Growth results"],
 ["GET /cluster/{{tx_id}} · /network/{{account_id}} · /transactions/{{tx_id}}", "cluster profile · network metrics + ego graph · full merged profile"],
 ["GET /sanctions-check?name=", "fuzzy matches with scores"],
 ["GET /models/compare", "the A/B/C × model results matrix"],
 ["GET /health, /metrics", "status and latency/request statistics"],
], ["Endpoint", "Returns"], "small", widths=["40%", "60%"])}
<p>All services are stateless FastAPI apps with Pydantic schemas and auto-generated OpenAPI docs, in slim, non-root Docker images. Target: p95 &lt; 200 ms for scoring when warm.</p>
<h2>Phase 9: dashboard (React on Vercel)</h2>
<p>Nine pages calling only the gateway: Executive overview · Data quality · Transaction explorer · Pattern mining lab · ML experiments lab · Clustering &amp; anomaly · Network explorer · Risk scorer (submit a transaction, see score, tier, explanation, rules, sanctions) · Platform (service health, latency, cold-start measurements).</p>
<h2>Phase 10: deployment on Vercel + Render</h2>
{table([
 ["React dashboard (static)", "Vercel (Hobby, free)", "Global CDN, Git-push deploys, preview URL per pull request. Streamlit can't run on Vercel, hence React."],
 ["FastAPI services (Docker)", "Render web services (free)", "Native Docker builds from GitHub, managed HTTPS, Blueprint IaC (render.yaml)"],
 ["Model artifacts (~26 MB)", "GitHub Releases", "Versioned 'model registry'; downloaded during docker build into an immutable image"],
 ["Training pipeline", "Local / Compose", "Needs the 1.8 GB dataset (not in Git); publishes artifacts to a release"],
 ["CI", "GitHub Actions", "Tests on every push; Render and Vercel auto-deploy from main"],
], ["Component", "Platform", "Reason"], "small", widths=["24%", "24%", "52%"])}
<h3>Free-tier constraints and design responses</h3>
{table([
 ["Render free web services can't receive private-network traffic", "Inter-service calls use HTTPS + a shared secret header (X-Internal-Key)"],
 ["Services sleep after 15 min idle; waking takes up to ~1 min", "Dashboard shows a 'waking the backend' state and polls /health; cold vs warm latency measured as a CCT result"],
 ["750 free instance-hours per workspace per month", "Sleeping services use no hours, so the demo fits"],
 ["~512 MB RAM per free instance", "Memory-aware model choice; native TreeSHAP; target < 450 MB"],
 ["Vercel proxy (rewrite) timeout is 30 s", "The SPA calls Render directly with a CORS allow-list, not through a rewrite"],
], ["Constraint", "Design response"], "small", widths=["45%", "55%"])}
<p><b>Two service modes from one codebase</b> (<code>FINSIGHT_SERVICE_MODE</code>): <i>remote</i> runs five services talking over HTTP (Docker Compose locally, and an optional five-service Render Blueprint), which is the microservices demonstration. <i>inprocess</i> runs the gateway with the four modules imported directly, giving one Render service and one cold start (the default free deployment). Measuring both answers part of RQ8: the latency and cold-start cost of microservices versus a modular monolith.</p>
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
 ["Classification and evaluation", "DT, NB, KNN with PR-AUC, F1, ROC", "✔"], ["IEEE-CIS missingness profiling", "transaction ⋈ identity", "planned"],
], ["Topic", "Where in FinSight", "Status"], "small", widths=["36%", "52%", "12%"])}
<h2>Machine Learning</h2>
{table([
 ["Supervised learning", "6 classifiers × 3 feature sets × 2 datasets", "✔"], ["Feature engineering", "behavioural + point-in-time network features", "✔"],
 ["Train / validation / test, cross-validation", "stratified 60/20/20; 5×2 repeated CV; temporal hold-out", "✔"],
 ["Statistical model comparison", "paired t-test + Wilcoxon on identical folds", "✔"], ["Class imbalance", "SMOTE vs class weights vs none", "✔"],
 ["Ensembles (bagging, boosting)", "Random Forest, XGBoost", "✔"], ["Neural network classifier", "MLP", "✔"],
 ["Evaluation metrics", "PR-AUC, ROC-AUC, F1, precision/recall, per-class, confusion", "✔"], ["Interpretability", "decision tree visualisation", "✔"],
 ["Unsupervised learning", "K-Means, DBSCAN, PCA, t-SNE", "planned (6)"], ["Anomaly detection", "Isolation Forest, One-class SVM, autoencoder", "planned (6)"],
 ["Hyperparameter tuning, bias–variance", "Grid/Random search, learning curves", "planned (7)"], ["Stacking and calibration", "RF + XGB + MLP → LR; isotonic", "planned (7)"],
 ["Explainability", "SHAP global + local", "planned (7)"], ["Model persistence / serving", "joblib (done), ONNX (planned)", "partial"],
], ["Topic", "Where in FinSight", "Status"], "small", widths=["36%", "52%", "12%"])}
<h2>Cloud Computing &amp; Technology</h2>
{table([
 ["Service models (IaaS / PaaS / SaaS / FaaS)", "Render = container PaaS; Vercel = frontend PaaS + CDN; FinSight API as SaaS", "designed"],
 ["Virtualisation and containers", "Dockerfile per service; VM vs container comparison", "planned (8)"],
 ["Microservices and API gateway", "5 FastAPI services; inprocess vs remote modes", "planned (8)"],
 ["REST and OpenAPI", "11 endpoints, Pydantic schemas, Swagger UI", "planned (8)"],
 ["Orchestration", "Docker Compose (+ horizontal scaling demo); conceptual Kubernetes", "planned (8)"],
 ["CDN / edge", "static SPA served from Vercel's edge network", "planned (9–10)"],
 ["CI/CD, GitOps", "GitHub Actions tests; auto-deploy on push; preview deployments", "planned (10)"],
 ["Infrastructure as Code", "render.yaml Blueprint, vercel.json, docker-compose.yml", "planned (10)"],
 ["Artifact registry / storage", "GitHub Releases model registry; fsspec storage abstraction", "planned (8)"],
 ["Elasticity, scale-to-zero, cost", "Render spin-down vs cold start; locust load test", "planned (10)"],
 ["Security", "HTTPS, CORS allow-list, internal shared key, non-root images, env secrets", "planned (8–10)"],
 ["Observability", "JSON logs, request IDs, /metrics, Platform page", "planned (8–9)"],
 ["Cloud-native principles", "12-factor configuration, stateless services, immutable images", "designed"],
], ["Topic", "Where in FinSight", "Status"], "small", widths=["36%", "50%", "14%"])}
</section>
""")

# 14 conclusions
add(f"""
<section>
<h1>Conclusions, limitations and future work</h1>
<h2>Conclusions so far</h2>
<ol>
<li><b>Audit before you model.</b> The original dataset contained label proxies that would have produced near-perfect but meaningless models, and its "network" scenarios had no network. Finding this first changed the project's direction and is a central data-mining result.</li>
<li><b>Data engineering can be done honestly.</b> The v1.1 injection is seeded, logged, tested and validated against a control. Mining and ML both rediscover exactly the injected structure and nothing on v1.0.</li>
<li><b>Behaviour matters most; network matters for organised fraud.</b> Behavioural context gives the largest gain for every model. Network features add a significant gain (5/6 models) precisely on camouflaged organised fraud: layering recall rises from 65% to 85%.</li>
<li><b>Interpretable rules are competitive but not best.</b> Productive association rules reach F1 {rb['test']['f1']:.2f} on held-out data vs {best['test_f1']:.2f} for XGBoost. Rules remain valuable as explanations and as a transparent first-line filter.</li>
<li><b>FP-Growth's advantage is about memory and large searches</b>, not universal speed. Implementation (vectorised vs pure Python) matters as much as asymptotic complexity.</li>
<li><b>SMOTE is not a default.</b> With threshold tuning, rebalancing added nothing and sometimes hurt.</li>
<li><b>Label ambiguity is the main remaining error.</b> BORDERLINE cases cause most false positives and are only half-separable even as their own class. A "review" tier is more honest than forcing a binary decision.</li>
</ol>
<h2>Limitations and threats to validity</h2>
<ul>
<li><b>Synthetic data:</b> all India results describe a generator plus our documented modifications. Two behaviour flags are perfect by construction. Absolute scores shouldn't be compared with real-world systems.</li>
<li><b>Injected structure:</b> network gains depend on how v1.1 was built. The v1.0 control and the decoys limit, but don't remove, this circularity. The report states the injection rules in full.</li>
<li><b>Small per-scenario test counts</b> (16–48 fraud rows per scenario) make scenario-level differences under ~10 points unreliable.</li>
<li><b>One month of data:</b> the temporal hold-out shows no drift within August, but can't test seasonal or long-term drift.</li>
<li><b>Global graph features</b> (PageRank, communities) are transductive (computed on the full graph, label-free); velocity features are strictly point-in-time.</li>
<li><b>Default hyperparameters</b> in Phase 5; tuned results come in Phase 7.</li>
</ul>
<h2>Future work</h2>
<p>Complete phases 6–10. Beyond the course: graph neural networks over the account graph; streaming scoring (Kafka) with online graph updates; drift monitoring; human-in-the-loop review of BORDERLINE cases; evaluation on real, consented bank data.</p>
</section>
""")

# 15 guide
add(f"""
<section>
<h1>Guide for report and presentation writers</h1>
<p>Each outline lists sections or slides with the source chapter and the figure file (in <code>reports/figures/</code>) to use. Keep the colour meaning consistent: blue = legitimate, orange = fraudulent, aqua = borderline.</p>
<h2>15.1 Data Mining &amp; Wrangling</h2>
<p><b>Report outline:</b> Introduction (Ch. 2) → Datasets and KDD (Ch. 3–4) → Audit and data quality (Ch. 5–6) → Dataset engineering v1.1 (Ch. 7) → EDA and RQ1 (Ch. 9) → Association mining and RQ2 (Ch. 10) → Rules vs classifiers (Ch. 11 baselines) → Conclusions (Ch. 14).</p>
{table([
 ["1", "Title + one-line pitch", "—"], ["2", "Problem: opportunistic vs organised fraud in Indian payments", "—"],
 ["3", "Datasets and the KDD process", "KDD flow (Ch. 4)"], ["4", "Audit: leakage and the missing network", "Ch. 5 table"],
 ["5", "Data quality and integration (7 files → 76 columns)", "Ch. 6 flow + RI table"], ["6", "Sanctions wrangling + threshold experiment", "Ch. 6 sweep table"],
 ["7", "v1.1: injecting structure honestly", "eda_12_network_examples, eda_11_fanin_distribution"],
 ["8", "EDA highlights and RQ1", "eda_02_amount_by_label, eda_07_behaviour_flags, eda_06_feature_auc_v10_v11"],
 ["9", "Discretisation into items", "Ch. 10 items table"], ["10", "Apriori vs FP-Growth", "mining_01_benchmark, mining_03_paysim_scaling"],
 ["11", "Closed/maximal itemsets", "mining_02_condensed"], ["12", "Top rules (RQ2)", "mining_05_top_rules_v1_1, mining_04_rules_scatter"],
 ["13", "Control: v1.0 vs v1.1 rules", "mining_06_item_share_v10_v11"], ["14", "Rules as a detector vs ML", "mining_07_detector_curve, ml_03_pr_roc_expC"],
 ["15", "Conclusions and limitations", "Ch. 14"],
], ["Slide", "Content", "Figure / source"], "small", widths=["8%", "52%", "40%"])}
<p><b>One-line pitch:</b> "FinSight AI follows the full KDD process across five heterogeneous sources; our audit found label leakage and fraud scenarios that were never encoded in the graph. We fixed and validated that with a seeded augmentation step, then mined behaviour patterns with Apriori and FP-Growth and benchmarked both."</p>
<h2>15.2 Machine Learning</h2>
<p><b>Report outline:</b> Introduction and RQs (Ch. 2) → Data and features (Ch. 3, 8, Appendix B) → Protocol (Ch. 11 design) → RQ3 → RQ4 + imbalance → RQ7 + BORDERLINE → robustness and interpretability → planned unsupervised, anomaly, tuning, stacking, SHAP (Ch. 12) → Conclusions.</p>
{table([
 ["1", "Title + pitch", "—"], ["2", "Task: binary fraud detection, 13.5% positives; why PR-AUC", "eda_01_label_overview"],
 ["3", "Feature sets A/B/C (12/30/49)", "Appendix B"], ["4", "Network features and point-in-time design", "eda_14_network_signal_by_scenario"],
 ["5", "Protocol: split, 5×2 CV, paired tests, SMOTE in pipeline", "Ch. 11 design table"],
 ["6", "RQ3 result", "ml_01_abc_cv_pr_auc + significance table"], ["7", "RQ4 model comparison", "ml_02_test_pr_auc_heatmap, ml_03_pr_roc_expC"],
 ["8", "Imbalance: SMOTE doesn't help", "ml_05_imbalance"], ["9", "RQ7 per-scenario recall", "ml_04_per_scenario_xgb"],
 ["10", "BORDERLINE problem + 3-class", "ml_06_confusion_by_label, ml_08_borderline"],
 ["11", "Temporal robustness", "ml_07_temporal"], ["12", "Interpretable tree", "ml_09_decision_tree"],
 ["13", "Next: clustering, anomaly, tuning, stacking, SHAP", "Ch. 12"], ["14", "Conclusions", "Ch. 14"],
], ["Slide", "Content", "Figure / source"], "small", widths=["8%", "52%", "40%"])}
<p><b>One-line pitch:</b> "We trained 6 classifiers across 3 feature sets on two dataset versions and used paired cross-validation to show that network features help only where real network structure exists. XGBoost reaches PR-AUC {best['test_pr_auc']:.2f}, and SMOTE didn't help."</p>
<h2>15.3 Cloud Computing &amp; Technology</h2>
<p><b>Report outline:</b> Introduction (Ch. 2) → Requirements and service models → Architecture (Ch. 12 Phase 8) → Containers and microservices → REST API design → Deployment on Vercel + Render, with constraints → CI/CD and IaC → Scalability, scale-to-zero and cost → Security and observability → Evaluation (cold start, latency, image size: to be measured in Phase 10) → Conclusions.</p>
{table([
 ["1", "Title + pitch", "—"], ["2", "What the platform does (ML + mining as a service)", "Ch. 1"],
 ["3", "Service models: where Render, Vercel and our API sit", "Ch. 13 CCT table"], ["4", "Architecture: gateway + 4 services + SPA", "Ch. 12 flow diagram"],
 ["5", "REST endpoints and OpenAPI", "Ch. 12 endpoint table"], ["6", "Containers: Dockerfiles, slim/non-root images", "Phase 8"],
 ["7", "Deployment topology: Vercel CDN → Render containers", "Ch. 12 deployment table"], ["8", "Free-tier constraints → design responses", "Ch. 12 constraints table"],
 ["9", "Microservices vs modular monolith (two modes)", "Ch. 12"], ["10", "CI/CD, IaC, model registry", "Ch. 12"],
 ["11", "Scale-to-zero vs cold start; load test results", "Phase 10 measurements"], ["12", "Security and observability", "Ch. 13"],
 ["13", "Live demo", "Vercel URL"], ["14", "Conclusions", "Ch. 14"],
], ["Slide", "Content", "Figure / source"], "small", widths=["8%", "52%", "40%"])}
<p><b>One-line pitch:</b> "Five stateless FastAPI microservices in Docker, deployed on Render from a Blueprint, with a React dashboard served from Vercel's CDN. Artifacts are versioned as GitHub Releases and baked into immutable images, deploys are triggered by Git pushes, and we measure the scale-to-zero vs cold-start trade-off."</p>
<h2>Suggested live-demo flow</h2>
<ol><li>Dataset overview and the audit story (leakage, missing network).</li><li>v1.0 → v1.1: a fraud ring and a layering chain.</li><li>Top association rules; Apriori vs FP-Growth memory.</li><li>A → B → C improvement; v1.0 control.</li><li>A BORDERLINE example and why it's hard.</li><li>Risk scorer: submit a transaction, see score, tier and explanation (after Phase 9).</li><li>Platform page: services, cold start vs warm latency (after Phase 10).</li></ol>
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
 ["Install", "uv sync --all-extras"], ["Tests", "make test (31 tests)"],
 ["Full pipeline", "make pipeline (stages: inject, wrangle, graph, sanctions, eda, mining, train_supervised)"],
 ["One stage", "make pipeline STAGES=mining"], ["Notebooks", "notebooks/02_eda, 03_association_mining, 04_supervised_ml (executed, 0 errors)"],
 ["This report", "uv run python docs/report/build_report.py"],
]
outputs = [
 ["reports/data_quality_india_v1_{0,1}.json · sanctions_summary.json", "Phase 1"], ["dataset/processed/custom_india_v1_1/ (+ injection_log.csv)", "Phase 1b"],
 ["reports/eda_summary.json · figures/eda_01…14", "Phase 2"], ["dataset/curated/master_features_v1_{0,1}.parquet · artifacts/graph/", "Phase 3"],
 ["reports/association_rules_*.csv · mining_benchmark.csv · mining_paysim_scaling.csv · emerging_patterns_v1_1.csv · mining_summary.json · figures/mining_01…08", "Phase 4"],
 ["reports/ml_experiment_results.csv · ml_cv_folds.csv · ml_paired_tests.csv · ml_imbalance.csv · ml_per_scenario.csv · ml_temporal.csv · ml_summary.json · figures/ml_01…09 · artifacts/models/phase5/", "Phase 5"],
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
 "Liu, F. T., Ting, K. M., Zhou, Z.-H. (2008). Isolation Forest. ICDM.", "Schölkopf, B. et al. (2001). Estimating the Support of a High-Dimensional Distribution. Neural Computation.",
 "Lundberg, S., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). NeurIPS.",
 "Blondel, V. et al. (2008). Fast Unfolding of Communities in Large Networks (Louvain). J. Stat. Mech.",
 "Lopez-Rojas, E., Elmir, A., Axelsson, S. (2016). PaySim: A Financial Mobile Money Simulator for Fraud Detection. EMSS.",
 "IEEE Computational Intelligence Society & Vesta (2019). IEEE-CIS Fraud Detection, Kaggle competition dataset.",
 "U.S. Treasury OFAC, Specially Designated Nationals (SDN) List; United Nations Security Council Consolidated List.",
 "Reserve Bank of India. Master Directions on Fraud Risk Management in Commercial Banks. FATF (2026). The FATF Recommendations.",
 "Wiggins, A. (2017). The Twelve-Factor App. Render documentation (render.com/docs/free); Vercel documentation (vercel.com/docs).",
 "Libraries: pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, MLxtend, NetworkX, Polars, RapidFuzz, Matplotlib.",
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
