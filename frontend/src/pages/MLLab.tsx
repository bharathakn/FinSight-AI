import { useState } from "react";
import { fmt, useApi } from "../api";
import { css, Plot } from "../components/Plot";
import { Async, Card, PageHead, Seg, Stat, Table } from "../components/ui";

type Row = Record<string, number | string>;
type Compare = {
  production: { production_model: string; decision_threshold: number; test: Record<string, number>; candidates_val_pr_auc: Record<string, number>; artifact_sizes_mb: Record<string, number> };
  phase5_grid: Row[]; phase5_paired_tests: Row[]; phase7_final: Row[]; phase7_tuning: Row[];
  shap_importance: { feature: string; mean_abs_shap: number }[]; per_scenario: { scenario: string; risk_label: string; flag_rate: number; n: number }[];
  learning_curves: { n_train: number; train_mean: number; train_std: number; cv_mean: number; cv_std: number; model: string }[];
};

const MODELS = ["decision_tree", "naive_bayes", "knn", "random_forest", "xgboost", "mlp"];
const NAME: Record<string, string> = { decision_tree: "Decision Tree", naive_bayes: "Naive Bayes", knn: "KNN", random_forest: "Random Forest", xgboost: "XGBoost", mlp: "MLP",
  stack_uncalibrated: "Stack (RF+XGB+MLP→LR)", xgboost_calibrated: "XGBoost + isotonic", stack_calibrated: "Stack + isotonic" };
const EXP = { A: "A · transaction", B: "B · + behavioural", C: "C · + network" } as Record<string, string>;
const modelColor = (m: string) => { const i = MODELS.indexOf(m); return i >= 0 ? css(`--s${i + 1}`) : css("--ink"); };

export default function MLLabPage() {
  const st = useApi<Compare>("/models/compare");
  const [version, setVersion] = useState<"v1_1" | "v1_0">("v1_1");
  const [lcModel, setLcModel] = useState<"xgboost" | "decision_tree">("xgboost");
  return (
    <>
      <PageHead title="ML experiments lab" sub="Phase 5: 6 classifiers × 3 feature sets × 2 dataset versions with 5×2 cross-validation. Phase 7: tuning, stacking, calibration and explanations. All numbers are recorded results, not live training." />
      <Async state={st}>{(c) => {
        const grid = c.phase5_grid.filter((r) => r.version === version);
        const sig = c.phase5_paired_tests.filter((r) => r.version === version && r.comparison === "C vs B");
        const prodTest = c.production.test;
        const fin = [...c.phase7_final].sort((a, b) => Number(a.test_pr_auc) - Number(b.test_pr_auc));
        const scen = c.per_scenario.filter((r) => r.risk_label === "FRAUDULENT").sort((a, b) => a.flag_rate - b.flag_rate);
        const shap = [...c.shap_importance].slice(0, 12).reverse();
        const lc = c.learning_curves.filter((r) => r.model === lcModel);
        return (
          <div className="stack">
            <div className="grid g4">
              <Stat label="Production model" value={NAME[c.production.production_model] ?? c.production.production_model} hint={`chosen on validation · ${c.production.artifact_sizes_mb["production_model.joblib"]} MB`} />
              <Stat label="Test PR-AUC" value={fmt.num(prodTest.test_pr_auc)} hint={`ROC-AUC ${fmt.num(prodTest.test_roc_auc)}`} />
              <Stat label="Test F1" value={fmt.num(prodTest.test_f1)} hint={`precision ${fmt.pct(prodTest.test_precision, 0)} · recall ${fmt.pct(prodTest.test_recall, 0)}`} />
              <Stat label="Decision threshold" value={fmt.num(c.production.decision_threshold, 2)} hint="max-F1 on validation, never on test" />
            </div>
            <div className="filters" style={{ marginBottom: 0 }}>
              <span className="field">Dataset version
                <Seg value={version} onChange={setVersion} options={[{ value: "v1_1", label: "v1.1 (network structure injected)" }, { value: "v1_0", label: "v1.0 (control)" }]} /></span>
            </div>
            <div className="grid g2">
              <Card title="RQ3: does each feature group help?" aside="5×2 CV PR-AUC, mean ± sd"
                    note={`C vs B significant (paired t-test, p < 0.05) for ${sig.filter((r) => Number(r.t_pvalue) < 0.05 && Number(r.mean_diff) > 0).length} of 6 models on ${version === "v1_1" ? "v1.1" : "v1.0"}.`}>
                <Plot height={320} data={() => MODELS.map((m) => {
                  const r = ["A", "B", "C"].map((e) => grid.find((x) => x.model === m && x.exp === e)!);
                  return { type: "scatter", mode: "lines+markers", name: NAME[m], x: ["A", "B", "C"].map((e) => EXP[e]), y: r.map((x) => x.cv_pr_auc_mean),
                    error_y: { type: "data", array: r.map((x) => x.cv_pr_auc_std), thickness: 1, width: 3, color: modelColor(m) },
                    line: { color: modelColor(m), width: 2 }, marker: { size: 7 }, hovertemplate: NAME[m] + " · %{x}<br>PR-AUC <b>%{y:.3f}</b><extra></extra>" };
                })} layout={() => ({ yaxis: { title: { text: "PR-AUC (cross-validated)" } }, legend: { orientation: "h", y: -0.18, x: 0 } })} />
              </Card>
              <Card title="Held-out test PR-AUC" aside={version === "v1_1" ? "v1.1" : "v1.0"}>
                <Plot height={320} data={() => {
                  const z = MODELS.map((m) => ["A", "B", "C"].map((e) => Number(grid.find((x) => x.model === m && x.exp === e)!.test_pr_auc)));
                  const seq = [css("--seq-1"), css("--seq-3"), css("--seq-5"), css("--seq-7"), css("--seq-9"), css("--seq-11")];
                  return [{ type: "heatmap", z, x: ["A", "B", "C"].map((e) => EXP[e]), y: MODELS.map((m) => NAME[m]), zmin: 0.4, zmax: 0.95,
                    colorscale: seq.map((col, i) => [i / (seq.length - 1), col]), xgap: 2, ygap: 2,
                    hovertemplate: "%{y} · %{x}<br>test PR-AUC <b>%{z:.3f}</b><extra></extra>", colorbar: { thickness: 10, outlinewidth: 0 } }];
                }} layout={() => {
                  // cell labels: ink chosen per cell from its position on the ramp (the ramp runs the other way in dark mode)
                  const dark = getComputedStyle(document.documentElement).colorScheme === "dark";
                  const ann = MODELS.flatMap((m) => ["A", "B", "C"].map((e) => {
                    const v = Number(grid.find((x) => x.model === m && x.exp === e)!.test_pr_auc);
                    const t = (v - 0.4) / 0.55;
                    return { x: EXP[e], y: NAME[m], text: v.toFixed(3), showarrow: false,
                      font: { size: 12, color: (dark ? t > 0.55 : t < 0.55) ? "#0b0b0b" : "#ffffff" } };
                  }));
                  return { showlegend: false, annotations: ann, yaxis: { autorange: "reversed", gridcolor: "rgba(0,0,0,0)" }, xaxis: { gridcolor: "rgba(0,0,0,0)" } };
                }} />
              </Card>
            </div>
            <div className="grid g2">
              <Card title="Phase 7: tuning" note="Grid search (DT, KNN) and randomized search (RF, XGBoost, MLP), 5-fold CV on the training split, same folds for default and tuned.">
                <Plot height={300} data={() => [
                  { type: "bar", name: "Default (Phase 5)", y: c.phase7_tuning.map((r) => NAME[String(r.model)]), x: c.phase7_tuning.map((r) => r.default_cv_pr_auc), orientation: "h", marker: { color: css("--baseline") },
                    hovertemplate: "%{y} default: <b>%{x:.3f}</b><extra></extra>" },
                  { type: "bar", name: "Tuned", y: c.phase7_tuning.map((r) => NAME[String(r.model)]), x: c.phase7_tuning.map((r) => r.tuned_cv_pr_auc), orientation: "h", marker: { color: css("--s1") },
                    customdata: c.phase7_tuning.map((r) => [Number(r.paired_mean_diff), Number(r.paired_t_pvalue)]),
                    hovertemplate: "%{y} tuned: <b>%{x:.3f}</b> (%{customdata[0]:+.3f}, p=%{customdata[1]:.3f})<extra></extra>" },
                ]} layout={() => ({ barmode: "group", xaxis: { range: [0.7, 0.95], title: { text: "CV PR-AUC" } }, legend: { orientation: "h", y: 1.12, x: 0 } })} />
              </Card>
              <Card title="Final candidates (test, scored once)" note="Stacking did not beat tuned XGBoost; the calibrated XGBoost was chosen on validation (D-031).">
                <Plot height={300} data={() => [{ type: "bar", orientation: "h", y: fin.map((r) => NAME[String(r.model)] ?? r.model), x: fin.map((r) => r.test_pr_auc),
                  marker: { color: fin.map((r) => (r.model === c.production.production_model ? css("--s2") : css("--s1"))) },
                  text: fin.map((r) => Number(r.test_pr_auc).toFixed(3) + (r.model === c.production.production_model ? "  production" : "")), textposition: "outside", cliponaxis: false,
                  customdata: fin.map((r) => [Number(r.test_f1), Number(r.val_pr_auc)]),
                  hovertemplate: "%{y}<br>test PR-AUC <b>%{x:.3f}</b> · F1 %{customdata[0]:.3f} · val %{customdata[1]:.3f}<extra></extra>" }]}
                  layout={() => ({ showlegend: false, xaxis: { range: [0.7, 0.98] }, margin: { l: 10, r: 80, t: 6, b: 10 } })} />
              </Card>
            </div>
            <div className="grid g3">
              <Card title="What drives predictions (SHAP)" note="Mean |SHAP| in log-odds on the test set (tuned XGBoost).">
                <Plot height={340} data={() => [{ type: "bar", orientation: "h", y: shap.map((r) => fmt.feature(r.feature)), x: shap.map((r) => r.mean_abs_shap),
                  marker: { color: css("--s5") }, hovertemplate: "%{y}<br>mean |SHAP| <b>%{x:.3f}</b><extra></extra>" }]} layout={() => ({ showlegend: false })} />
              </Card>
              <Card title="Recall per fraud scenario" note="Production model on the test set (validation threshold). Test counts are 16–48 per scenario.">
                <Plot height={340} data={() => [{ type: "bar", orientation: "h", y: scen.map((r) => r.scenario), x: scen.map((r) => r.flag_rate), marker: { color: css("--s2") },
                  text: scen.map((r) => fmt.pct(r.flag_rate, 0)), textposition: "outside", cliponaxis: false, customdata: scen.map((r) => r.n),
                  hovertemplate: "%{y}<br>recall <b>%{x:.0%}</b> of %{customdata} fraud rows<extra></extra>" }]}
                  layout={() => ({ showlegend: false, xaxis: { range: [0, 1.15], tickformat: ".0%" }, margin: { l: 10, r: 40, t: 6, b: 10 } })} />
              </Card>
              <Card title="Learning curve" aside={<Seg value={lcModel} onChange={setLcModel} options={[{ value: "xgboost", label: "XGBoost" }, { value: "decision_tree", label: "Tree" }]} />}
                    note="Gap between training and CV = variance; a rising CV curve means more data would still help.">
                <Plot height={300} data={() => [["train", "Training", "--s1"], ["cv", "Cross-validation", "--s2"]].map(([k, n, col]) => ({
                  type: "scatter", mode: "lines+markers", name: n, x: lc.map((r) => r.n_train), y: lc.map((r) => (r as unknown as Record<string, number>)[`${k}_mean`]),
                  line: { color: css(col), width: 2 }, hovertemplate: `${n}: <b>%{y:.3f}</b> at %{x:,} rows<extra></extra>` }))}
                  layout={() => ({ xaxis: { title: { text: "training rows" } }, yaxis: { range: [0.5, 1.02], title: { text: "PR-AUC" } }, legend: { orientation: "h", y: 1.12, x: 0 } })} />
              </Card>
            </div>
            <Card title={`All ${version === "v1_1" ? "v1.1" : "v1.0"} configurations`}>
              <Table rows={grid} columns={[
                { key: "model", label: "Model", render: (r) => NAME[String(r.model)] }, { key: "exp", label: "Features", render: (r) => EXP[String(r.exp)] },
                { key: "cv", label: "CV PR-AUC", right: true, render: (r) => `${Number(r.cv_pr_auc_mean).toFixed(3)} ± ${Number(r.cv_pr_auc_std).toFixed(3)}` },
                { key: "test_pr_auc", label: "Test PR-AUC", right: true, render: (r) => Number(r.test_pr_auc).toFixed(3) },
                { key: "test_f1", label: "Test F1", right: true, render: (r) => Number(r.test_f1).toFixed(3) },
                { key: "test_recall", label: "Recall", right: true, render: (r) => fmt.pct(Number(r.test_recall), 0) },
              ]} />
            </Card>
          </div>
        );
      }}</Async>
    </>
  );
}
