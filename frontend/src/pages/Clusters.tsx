import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { fmt, LABELS, useApi } from "../api";
import { css, labelColor, Plot } from "../components/Plot";
import { Async, Card, PageHead, Seg } from "../components/ui";

type Pt = { transaction_id: string; pc1: number; pc2: number; cluster: number; risk_label: string; scenario: string };
type Clusters = {
  sweep: { k: number; inertia: number; silhouette: number; davies_bouldin: number; exp: string }[];
  clusters: { cluster: number; n: number; fraud_rate: number; share_LEGITIMATE: number; share_FRAUDULENT: number; share_BORDERLINE: number; top_scenario: string; exp: string; method: string }[];
  profiles: { cluster: number; size: number; fraud_rate: number; top_features: { feature: string; z: number }[] }[];
  embedding: { method: string; explained_variance: number[]; points: Pt[]; sampled_from: number; note: string };
  detectors: Record<string, number | string>[];
};

const CLUSTER_NAME: Record<number, string> = { 0: "Velocity bursts", 1: "Normal traffic", 2: "Unusual amounts, community flows", 3: "Fan-in hubs" };
const DET = ["isolation_forest", "ocsvm_nu0.01", "ocsvm_nu0.05", "autoencoder"];
const DET_NAME: Record<string, string> = { isolation_forest: "Isolation Forest", "ocsvm_nu0.01": "OC-SVM ν=0.01", "ocsvm_nu0.05": "OC-SVM ν=0.05", autoencoder: "Autoencoder" };
const clusterColor = (c: number) => css(`--s${4 + c}`);     // slots 4–7: never confused with the label colours

export default function ClustersPage() {
  const st = useApi<Clusters>("/clusters");
  const [colorBy, setColorBy] = useState<"label" | "cluster">("label");
  const [metric, setMetric] = useState<"pr_auc" | "p95_borderline_flag_rate">("pr_auc");
  const nav = useNavigate();
  return (
    <>
      <PageHead title="Clustering & anomaly detection" sub="Phase 6: what can be found without labels. K-Means groups transactions; detectors trained only on legitimate behaviour flag the unusual." />
      <Async state={st}>{(d) => {
        const pts = d.embedding.points;
        const prof = [...d.profiles].sort((a, b) => a.fraud_rate - b.fraud_rate);
        return (
          <div className="stack">
            <div className="grid g3">
              <Card className="span2" title="Transactions in 2-D (PCA of the scaled feature space)"
                    aside={<Seg value={colorBy} onChange={setColorBy} options={[{ value: "label", label: "Colour by label" }, { value: "cluster", label: "Colour by cluster" }]} />}
                    note={`${d.embedding.note}; ${fmt.int(pts.length)} of ${fmt.int(d.embedding.sampled_from)} shown. PC1 + PC2 keep ${fmt.pct(d.embedding.explained_variance[0] + d.embedding.explained_variance[1], 0)} of the variance. Click a point to score it.`}>
                <Plot height={440} onClick={(p) => p.customdata && nav(`/score?tx=${p.customdata}`)} data={() => {
                  const groups = colorBy === "label" ? LABELS.map((l) => ({ key: l, name: l.charAt(0) + l.slice(1).toLowerCase(), sel: (p: Pt) => p.risk_label === l, color: labelColor(l) }))
                    : [1, 2, 0, 3].map((c) => ({ key: String(c), name: `Cluster ${c}: ${CLUSTER_NAME[c]}`, sel: (p: Pt) => p.cluster === c, color: clusterColor(c) }));
                  return groups.map((g) => {
                    const r = pts.filter(g.sel);
                    return { type: "scattergl", mode: "markers", name: `${g.name} (${fmt.int(r.length)})`, x: r.map((p) => p.pc1), y: r.map((p) => p.pc2),
                      marker: { size: 5, color: g.color, opacity: g.key === "LEGITIMATE" || g.key === "1" ? 0.45 : 0.8 },
                      customdata: r.map((p) => p.transaction_id), text: r.map((p) => p.scenario),
                      hovertemplate: "%{customdata}<br>%{text}<extra>" + g.name + "</extra>" };
                  });
                }} layout={() => ({ hovermode: "closest", hoverdistance: 12, xaxis: { title: { text: "PC1" }, zeroline: false }, yaxis: { title: { text: "PC2" }, zeroline: false }, legend: { orientation: "h", y: -0.14, x: 0 } })} />
              </Card>
              <Card title="K-Means clusters (K = 4)" note="Found without labels; labels attached afterwards. Named by their most extreme centroid features (σ).">
                {prof.map((p) => (
                  <div key={p.cluster} style={{ padding: "8px 0", borderBottom: "1px solid var(--grid)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: 8 }}>
                      <b className="small"><span className="dot" style={{ background: `var(--s${4 + p.cluster})`, marginRight: 6 }} />{p.cluster} · {CLUSTER_NAME[p.cluster]}</b>
                      <span className="small num">{fmt.int(p.size)} rows</span>
                    </div>
                    <div className="small ink2">fraud <b>{fmt.pct(p.fraud_rate, 0)}</b> · {p.top_features.map((f) => `${fmt.feature(f.feature)} ${f.z > 0 ? "+" : ""}${f.z}σ`).join(" · ")}</div>
                    <div className="bar-inline" style={{ marginTop: 4 }}><span style={{ width: `${100 * p.fraud_rate}%`, background: "var(--s2)" }} /></div>
                  </div>
                ))}
              </Card>
            </div>
            <div className="grid g3">
              {(["silhouette", "davies_bouldin"] as const).map((m) => (
                <Card key={m} title={m === "silhouette" ? "Silhouette (higher is better)" : "Davies–Bouldin (lower is better)"}
                      note={m === "silhouette" ? "Low and flat: cluster structure is weak overall." : "Chosen K: lowest DB on the silhouette plateau (K = 4 for Exp C)."}>
                  <Plot height={240} data={() => ["B", "C"].map((e, i) => {
                    const r = d.sweep.filter((x) => x.exp === e);
                    // neutral inks: on this page the categorical hues already mean labels (1–3) and clusters (4–7)
                    return { type: "scatter", mode: "lines+markers", name: e === "B" ? "Exp B (no network)" : "Exp C (+ network)", x: r.map((x) => x.k), y: r.map((x) => x[m]),
                      line: { color: css(i ? "--ink" : "--muted"), width: 2, dash: i ? "solid" : "dash" }, hovertemplate: `K=%{x}: <b>%{y:.3f}</b><extra>Exp ${e}</extra>` };
                  })} layout={() => ({ xaxis: { title: { text: "K" }, dtick: 1 }, legend: { orientation: "h", y: 1.18, x: 0 } })} />
                </Card>
              ))}
              <Card title="Anomaly detectors" aside={<Seg value={metric} onChange={setMetric} options={[{ value: "pr_auc", label: "PR-AUC" }, { value: "p95_borderline_flag_rate", label: "Borderline" }]} />}
                    note={metric === "pr_auc" ? "Trained on legitimate rows only. Supervised XGBoost reaches 0.885." : "Share of BORDERLINE rows flagged at the p95 threshold: unusual is not the same as fraudulent."}>
                <Plot height={240} data={() => {
                  const r = DET.map((k) => d.detectors.find((x) => x.detector === k && x.exp === "C")!);
                  return [{ type: "bar", x: DET.map((k) => DET_NAME[k]), y: r.map((x) => Number(x[metric])), marker: { color: css("--ink-2") },
                    text: r.map((x) => Number(x[metric]).toFixed(2)), textposition: "outside", cliponaxis: false,
                    hovertemplate: "%{x}: <b>%{y:.3f}</b><extra></extra>" }];
                }} layout={() => ({ showlegend: false, yaxis: { range: [0, 1.05], title: { text: metric === "pr_auc" ? "PR-AUC (test, Exp C)" : "BORDERLINE flagged at p95" } } })} />
              </Card>
            </div>
          </div>
        );
      }}</Async>
    </>
  );
}
