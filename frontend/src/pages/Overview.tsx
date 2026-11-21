import { Link } from "react-router-dom";
import { fmt, useApi } from "../api";
import { css, Plot } from "../components/Plot";
import { Async, Card, PageHead, Stat, Table, TierBadge } from "../components/ui";

type Overview = {
  transactions: number; accounts: number; customers: number; labels: Record<string, number>; fraud_rate: number;
  amount_total_inr: number; period: [string, string];
  by_channel: { channel: string; n: number; fraud_rate: number; amount_median: number }[];
  daily: { day: string; n: number; fraud: number }[];
  scenarios: { scenario: string; n: number; fraud: number }[];
  model: { version: string; weights: Record<string, number>; cutoffs: Record<string, number> };
};
type Compare = { production: { production_model: string; decision_threshold: number; test: Record<string, number> } };

export default function OverviewPage() {
  const ov = useApi<Overview>("/overview");
  const cmp = useApi<Compare>("/models/compare");
  return (
    <>
      <PageHead title="Executive overview"
        sub="Synthetic Indian payments (UPI · NEFT · RTGS), August 2026. FinSight scores every transaction 0–100 with a calibrated model, an anomaly detector and network red flags, and explains why." />
      <Async state={ov} what="Loading overview">{(o) => (
        <div className="stack">
          <div className="grid g4">
            <Stat label="Transactions" value={fmt.int(o.transactions)} hint={`${fmt.date(o.period[0]).split(",")[0]} – ${fmt.date(o.period[1]).split(",")[0]} · ${fmt.int(o.accounts)} accounts`} />
            <Stat label="Fraud rate" value={fmt.pct(o.fraud_rate)} hint={`${fmt.int(o.labels.FRAUDULENT)} fraudulent · ${fmt.int(o.labels.BORDERLINE)} borderline`} />
            <Stat label="Value moved" value={"₹" + (o.amount_total_inr / 1e7).toFixed(1) + " Cr"} hint="sum of all transaction amounts" />
            {cmp.data ? (
              <Stat label="Production model (test)" value={`PR-AUC ${fmt.num(cmp.data.production.test.test_pr_auc)}`}
                hint={`F1 ${fmt.num(cmp.data.production.test.test_f1)} · precision ${fmt.pct(cmp.data.production.test.test_precision, 0)} · recall ${fmt.pct(cmp.data.production.test.test_recall, 0)}`} />
            ) : <Stat label="Production model (test)" value="…" />}
          </div>
          <div className="grid g2">
            <Card title="Daily transactions" aside="stacked: fraud vs not fraud">
              <Plot height={280} data={() => [
                { type: "bar", name: "Not fraud", x: o.daily.map((d) => d.day), y: o.daily.map((d) => d.n - d.fraud), marker: { color: css("--s1") }, hovertemplate: "%{x}<br><b>%{y}</b> not fraud<extra></extra>" },
                { type: "bar", name: "Fraud", x: o.daily.map((d) => d.day), y: o.daily.map((d) => d.fraud), marker: { color: css("--s2") }, hovertemplate: "%{x}<br><b>%{y}</b> fraud<extra></extra>" },
              ]} layout={() => ({ barmode: "stack", bargap: 0.15, yaxis: { title: { text: "transactions" } }, legend: { orientation: "h", y: 1.12, x: 0 } })} />
            </Card>
            <Card title="Fraud rate by channel" note="Fraud is spread across all three rails; channel alone separates little (Phase 2).">
              <Plot height={280} data={() => [{
                type: "bar", x: o.by_channel.map((c) => c.channel), y: o.by_channel.map((c) => c.fraud_rate), marker: { color: css("--s2") },
                text: o.by_channel.map((c) => fmt.pct(c.fraud_rate)), textposition: "outside", cliponaxis: false,
                customdata: o.by_channel.map((c) => [c.n, c.amount_median]),
                hovertemplate: "%{x}: <b>%{y:.1%}</b> fraud<br>%{customdata[0]:,} transactions · median ₹%{customdata[1]:,.0f}<extra></extra>",
              }]} layout={() => ({ showlegend: false, yaxis: { tickformat: ".0%", rangemode: "tozero" } })} />
            </Card>
          </div>
          <div className="grid g2">
            <Card title="Scenarios" aside="12 generator scenarios">
              <Table rows={[...o.scenarios].sort((a, b) => fmt.scenarioOrder(a.scenario, b.scenario))} rowKey="scenario" columns={[
                { key: "scenario", label: "Scenario" },
                { key: "n", label: "Rows", right: true, render: (r) => fmt.int(r.n) },
                { key: "fraud", label: "Fraud", right: true, render: (r) => fmt.int(r.fraud) },
                { key: "share", label: "Fraud share", render: (r) => (
                  <div className="bar-inline" style={{ width: 120 }} title={fmt.pct(r.fraud / r.n)}><span style={{ width: `${(100 * r.fraud) / r.n}%`, background: "var(--s2)" }} /></div>) },
              ]} />
            </Card>
            <div className="stack">
              <Card title="How a transaction is scored">
                <p className="ink2 small">risk = 100 × ({o.model.weights.model} × calibrated P(fraud) + {o.model.weights.anomaly} × anomaly percentile + {o.model.weights.graph} × network-flag risk), then rule overrides (sanctions match → CRITICAL, PEP → at least MEDIUM, structuring +10).</p>
                <div className="small" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <TierBadge tier="LOW" /><span className="muted">&lt; {o.model.cutoffs.MEDIUM.toFixed(0)}</span>
                  <TierBadge tier="MEDIUM" /><span className="muted">≥ {o.model.cutoffs.MEDIUM.toFixed(0)} review</span>
                  <TierBadge tier="HIGH" /><span className="muted">≥ {o.model.cutoffs.HIGH.toFixed(0)} alert</span>
                  <TierBadge tier="CRITICAL" /><span className="muted">≥ {o.model.cutoffs.CRITICAL.toFixed(0)} block</span>
                </div>
                <p className="small" style={{ marginTop: 10 }}><Link to="/score">Score a transaction →</Link></p>
              </Card>
              <Card title="Key findings">
                <ul className="small ink2" style={{ margin: 0, paddingLeft: 18 }}>
                  <li>The original data leaked labels and had no real network structure; both were found by an audit before modelling (Phase 0).</li>
                  <li>Behavioural features help every model; network features help significantly only where network structure exists (5 of 6 models on v1.1, 0 of 6 on the v1.0 control).</li>
                  <li>Tuned, calibrated XGBoost is the production model; a stacking ensemble did not beat it.</li>
                  <li>Anomaly detectors trained on legitimate data reach PR-AUC ≈ 0.55 and flag most BORDERLINE cases: unusual is not the same as fraudulent.</li>
                </ul>
              </Card>
            </div>
          </div>
        </div>
      )}</Async>
    </>
  );
}
