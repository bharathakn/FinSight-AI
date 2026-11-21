import { useCallback, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api, ApiError, fmt, TX_EXAMPLES } from "../api";
import { css, Plot } from "../components/Plot";
import { Card, ErrorBox, LabelBadge, Loading, PageHead, Table, TierBadge } from "../components/ui";

type Reason = { feature: string; value: unknown; contribution: number; direction: string };
type Score = {
  transaction_id: string; risk_score: number; tier: string; base_score: number;
  components: { model_probability: number; anomaly_pct: number; graph_risk: number }; weights: Record<string, number>; cutoffs: Record<string, number>;
  overrides_applied: string[]; decision: { model_flagged: boolean; model_threshold: number; alert: boolean };
  reasons: Reason[]; base_log_odds: number;
  network: { graph_risk: number; flags: Record<string, boolean>; flag_rates: Record<string, number>; structuring: boolean };
  rules: { items: string[]; fired_rules: { rule: string; consequent: string; support: number; confidence: number; lift: number }[]; n_fired: number; n_fired_fraud: number };
  cluster: { cluster: number; size: number; fraud_rate: number; top_features: { feature: string; z: number }[] };
  anomaly: { error: number; pct: number; above_p95: boolean };
  sanctions: { name: string; hit: boolean; best_score: number | null; matches: { matched_name: string; source: string; program: string; score: number }[] }[];
  input: { source: string; overrides: string[]; derived: string[] };
  context: Record<string, unknown>; timings_ms: Record<string, number>; mode: string; model_version: string;
};

type WhatIf = { amount_inr: string; is_new_device: string; is_new_beneficiary: string; time_since_beneficiary_added_min: string; distance_from_home_km: string; sender_tx_10m: string };
const EMPTY: WhatIf = { amount_inr: "", is_new_device: "", is_new_beneficiary: "", time_since_beneficiary_added_min: "", distance_from_home_km: "", sender_tx_10m: "" };
const CLUSTER_NAME: Record<number, string> = { 0: "Velocity bursts", 1: "Normal traffic", 2: "Unusual amounts, community flows", 3: "Fan-in hubs" };

function Component({ label, value, weight, hint }: { label: string; value: number; weight: number; hint: string }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div className="small" style={{ display: "flex", justifyContent: "space-between" }}><span>{label} <span className="muted">× {weight}</span></span><b className="num">{fmt.pct(value)}</b></div>
      <div className="bar-inline"><span style={{ width: `${100 * value}%`, background: "var(--accent)" }} /></div>
      <div className="small muted">{hint}</div>
    </div>
  );
}

export default function RiskScorerPage() {
  const [params, setParams] = useSearchParams();
  const [tx, setTx] = useState(params.get("tx") ?? TX_EXAMPLES.fraud);
  const [whatIf, setWhatIf] = useState<WhatIf>(EMPTY);
  const [cp, setCp] = useState("");
  const [res, setRes] = useState<Score | null>(null);
  const [err, setErr] = useState<ApiError | null>(null);
  const [busy, setBusy] = useState(false);

  const run = useCallback(async (id: string, w: WhatIf, counterparty: string) => {
    setBusy(true); setErr(null);
    const overrides: Record<string, number> = {};
    Object.entries(w).forEach(([k, v]) => { if (v !== "") overrides[k] = Number(v); });
    try {
      const r = await api.post<Score>("/score-transaction", { transaction_id: id.trim(), overrides, counterparty_name: counterparty || undefined });
      setRes(r);
      setParams({ tx: id.trim() }, { replace: true });
    } catch (e) { setErr(e as ApiError); } finally { setBusy(false); }
  }, [setParams]);

  useEffect(() => { run(tx, EMPTY, ""); /* score on first load */ // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const example = (id: string) => { setTx(id); setWhatIf(EMPTY); setCp(""); run(id, EMPTY, ""); };
  const set = (k: keyof WhatIf, v: string) => setWhatIf({ ...whatIf, [k]: v });
  const ctx = res?.context ?? {};

  return (
    <>
      <PageHead title="Risk scorer" sub="Score a transaction 0–100. Change fields to ask “what if?”: the model, anomaly detector, network flags, rules and sanctions screening all run on the edited transaction." />
      <Card>
        <div className="filters" style={{ marginBottom: 8 }}>
          <label className="field">Transaction ID <input type="text" value={tx} onChange={(e) => setTx(e.target.value.toUpperCase())} style={{ minWidth: 230 }} /></label>
          <button className="primary" onClick={() => run(tx, whatIf, cp)} disabled={busy}>{busy ? "Scoring…" : "Score"}</button>
          <span className="small muted">Examples:</span>
          <button onClick={() => example(TX_EXAMPLES.fraud)}>Mule pass-through (fraud)</button>
          <button onClick={() => example(TX_EXAMPLES.borderline)}>New device (borderline)</button>
          <button onClick={() => example(TX_EXAMPLES.legit)}>Ordinary payment</button>
          <Link to="/transactions" className="small">Browse transactions →</Link>
        </div>
        <div className="filters" style={{ marginBottom: 0 }}>
          <span className="small muted" style={{ alignSelf: "center" }}>What if…</span>
          <label className="field">Amount (₹) <input type="number" min={1} value={whatIf.amount_inr} placeholder={ctx.amount_inr ? String(Math.round(ctx.amount_inr as number)) : ""} onChange={(e) => set("amount_inr", e.target.value)} /></label>
          <label className="field">New device <select value={whatIf.is_new_device} onChange={(e) => set("is_new_device", e.target.value)}><option value="">as is</option><option value="1">yes</option><option value="0">no</option></select></label>
          <label className="field">New beneficiary <select value={whatIf.is_new_beneficiary} onChange={(e) => set("is_new_beneficiary", e.target.value)}><option value="">as is</option><option value="1">yes</option><option value="0">no</option></select></label>
          <label className="field">Payee added (min ago) <input type="number" min={0} value={whatIf.time_since_beneficiary_added_min} onChange={(e) => set("time_since_beneficiary_added_min", e.target.value)} /></label>
          <label className="field">Distance from home (km) <input type="number" min={0} value={whatIf.distance_from_home_km} onChange={(e) => set("distance_from_home_km", e.target.value)} /></label>
          <label className="field">Sender tx in 10 min <input type="number" min={0} value={whatIf.sender_tx_10m} onChange={(e) => set("sender_tx_10m", e.target.value)} /></label>
          <label className="field">Counterparty name (sanctions) <input type="text" value={cp} onChange={(e) => setCp(e.target.value)} placeholder="optional" /></label>
          <button onClick={() => { setWhatIf(EMPTY); setCp(""); run(tx, EMPTY, ""); }}>Reset</button>
        </div>
      </Card>
      {err && <div style={{ marginTop: 14 }}><ErrorBox error={err} /></div>}
      {!res && !err && <Loading what="Scoring" />}
      {res && (
        <div className={"stack " + (busy ? "refetch" : "")} style={{ marginTop: 14 }}>
          <div className="grid g3">
            <Card title="Risk score">
              <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
                <div className="hero" aria-label={`risk score ${res.risk_score.toFixed(0)} of 100`}>{res.risk_score.toFixed(0)}</div>
                <TierBadge tier={res.tier} />
              </div>
              <div className="small ink2" style={{ marginTop: 6 }}>
                {res.decision.alert ? "Alert: an analyst should review this payment." : res.tier === "MEDIUM" ? "Review band: low-priority check." : "Allow."}
                {res.overrides_applied.length > 0 && <> Overrides: {res.overrides_applied.join(", ")} (base score {res.base_score.toFixed(0)}).</>}
              </div>
              <div className="bar-inline" style={{ marginTop: 12, height: 10, position: "relative" }}><span style={{ width: `${res.risk_score}%`, background: res.tier === "CRITICAL" ? "var(--critical)" : "var(--accent)" }} /></div>
              <div className="small muted num" style={{ display: "flex", justifyContent: "space-between" }}>
                <span>0</span><span>MEDIUM {res.cutoffs.MEDIUM.toFixed(0)}</span><span>HIGH {res.cutoffs.HIGH.toFixed(0)}</span><span>CRITICAL {res.cutoffs.CRITICAL.toFixed(0)}</span><span>100</span>
              </div>
              <dl className="kv" style={{ marginTop: 12 }}>
                <dt>Transaction</dt><dd>{res.transaction_id}</dd>
                <dt>Amount</dt><dd>{fmt.inr(ctx.amount_inr as number)} · {String(ctx.channel)}</dd>
                <dt>From → to</dt><dd>{String(ctx.sender_name)} → {String(ctx.receiver_name)}</dd>
                <dt>Ground truth</dt><dd><LabelBadge label={String(ctx.risk_label)} /> <span className="small muted">{String(ctx.scenario)}</span></dd>
                {res.input.overrides.length > 0 && <><dt>What-if</dt><dd className="small">{res.input.overrides.join(", ")}{res.input.derived.length ? ` (recomputed: ${res.input.derived.join(", ")})` : ""}</dd></>}
              </dl>
            </Card>
            <Card title="Components">
              <Component label="Model probability" value={res.components.model_probability} weight={res.weights.model} hint={`calibrated XGBoost · alert threshold ${fmt.pct(res.decision.model_threshold, 0)}`} />
              <Component label="Anomaly percentile" value={res.components.anomaly_pct} weight={res.weights.anomaly} hint={`autoencoder error vs legitimate traffic${res.anomaly.above_p95 ? " · above p95" : ""}`} />
              <Component label="Network risk" value={res.components.graph_risk} weight={res.weights.graph} hint="training fraud rate of the strongest network red flag" />
              <h3 style={{ marginTop: 12 }}>Network red flags</h3>
              <div>{Object.entries(res.network.flags).map(([k, v]) => (
                <span key={k} className={"chip" + (v ? " on" : "")}>{v ? "●" : "○"} {k.replace(/_/g, " ").toLowerCase()}{v && res.network.flag_rates[k] != null ? ` · ${fmt.pct(res.network.flag_rates[k], 0)}` : ""}</span>))}</div>
            </Card>
            <Card title="Context">
              <h3>Cluster (no labels used)</h3>
              <p className="small"><span className="dot" style={{ background: `var(--s${4 + res.cluster.cluster})`, marginRight: 6 }} /><b>{res.cluster.cluster} · {CLUSTER_NAME[res.cluster.cluster]}</b> — {fmt.int(res.cluster.size)} rows, {fmt.pct(res.cluster.fraud_rate, 0)} fraud</p>
              <h3>Sanctions screening (OFAC + UN)</h3>
              {res.sanctions.map((s) => (
                <p key={s.name} className="small">{s.hit ? "▲" : "✓"} <b>{s.name}</b>: {s.hit ? `match ${s.best_score} with ${s.matches[0].matched_name} (${s.matches[0].source}, ${s.matches[0].program ?? "—"})` : "no match ≥ 90"}</p>
              ))}
              <h3>Served by</h3>
              <p className="small ink2 num">{res.mode} mode · total {fmt.ms(res.timings_ms.total)} (ml {fmt.ms(res.timings_ms.ml)}, mining {fmt.ms(res.timings_ms.mining)}, graph {fmt.ms(res.timings_ms.graph)}, sanctions {fmt.ms(res.timings_ms.sanctions)})</p>
              <p className="small muted">model {res.model_version}</p>
            </Card>
          </div>
          <div className="grid g2">
            <Card title="Why: top feature contributions" note={`Native TreeSHAP from the XGBoost model, in log-odds (baseline ${res.base_log_odds.toFixed(2)}). Orange pushes towards fraud, blue towards legitimate.`}>
              <Plot height={300} data={() => {
                const r = [...res.reasons].reverse();
                return [{ type: "bar", orientation: "h", y: r.map((x) => `${fmt.feature(x.feature)} = ${typeof x.value === "number" ? +Number(x.value).toFixed(2) : String(x.value)}`),
                  x: r.map((x) => x.contribution), marker: { color: r.map((x) => (x.contribution > 0 ? css("--s2") : css("--s1"))) },
                  text: r.map((x) => (x.contribution > 0 ? "+" : "") + x.contribution.toFixed(2)), textposition: "outside", cliponaxis: false,
                  hovertemplate: "%{y}<br>contribution <b>%{x:+.3f}</b> log-odds<extra></extra>" }];
              }} layout={() => ({ showlegend: false, xaxis: { zeroline: true, zerolinewidth: 1.5, title: { text: "contribution (log-odds)" } }, margin: { l: 10, r: 50, t: 6, b: 10 } })} />
            </Card>
            <Card title="Association rules that fire" aside={`${res.rules.n_fired} rules (${res.rules.n_fired_fraud} → fraud)`} note="Mined in Phase 4. Items true for this transaction are listed below.">
              {res.rules.fired_rules.length ? (
                <Table rows={res.rules.fired_rules} columns={[
                  { key: "rule", label: "Rule", render: (r) => <span className="small">{r.rule}</span> },
                  { key: "confidence", label: "Confidence", right: true, render: (r) => fmt.pct(r.confidence, 0) },
                  { key: "lift", label: "Lift", right: true, render: (r) => r.lift.toFixed(1) },
                ]} />
              ) : <p className="small muted">No mined rule fires for this transaction.</p>}
              <div style={{ marginTop: 10 }}>{res.rules.items.map((i) => <span key={i} className="chip">{i}</span>)}</div>
            </Card>
          </div>
        </div>
      )}
    </>
  );
}
