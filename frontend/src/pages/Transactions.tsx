import { Fragment, useState } from "react";
import { Link } from "react-router-dom";
import { fmt, useApi } from "../api";
import { Async, Card, LabelBadge, PageHead, Table } from "../components/ui";

type Tx = { transaction_id: string; timestamp: string; channel: string; amount_inr: number; sender_account_id: string; receiver_account_id: string;
  sender_name: string; receiver_name: string; tx_city: string; device_type: string; risk_label: string; is_fraud: number; scenario: string };
type Page = { total: number; offset: number; limit: number; items: Tx[] };

const SCENARIOS = ["S1 ", "S2 ", "S3 ", "S4 ", "S5 ", "S6 ", "S7 ", "S8 ", "S9 ", "S10", "S11", "S12"];
const DETAIL = [
  ["amount_ratio_vs_30d_avg", "Amount vs 30-day avg", (v: number) => v.toFixed(2) + "×"],
  ["is_new_device", "New device", (v: number) => (v ? "yes" : "no")],
  ["is_new_beneficiary", "New beneficiary", (v: number) => (v ? "yes" : "no")],
  ["time_since_beneficiary_added_min", "Payee added (min ago)", (v: number) => fmt.int(v)],
  ["distance_from_home_km", "Distance from home", (v: number) => v.toFixed(1) + " km"],
  ["sender_tx_10m", "Sender tx in 10 min", (v: number) => String(v)],
  ["sec_since_sender_last_inflow", "Seconds since inflow", (v: number) => (v == null ? "—" : fmt.int(v))],
  ["is_pass_through", "Pass-through", (v: number) => (v ? "yes" : "no")],
  ["receiver_distinct_senders_24h", "Receiver's senders (24 h)", (v: number) => String(v)],
  ["sender_kyc_status", "Sender KYC", (v: string) => v],
  ["sender_pep_flag", "PEP customer", (v: number) => (v ? "yes" : "no")],
] as const;

export default function TransactionsPage() {
  const [f, setF] = useState({ label: "", scenario: "", channel: "", min_amount: "", account_id: "" });
  const [offset, setOffset] = useState(0);
  const [sel, setSel] = useState<string | null>(null);
  const limit = 25;
  const page = useApi<Page>("/transactions", { ...f, scenario: f.scenario.trim(), min_amount: f.min_amount || undefined, limit, offset });
  const detail = useApi<Record<string, unknown>>(sel ? `/transactions/${sel}` : null, undefined, [sel]);
  const upd = (k: keyof typeof f, v: string) => { setF({ ...f, [k]: v }); setOffset(0); };

  return (
    <>
      <PageHead title="Transaction explorer" sub="Browse the 10,000 transactions with their labels and scenarios. Select one to see its profile, then score it." />
      <div className="filters panel">
        <label className="field">Label
          <select value={f.label} onChange={(e) => upd("label", e.target.value)}>
            <option value="">All</option><option>LEGITIMATE</option><option>FRAUDULENT</option><option>BORDERLINE</option>
          </select></label>
        <label className="field">Scenario
          <select value={f.scenario} onChange={(e) => upd("scenario", e.target.value)}>
            <option value="">All</option>{SCENARIOS.map((s) => <option key={s} value={s}>{s.trim()}</option>)}
          </select></label>
        <label className="field">Channel
          <select value={f.channel} onChange={(e) => upd("channel", e.target.value)}>
            <option value="">All</option><option>UPI</option><option>NEFT</option><option>RTGS</option>
          </select></label>
        <label className="field">Min amount (₹)
          <input type="number" min={0} value={f.min_amount} onChange={(e) => upd("min_amount", e.target.value)} placeholder="e.g. 50000" /></label>
        <label className="field">Account
          <input type="text" value={f.account_id} onChange={(e) => upd("account_id", e.target.value.toUpperCase())} placeholder="ACC_IND_000013" /></label>
      </div>
      <div className="grid g3">
        <Card className="span2" title="Transactions" aside={page.data ? `${fmt.int(page.data.total)} match` : ""}>
          <Async state={page}>{(p) => (
            <>
              <Table rows={p.items} rowKey="transaction_id" selected={sel ?? undefined} onRow={(r) => setSel(r.transaction_id)} columns={[
                { key: "timestamp", label: "Time", render: (r) => <span className="num" style={{ whiteSpace: "nowrap" }}>{fmt.short(r.timestamp)}</span> },
                { key: "channel", label: "Channel" },
                { key: "amount_inr", label: "Amount", right: true, render: (r) => fmt.inr(r.amount_inr) },
                { key: "sender_name", label: "From → to", render: (r) => <span className="small">{r.sender_name} → {r.receiver_name}</span> },
                { key: "risk_label", label: "Label", render: (r) => <LabelBadge label={r.risk_label} /> },
                { key: "scenario", label: "Scenario", render: (r) => <span className="small muted">{r.scenario}</span> },
              ]} />
              <div className="filters" style={{ marginTop: 10, marginBottom: 0 }}>
                <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>← Previous</button>
                <span className="small muted num">{p.total ? `${offset + 1}–${Math.min(offset + limit, p.total)} of ${fmt.int(p.total)}` : "No matches"}</span>
                <button disabled={offset + limit >= p.total} onClick={() => setOffset(offset + limit)}>Next →</button>
              </div>
            </>
          )}</Async>
        </Card>
        <Card title="Profile">
          {!sel && <p className="empty">Select a transaction to see its profile.</p>}
          {sel && <Async state={detail}>{(t) => (
            <div className="stack">
              <div><div className="small muted">{String(t.transaction_id)}</div>
                <div className="hero" style={{ fontSize: "2.4rem" }}>{fmt.inr(t.amount_inr as number)}</div>
                <div className="small ink2">{String(t.channel)} · {fmt.date(t.timestamp as string)} · {String(t.tx_city)}</div></div>
              <div><LabelBadge label={String(t.risk_label)} /> <span className="small muted">{String(t.scenario)}</span></div>
              <dl className="kv">
                <dt>From</dt><dd>{String(t.sender_name)} <span className="muted">({String(t.sender_account_id)})</span></dd>
                <dt>To</dt><dd>{String(t.receiver_name)} <span className="muted">({String(t.receiver_account_id)})</span></dd>
                {DETAIL.map(([k, l, f]) => <Fragment key={k}><dt>{l}</dt><dd>{(f as (v: never) => string)(t[k] as never)}</dd></Fragment>)}
              </dl>
              <div className="actions">
                <Link className="btn primary" to={`/score?tx=${sel}`}>Score this transaction</Link>
                <Link className="btn" to={`/network?account=${String(t.receiver_account_id)}`}>Receiver's network</Link>
              </div>
            </div>
          )}</Async>}
        </Card>
      </div>
    </>
  );
}
