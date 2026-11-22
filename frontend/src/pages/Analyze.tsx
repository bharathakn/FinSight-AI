// Analyze transactions (home page, Phase 9b): run the full pipeline live on the demo dataset, an uploaded CSV or a
// simulated attack, then explore what was found. Every number comes from POST /analyze/* and /analyses/{id}/*.
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, ApiError, fmt, useApi } from "../api";
import { css, Plot } from "../components/Plot";
import { Icon, IconName } from "../components/Icon";
import { Card, ErrorBox, LabelBadge, PageHead, Stat, Table, TierBadge } from "../components/ui";

type Row = { transaction_id: string; timestamp: string; channel: string; amount_inr: number; sender_account_id: string; receiver_account_id: string;
  sender_name: string; receiver_name: string; risk_score: number; tier: string; p_fraud: number; anomaly_pct: number; graph_risk: number;
  cluster: number; network_flags: string; top_rule: string; n_fraud_rules: number; overrides: string; sanctions_score: number;
  is_fraud?: number; risk_label?: string; scenario?: string };
type Eval = { positives: number; rows: number; pr_auc: number; roc_auc: number; precision: number; recall: number; f1: number; tp: number; fp: number; fn: number; tn: number;
  alert_rule: string; note?: string; tiers_by_label?: Record<string, number | string>[]; recall_by_scenario?: { scenario: string; n: number; recall: number }[];
  held_out?: Eval; held_out_note?: string };
type Summary = {
  analysis_id: string; source: Record<string, unknown>;
  data_quality: { rows_received: number; rows_valid: number; rows_dropped: number; issues: { problem: string; rows: number }[]; unknown_accounts: number;
    unknown_account_examples: string[]; labels_present: string[]; period: [string, string]; defaulted_columns: Record<string, string>; imputed_features: Record<string, number> };
  timings: Record<string, number>;
  kpis: { transactions: number; alerts: number; alert_rate: number; critical: number; value_at_risk_inr: number; amount_total_inr: number; accounts: number };
  tiers: Record<string, number>; cutoffs: Record<string, number>; score_histogram: { edges: number[]; counts: number[] };
  timeline: { time: string; n: number; alerts: number }[];
  groups: { hubs: Record<string, number | string>[]; chains: Record<string, number | string>[]; bursts: Record<string, number | string>[]; shared_devices: Record<string, number | string>[];
    graph: { nodes: { id: string; hub: boolean }[]; links: { source: string; target: string; n: number; amount_inr: number; max_risk: number }[] } };
  rules: { rule: string; alerts: number }[]; network_flags: Record<string, number>;
  anomalies: { count: number; note: string; top: Row[] }; evaluation: Eval | null; top: Row[];
};
type Schema = { limits: { max_rows: number; max_megabytes: number }; required: { column: string; description: string }[];
  optional: { column: string; description: string; default_if_missing: string }[]; labels: { column: string; description: string }[];
  notes: string[]; attacks: Record<string, string> };
type Detail = { risk_score: number; tier: string; components: Record<string, number>; reasons: { feature: string; value: unknown; contribution: number }[];
  network: { flags: Record<string, boolean> }; rules: { fired_rules: { rule: string; confidence: number }[] }; cluster: { cluster: number; fraud_rate: number };
  sanctions: { name: string; hit: boolean; best_score: number | null }[]; overrides_applied: string[] };

const ATTACK_LABEL: Record<string, string> = { fraud_ring: "Fraud ring", mule_chain: "Mule chain", burst: "Burst", structuring: "Structuring", shared_device: "Shared device" };
const TIER_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
const TIER_VAR: Record<string, string> = { LOW: "--tier-low", MEDIUM: "--tier-medium", HIGH: "--tier-high", CRITICAL: "--critical" };
const SIGNALS: [IconName, string][] = [["model", "Calibrated ML model"], ["cluster", "Anomaly detector"], ["network", "Network red flags"], ["pattern", "Mined rules"], ["quality", "Sanctions screening"]];

// ---------------------------------------------------------------- input panels
function Inputs({ schema, busy, onRun }: { schema: Schema | null; busy: boolean; onRun: (kind: string, run: () => Promise<Summary>) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [drag, setDrag] = useState(false);
  const [attacks, setAttacks] = useState<string[]>(["fraud_ring", "mule_chain"]);
  const [size, setSize] = useState(10);
  const [camo, setCamo] = useState(false);
  const [bg, setBg] = useState(1500);
  const [showSchema, setShowSchema] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const toggle = (a: string) => setAttacks(attacks.includes(a) ? attacks.filter((x) => x !== a) : [...attacks, a]);
  return (
    <div className="grid intake">
      <div className="stack">
        <Card title="Upload your transactions" note={schema ? `Up to ${fmt.int(schema.limits.max_rows)} rows / ${schema.limits.max_megabytes.toFixed(0)} MB. Files are processed in memory and never stored.` : undefined}>
          <div className={"drop" + (drag ? " over" : "")} role="button" tabIndex={0} aria-label={file ? `Selected file ${file.name}. Choose another file` : "Choose a CSV file"}
               onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
               onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget as Node)) setDrag(false); }}
               onDrop={(e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); }}
               onClick={() => input.current?.click()}
               onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.current?.click(); } }}>
            <span className="tile large"><Icon name="file" /></span>
            <div className="drop-title">{file ? file.name : "Drop a CSV of payments here"}</div>
            <div className="small muted">{file ? `${(file.size / 1024).toFixed(0)} KB` : "transactions in the FinSight format"}</div>
            <span className="btn" aria-hidden="true">{file ? "Choose another file" : "Choose file"}</span>
            <input ref={input} type="file" accept=".csv,text/csv" hidden onClick={(e) => e.stopPropagation()} onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </div>
          <div className="actions" style={{ marginTop: 14 }}>
            <button className="primary" disabled={busy || !file} onClick={() => file && onRun(`upload: ${file.name}`, () => api.upload<Summary>("/analyze/upload", file))}>Analyze file</button>
            <a className="small" href={api.url("/analyze/template.csv")}>Template</a>
            <a className="small" href={api.url("/analyze/sample.csv", { rows: 1000 })}>Sample (1,000 rows)</a>
            <button className="link" aria-expanded={showSchema} onClick={() => setShowSchema(!showSchema)}>{showSchema ? "Hide" : "Show"} columns</button>
          </div>
          {showSchema && schema && (
            <div className="small" style={{ marginTop: 14 }}>
              <b>Required:</b> {schema.required.map((c) => <code key={c.column} title={c.description} style={{ marginRight: 4 }}>{c.column}</code>)}
              <div style={{ marginTop: 6 }}><b>Optional</b> (filled from the bank's records if missing):</div>
              <ul style={{ margin: "4px 0", paddingLeft: 18 }}>{schema.optional.map((c) => <li key={c.column}><code>{c.column}</code> <span className="muted">— default: {c.default_if_missing}</span></li>)}</ul>
              <div><b>Labels</b> (optional, evaluation only): {schema.labels.map((c) => <code key={c.column} style={{ marginRight: 4 }}>{c.column}</code>)}</div>
            </div>
          )}
        </Card>
        <Card title="No file? Use the demo dataset" icon="data" warm
              note="On the free cloud server the first run takes about half a minute; after that the demo results are cached and open instantly.">
          <p className="small ink2">10,000 synthetic Indian payments (August 2026) with ground-truth labels. Runs the complete pipeline and also shows how accurate the model is.</p>
          <div className="actions">
            <button className="primary" disabled={busy} onClick={() => onRun("demo dataset (10,000)", () => api.post<Summary>("/analyze/demo", {}, 120000))}>Analyze all 10,000</button>
            <button disabled={busy} onClick={() => onRun("demo sample (2,000)", () => api.post<Summary>("/analyze/demo", { sample: 2000 }, 120000))}>Quick sample (2,000)</button>
          </div>
        </Card>
      </div>
      <Card title="Simulate an attack" icon="attack" warm note="A normal day of payments (31 Aug) with the chosen attacks injected and labelled, so you can see whether the system catches them.">
        <div className="attacks">
          {schema && Object.entries(schema.attacks).map(([k, d]) => (
            <label key={k} className={"attack" + (attacks.includes(k) ? " on" : "")} title={d}>
              <input type="checkbox" checked={attacks.includes(k)} onChange={() => toggle(k)} />
              <span><b>{ATTACK_LABEL[k]}</b><small>{d.split(": ")[1] ?? d}</small></span>
            </label>
          ))}
        </div>
        <div className="filters" style={{ marginTop: 14, marginBottom: 6 }}>
          <label className="field">Intensity: {size}<input type="range" min={3} max={40} value={size} onChange={(e) => setSize(Number(e.target.value))} /></label>
          <label className="field">Normal payments
            <select value={bg} onChange={(e) => setBg(Number(e.target.value))}><option value={500}>500</option><option value={1500}>1,500</option><option value={5000}>5,000</option></select></label>
        </div>
        <label className="check" title="Each payment looks normal on behaviour (known device and payee, near home); only the network pattern gives it away">
          <input type="checkbox" checked={camo} onChange={(e) => setCamo(e.target.checked)} /> Camouflage (looks normal per payment)</label>
        <div style={{ marginTop: 12 }}>
          <button className="primary" disabled={busy || attacks.length === 0}
            onClick={() => onRun(`simulation: ${attacks.map((a) => ATTACK_LABEL[a]).join(" + ")}${camo ? " (camouflaged)" : ""}`,
              () => api.post<Summary>("/analyze/simulate", { attacks, size, camouflage: camo, background: bg, seed: Math.floor(Math.random() * 1e6) }, 120000))}>Run simulation</button>
        </div>
      </Card>
    </div>
  );
}

// ---------------------------------------------------------------- results pieces
function DataQuality({ s }: { s: Summary }) {
  const dq = s.data_quality;
  const imputed = Object.entries(dq.imputed_features);
  const defaulted = Object.entries(dq.defaulted_columns);
  return (
    <Card title="Data quality of the input" aside={`${fmt.int(dq.rows_valid)} of ${fmt.int(dq.rows_received)} rows usable`}>
      <dl className="kv">
        <dt>Period</dt><dd>{fmt.date(dq.period[0])} – {fmt.date(dq.period[1])}</dd>
        <dt>Rejected rows</dt><dd>{dq.issues.length ? dq.issues.map((i) => `${i.problem}: ${i.rows}`).join(" · ") : "none ✓"}</dd>
        <dt>Unknown accounts</dt><dd>{dq.unknown_accounts ? `${dq.unknown_accounts} (e.g. ${dq.unknown_account_examples.join(", ")}): typical values used` : "none ✓"}</dd>
        <dt>Columns defaulted</dt><dd>{defaulted.length ? defaulted.map(([c, d]) => <div key={c} className="small"><code>{c}</code> <span className="muted">{d}</span></div>) : "none: every input column was provided ✓"}</dd>
        <dt>Features imputed</dt><dd>{imputed.length ? `${imputed.length} features on up to ${Math.max(...imputed.map(([, n]) => n))} rows (unknown accounts or devices)` : "none ✓"}</dd>
        <dt>Labels</dt><dd>{dq.labels_present.length ? `${dq.labels_present.join(", ")}: used only to evaluate below` : "none (no accuracy check possible)"}</dd>
      </dl>
    </Card>
  );
}

function Evaluation({ e }: { e: Eval }) {
  const view = e.held_out ?? null;
  const Box = ({ x, title }: { x: Eval; title: string }) => (
    <div>
      <h3>{title}</h3>
      <div className="metrics">
        {[["PR-AUC", x.pr_auc], ["Precision", x.precision], ["Recall", x.recall], ["F1", x.f1]].map(([l, v]) => (
          <div key={l as string}><div className="small muted">{l}</div><div className="value">{(v as number).toFixed(3)}</div></div>))}
      </div>
      <div className="small ink2 num" style={{ marginTop: 6 }}>{x.tp} frauds caught · {x.fn} missed · {x.fp} false alarms · {fmt.int(x.tn)} correctly allowed ({x.alert_rule})</div>
    </div>
  );
  if (e.note) return <Card title="Accuracy check (labels in the file)"><p className="small ink2">{e.note}</p></Card>;
  return (
    <Card title="Accuracy check (labels in the file)" note="The labels never influence a score; they are compared with the results afterwards.">
      <div className="stack">
        <Box x={e} title={`All ${fmt.int(e.rows)} labelled rows (${fmt.int(e.positives)} frauds)`} />
        {view && <><Box x={view} title={`Held-out test rows only (${fmt.int(view.rows)})`} /><p className="small muted">{e.held_out_note}</p></>}
        {e.recall_by_scenario && e.recall_by_scenario.length > 0 && (
          <Plot height={Math.max(160, 26 * e.recall_by_scenario.length + 40)} data={() => {
            const r = [...e.recall_by_scenario!].sort((a, b) => a.recall - b.recall);
            return [{ type: "bar", orientation: "h", y: r.map((x) => x.scenario), x: r.map((x) => x.recall), marker: { color: css("--s2") },
              text: r.map((x) => `${fmt.pct(x.recall, 0)} of ${x.n}`), textposition: "outside", cliponaxis: false,
              hovertemplate: "%{y}: <b>%{x:.0%}</b> caught<extra></extra>" }];
          }} layout={() => ({ showlegend: false, xaxis: { range: [0, 1.25], tickformat: ".0%", title: { text: "frauds alerted (HIGH or CRITICAL)" } }, margin: { l: 10, r: 60, t: 6, b: 10 } })} />
        )}
      </div>
    </Card>
  );
}

const KIND_COLOR: Record<string, string> = { "fan-in": "var(--critical)", "pass-through": "var(--s2)", burst: "var(--s7)" };
const KIND_LABEL: Record<string, string> = { "fan-in": "fan-in to a hub", "pass-through": "pass-through / mule chain", burst: "burst from one sender" };

type GLink = Summary["groups"]["graph"]["links"][number] & { kind?: string };

function GroupGraph({ g }: { g: Summary["groups"]["graph"] }) {
  // One small diagram per structure: a star per fan-in hub (senders around the hub), a star per burst sender,
  // and each pass-through chain on its own. An account in two structures appears in both. Deterministic layout.
  const layout = useMemo(() => {
    const links = g.links as GLink[];
    const groups: { key: string; center: string; kind: string; links: GLink[] }[] = [];
    const byKey: Record<string, number> = {};
    const add = (key: string, center: string, kind: string, l: GLink) => {
      if (byKey[key] === undefined) { byKey[key] = groups.length; groups.push({ key, center, kind, links: [] }); }
      groups[byKey[key]].links.push(l);
    };
    links.filter((l) => (l.kind ?? "fan-in") === "fan-in").forEach((l) => add("hub:" + l.target, l.target, "fan-in", l));
    links.filter((l) => l.kind === "burst").forEach((l) => add("burst:" + l.source, l.source, "burst", l));
    // pass-through: connected components of the chain links
    const pt = links.filter((l) => l.kind === "pass-through");
    const parent: Record<string, string> = {};
    const find = (x: string): string => (parent[x] === undefined || parent[x] === x ? (parent[x] = x) : (parent[x] = find(parent[x])));
    pt.forEach((l) => { parent[find(l.source)] = find(l.target); });
    pt.forEach((l) => add("chain:" + find(l.source), "", "pass-through", l));
    const shown = groups.sort((a, b) => b.links.length - a.links.length).slice(0, 9);
    const cols = Math.min(3, shown.length), CW = 230, CH = 210;
    const nodes: { key: string; id: string; x: number; y: number; center: boolean; kind: string }[] = [];
    const edges: { x1: number; y1: number; x2: number; y2: number; l: GLink }[] = [];
    shown.forEach((grp, k) => {
      const cx = (k % cols) * CW + CW / 2, cy = Math.floor(k / cols) * CH + CH / 2;
      const pos: Record<string, [number, number]> = {};
      if (grp.kind === "pass-through") {           // chain drawn left → right in time order of first appearance
        const order: string[] = [];
        grp.links.forEach((l) => [l.source, l.target].forEach((n) => { if (!order.includes(n)) order.push(n); }));
        order.forEach((n, i) => { pos[n] = [cx - 90 + (180 * i) / Math.max(1, order.length - 1), cy + (i % 2 ? 22 : -22)]; });
      } else {
        const others = [...new Set(grp.links.map((l) => (l.source === grp.center ? l.target : l.source)))];
        const r = Math.min(85, 30 + 3 * others.length);
        pos[grp.center] = [cx, cy];
        others.forEach((n, i) => { const a = (2 * Math.PI * i) / Math.max(1, others.length) - Math.PI / 2; pos[n] = [cx + r * Math.cos(a), cy + r * 0.85 * Math.sin(a)]; });
      }
      Object.entries(pos).forEach(([id, [x, y]]) => nodes.push({ key: grp.key + id, id, x, y, center: id === grp.center, kind: grp.kind }));
      grp.links.forEach((l) => edges.push({ x1: pos[l.source][0], y1: pos[l.source][1], x2: pos[l.target][0], y2: pos[l.target][1], l }));
    });
    return { nodes, edges, W: Math.max(1, cols) * CW, H: Math.ceil(shown.length / Math.max(1, cols)) * CH, hidden: groups.length - shown.length,
             kinds: [...new Set(shown.map((x) => x.kind))] };
  }, [g]);
  if (!g.links.length) return <p className="small muted">No hub, chain or burst structure among the alerts.</p>;
  return (
    <>
      <svg viewBox={`0 0 ${layout.W} ${layout.H}`} width="100%" role="img" aria-label="Suspicious account structures" style={{ maxHeight: 640 }}>
        {layout.edges.map((e, i) => {
          const kind = e.l.kind ?? "fan-in";
          return <line key={i} x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2} stroke={KIND_COLOR[kind]} strokeWidth={1.2 + Math.min(2.5, e.l.n - 1)} opacity={0.85}>
            <title>{`${e.l.source} → ${e.l.target} (${KIND_LABEL[kind]})\n${e.l.n} alerted payment(s) · ${fmt.inr(e.l.amount_inr)} · max risk ${e.l.max_risk.toFixed(0)}`}</title></line>;
        })}
        {layout.nodes.map((n) => (
          <g key={n.key}>
            <circle cx={n.x} cy={n.y} r={n.center ? 8 : 4} fill={n.center ? KIND_COLOR[n.kind] : "var(--surface-2)"} stroke="var(--ink-2)" strokeWidth={1.2} />
            {n.center && <text x={n.x} y={n.y - 11} textAnchor="middle" fontSize="10" fill="var(--ink)" fontWeight={n.center ? 700 : 400}>{n.id.replace("ACC_IND_", "")}</text>}
            <title>{n.id}</title>
          </g>
        ))}
      </svg>
      <div className="small" style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
        {layout.kinds.map((k) => <span key={k}><span style={{ display: "inline-block", width: 18, height: 3, background: KIND_COLOR[k], verticalAlign: "middle", marginRight: 5 }} />{KIND_LABEL[k]}</span>)}
        {layout.hidden > 0 && <span className="muted">+{layout.hidden} smaller structure(s) not drawn</span>}
      </div>
    </>
  );
}

function RowDetail({ aid, row, onClose }: { aid: string; row: Row; onClose: () => void }) {
  const st = useApi<Detail>(`/analyses/${aid}/transactions/${row.transaction_id}`, undefined, [row.transaction_id]);
  return (
    <Card title={<span>{row.transaction_id} <TierBadge tier={row.tier} /></span>} aside={<button onClick={onClose}>Close</button>}>
      <div className="small ink2">{fmt.inr(row.amount_inr)} · {row.channel} · {fmt.date(row.timestamp)} · {row.sender_name ?? row.sender_account_id} → {row.receiver_name ?? row.receiver_account_id}
        {row.risk_label && <> · ground truth <LabelBadge label={row.risk_label} /> <span className="muted">{row.scenario}</span></>}</div>
      {st.error && <ErrorBox error={st.error} />}
      {st.data && (
        <div className="grid g2" style={{ marginTop: 10 }}>
          <div>
            <Plot height={240} data={() => {
              const r = [...st.data!.reasons].reverse();
              return [{ type: "bar", orientation: "h", y: r.map((x) => `${fmt.feature(x.feature)} = ${typeof x.value === "number" ? +Number(x.value).toFixed(2) : String(x.value)}`),
                x: r.map((x) => x.contribution), marker: { color: r.map((x) => (x.contribution > 0 ? css("--s2") : css("--s1"))) },
                hovertemplate: "%{y}<br><b>%{x:+.3f}</b> log-odds<extra></extra>" }];
            }} layout={() => ({ showlegend: false, xaxis: { title: { text: "contribution towards fraud (log-odds)" } } })} />
          </div>
          <div className="small">
            <dl className="kv">
              <dt>Risk score</dt><dd><b>{st.data.risk_score.toFixed(0)}</b> ({st.data.tier}){st.data.overrides_applied.length ? ` · overrides: ${st.data.overrides_applied.join(", ")}` : ""}</dd>
              <dt>Model</dt><dd>{fmt.pct(st.data.components.model_probability)} fraud probability</dd>
              <dt>Anomaly</dt><dd>{fmt.pct(st.data.components.anomaly_pct)} percentile vs legitimate traffic</dd>
              <dt>Network</dt><dd>{Object.entries(st.data.network.flags).filter(([, v]) => v).map(([k]) => k.replace(/_/g, " ").toLowerCase()).join(", ") || "no red flags"}</dd>
              <dt>Top rule</dt><dd>{st.data.rules.fired_rules[0] ? `${st.data.rules.fired_rules[0].rule} (${fmt.pct(st.data.rules.fired_rules[0].confidence, 0)})` : "none"}</dd>
              <dt>Cluster</dt><dd>{st.data.cluster.cluster} ({fmt.pct(st.data.cluster.fraud_rate, 0)} fraud)</dd>
              <dt>Sanctions</dt><dd>{st.data.sanctions.map((x) => `${x.name}: ${x.hit ? `match ${x.best_score}` : "no match"}`).join("; ")}</dd>
            </dl>
          </div>
        </div>
      )}
    </Card>
  );
}

function SuspiciousTable({ s, onPick }: { s: Summary; onPick: (r: Row) => void }) {
  const [tier, setTier] = useState("CRITICAL,HIGH");
  const [offset, setOffset] = useState(0);
  const page = useApi<{ total: number; items: Row[] }>(`/analyses/${s.analysis_id}/transactions`, { tier, limit: 15, offset }, [s.analysis_id]);
  const hasLabels = s.top.length > 0 && s.top[0].risk_label !== undefined;
  return (
    <Card title="Transactions by risk" aside={
      <select value={tier} onChange={(e) => { setTier(e.target.value); setOffset(0); }} aria-label="Tier filter">
        <option value="CRITICAL,HIGH">Alerts (HIGH + CRITICAL)</option><option value="CRITICAL">CRITICAL</option><option value="MEDIUM">MEDIUM (review)</option><option value="">All</option>
      </select>} note="Click a row for its explanation.">
      {page.error && <ErrorBox error={page.error} />}
      {page.data && (
        <div className={page.loading ? "refetch" : ""}>
          <Table rows={page.data.items} rowKey="transaction_id" onRow={onPick} columns={[
            { key: "risk_score", label: "Risk", right: true, render: (r) => <b className="num">{r.risk_score.toFixed(0)}</b> },
            { key: "tier", label: "Tier", render: (r) => <TierBadge tier={r.tier} /> },
            { key: "transaction_id", label: "Transaction", render: (r) => <span className="small">{r.transaction_id}</span> },
            { key: "amount_inr", label: "Amount", right: true, render: (r) => fmt.inr(r.amount_inr) },
            { key: "parties", label: "From → to", render: (r) => <span className="small">{r.sender_account_id.replace("ACC_IND_", "")} → {r.receiver_account_id.replace("ACC_IND_", "")}</span> },
            { key: "network_flags", label: "Network flags", render: (r) => <span className="small">{r.network_flags.replace(/_/g, " ").replace(/,/g, ", ").toLowerCase() || "—"}</span> },
            ...(hasLabels ? [{ key: "risk_label", label: "Truth", render: (r: Row) => (r.risk_label ? <LabelBadge label={r.risk_label} /> : "—") }] : []),
          ]} />
          <div className="filters" style={{ marginTop: 8, marginBottom: 0 }}>
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 15))}>← Previous</button>
            <span className="small muted num">{page.data.total ? `${offset + 1}–${Math.min(offset + 15, page.data.total)} of ${fmt.int(page.data.total)}` : "none"}</span>
            <button disabled={offset + 15 >= page.data.total} onClick={() => setOffset(offset + 15)}>Next →</button>
          </div>
        </div>
      )}
    </Card>
  );
}

function Results({ s }: { s: Summary }) {
  const [pick, setPick] = useState<Row | null>(null);
  useEffect(() => setPick(null), [s.analysis_id]);
  const groups = s.groups;
  return (
    <div className="stack">
      <div className="topbar results-head">
        <div><h2>Results</h2>
          <div className="small ink2">{fmt.int(s.kpis.transactions)} transactions analysed in {s.timings.total_seconds.toFixed(1)} s
            (features {s.timings.feature_seconds}s · scoring {s.timings.score_seconds}s · {fmt.int(s.timings.history_rows_used)} history rows as context) · results kept 30 min</div></div>
        <a className="btn" href={api.url(`/analyses/${s.analysis_id}/download.csv`)}><Icon name="download" />Download scored CSV</a>
      </div>
      <div className="grid g4">
        <Stat label="Transactions" value={fmt.int(s.kpis.transactions)} hint={`${fmt.int(s.kpis.accounts)} accounts · ${fmt.inr(s.kpis.amount_total_inr)}`} />
        <Stat label="Alerts (HIGH + CRITICAL)" value={fmt.int(s.kpis.alerts)} hint={`${fmt.pct(s.kpis.alert_rate)} of transactions`} />
        <Stat label="Critical" value={fmt.int(s.kpis.critical)} hint={`score ≥ ${s.cutoffs.CRITICAL.toFixed(0)}: block / escalate`} />
        <Stat label="Value at risk" value={fmt.inr(s.kpis.value_at_risk_inr)} hint="sum of alerted payments" />
      </div>
      <div className="grid g2">
        <Card title="Risk tiers" note={`MEDIUM ≥ ${s.cutoffs.MEDIUM.toFixed(0)} (review) · HIGH ≥ ${s.cutoffs.HIGH.toFixed(0)} (alert) · CRITICAL ≥ ${s.cutoffs.CRITICAL.toFixed(0)}`}>
          <Plot height={260} data={() => [{ type: "bar", x: TIER_ORDER, y: TIER_ORDER.map((t) => s.tiers[t]), marker: { color: TIER_ORDER.map((t) => css(TIER_VAR[t])) },
            text: TIER_ORDER.map((t) => fmt.int(s.tiers[t])), textposition: "outside", cliponaxis: false, hovertemplate: "%{x}: <b>%{y:,}</b><extra></extra>" }]}
            layout={() => ({ showlegend: false, yaxis: { title: { text: "transactions" }, rangemode: "tozero" } })} />
        </Card>
        <Card title="Transactions over time" aside="alerts vs the rest">
          <Plot height={260} data={() => [
            { type: "bar", name: "Not alerted", x: s.timeline.map((d) => d.time), y: s.timeline.map((d) => d.n - d.alerts), marker: { color: css("--mark") }, hovertemplate: "%{x}<br>%{y} not alerted<extra></extra>" },
            { type: "bar", name: "Alerted", x: s.timeline.map((d) => d.time), y: s.timeline.map((d) => d.alerts), marker: { color: css("--s2") }, hovertemplate: "%{x}<br><b>%{y}</b> alerted<extra></extra>" },
          ]} layout={() => ({ barmode: "stack", bargap: 0.1, legend: { orientation: "h", y: 1.12, x: 0 } })} />
        </Card>
      </div>
      <div className="grid g2">
        <DataQuality s={s} />
        {s.evaluation ? <Evaluation e={s.evaluation} /> : <Card title="Accuracy check"><p className="small muted">The input has no labels (is_fraud / risk_label), so accuracy cannot be measured; the results are the system's assessment only.</p></Card>}
      </div>
      <SuspiciousTable s={s} onPick={setPick} />
      {pick && <RowDetail aid={s.analysis_id} row={pick} onClose={() => setPick(null)} />}
      <div className="grid g2">
        <Card title="Suspicious structures (network view)" note="The strongest structures among alerted payments: fan-in hubs (red node = hub receiving from ≥ 3 senders), pass-through chains and bursts. Hover a line or node for details.">
          <GroupGraph g={groups.graph} />
        </Card>
        <Card title="Suspicious structures found">
          {[["Fan-in hubs", groups.hubs, "receiver_account_id", (r: Record<string, number | string>) => `${r.senders} senders · ${fmt.inr(Number(r.amount_inr))}`],
            ["Pass-through chains (mules / layering)", groups.chains, "sender_account_id", (r: Record<string, number | string>) => `${r.forwards} forward(s) · chain depth ${r.max_chain_depth}`],
            ["Bursts", groups.bursts, "sender_account_id", (r: Record<string, number | string>) => `${r.alerts} alerts · up to ${r.max_tx_in_1h} payments in 1 h`],
            ["Shared devices", groups.shared_devices, "device_id", (r: Record<string, number | string>) => `${r.accounts} accounts on one device`]].map(([title, rows, key, txt]) => (
            <Fragment key={title as string}>
              <h3 style={{ marginTop: 10 }}>{title as string} <span className="muted">({(rows as unknown[]).length})</span></h3>
              {(rows as Record<string, number | string>[]).length ? (rows as Record<string, number | string>[]).slice(0, 5).map((r) => (
                <div key={String(r[key as string])} className="small"><code>{String(r[key as string])}</code> <span className="ink2">{(txt as (r: Record<string, number | string>) => string)(r)}</span></div>
              )) : <div className="small muted">none found</div>}
            </Fragment>
          ))}
        </Card>
      </div>
      <div className="grid g2">
        <Card title="Most common rules among alerts" note="Phase 4 association rules (→ FRAUDULENT) that fired on alerted payments, counted by the rule that fired first.">
          {s.rules.length ? (
            <Plot height={Math.max(160, 30 * s.rules.length + 40)} data={() => {
              const r = [...s.rules].reverse();
              return [{ type: "bar", orientation: "h", y: r.map((x) => x.rule.replace(" → FRAUDULENT", "")), x: r.map((x) => x.alerts), marker: { color: css("--s2") },
                hovertemplate: "%{y}<br><b>%{x}</b> alerts<extra></extra>" }];
            }} layout={() => ({ showlegend: false, xaxis: { title: { text: "alerts" } } })} />
          ) : <p className="small muted">No mined rule fired on the alerts.</p>}
        </Card>
        <Card title="Unusual but not alerted" aside={`${fmt.int(s.anomalies.count)} transactions`} note={s.anomalies.note}>
          {s.anomalies.top.length ? <Table rows={s.anomalies.top} columns={[
            { key: "transaction_id", label: "Transaction", render: (r) => <span className="small">{r.transaction_id}</span> },
            { key: "anomaly_pct", label: "Anomaly", right: true, render: (r) => fmt.pct(r.anomaly_pct, 1) },
            { key: "risk_score", label: "Risk", right: true, render: (r) => r.risk_score.toFixed(0) },
            { key: "tier", label: "Tier", render: (r) => <TierBadge tier={r.tier} /> },
            { key: "amount_inr", label: "Amount", right: true, render: (r) => fmt.inr(r.amount_inr) },
          ]} /> : <p className="small muted">Nothing extremely unusual outside the alerts.</p>}
        </Card>
      </div>
    </div>
  );
}

export default function AnalyzePage() {
  const schema = useApi<Schema>("/analyze/schema");
  const [busy, setBusy] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [err, setErr] = useState<ApiError | null>(null);
  const [res, setRes] = useState<Summary | null>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const [params] = useSearchParams();
  const autoRan = useRef(false);                  // React StrictMode runs effects twice in development
  // Shareable demo links: /?run=demo | /?run=sample | /?run=simulate&attacks=fraud_ring,burst&camouflage=1&size=12
  useEffect(() => {
    if (autoRan.current) return;
    autoRan.current = true;
    const mode = params.get("run");
    if (mode === "demo") run("demo dataset (10,000)", () => api.post<Summary>("/analyze/demo", {}, 120000));
    else if (mode === "sample") run("demo sample (2,000)", () => api.post<Summary>("/analyze/demo", { sample: 2000 }, 120000));
    else if (mode === "simulate") {
      const attacks = (params.get("attacks") ?? "fraud_ring,mule_chain").split(",").filter(Boolean);
      const camouflage = params.get("camouflage") === "1";
      run(`simulation: ${attacks.join(" + ")}${camouflage ? " (camouflaged)" : ""}`, () => api.post<Summary>("/analyze/simulate",
        { attacks, camouflage, size: Number(params.get("size") ?? 10), background: Number(params.get("background") ?? 1500), seed: Number(params.get("seed") ?? 42) }, 120000));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const run = async (label: string, fn: () => Promise<Summary>) => {
    setBusy(label); setErr(null);
    const t0 = performance.now();
    const tick = setInterval(() => setElapsed((performance.now() - t0) / 1000), 200);
    try {
      const s = await fn();
      setRes(s);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    } catch (e) { setErr(e as ApiError); } finally { clearInterval(tick); setBusy(null); }
  };
  return (
    <>
      <PageHead className="home-head" title="See the risk, follow the money, know why."
        sub="Run FinSight on a batch of payments: it checks the data, builds behavioural and network features, scores every transaction 0–100 with the calibrated model, the anomaly detector, network red flags, mined rules and sanctions screening, then shows what it found." />
      <ul className="signals" aria-label="Signals behind every score">{SIGNALS.map(([icon, label]) => <li key={label}><Icon name={icon} />{label}</li>)}</ul>
      <Inputs schema={schema.data} busy={!!busy} onRun={run} />
      {busy && <div className="card progress" role="status"><span className="spinner" aria-hidden /><div><b>Analyzing {busy}…</b> <span className="muted num">{elapsed.toFixed(1)} s</span>
        <div className="small muted">validate → defaults → features (history + batch) → model, anomaly, network, rules, sanctions → summary</div></div></div>}
      {err && <div style={{ marginTop: 14 }}><ErrorBox error={err} /></div>}
      <div ref={resultsRef} className="results">{res && !busy && <Results s={res} />}</div>
      {!res && !busy && <p className="empty" style={{ marginTop: 18 }}>Start with the demo dataset or the attack simulator; results appear here. The other pages in the menu explore the demo dataset and the experiments in depth.</p>}
    </>
  );
}
