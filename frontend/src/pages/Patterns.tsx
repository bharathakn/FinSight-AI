import { useState } from "react";
import { fmt, useApi } from "../api";
import { css, Plot } from "../components/Plot";
import { Async, Card, LabelBadge, PageHead, Seg, Table } from "../components/ui";

type Rule = { rule: string; antecedents: string[]; consequent: string; n_antecedent: number; support: number; confidence: number; lift: number; has_network_item: boolean };
type Patterns = { total: number; items: Rule[]; item_definitions: Record<string, string>; network_items: string[] };
type Run = { items: string; n_items: number; n_rows: number; algorithm: string; min_support: number; itemsets: number; seconds: number; peak_mb: number };
type Bench = { runs: Run[]; apriori_equals_fpgrowth: boolean; hardest_point: Record<string, { itemsets: number; seconds: number; peak_mb: number }> };

const ALGOS = ["apriori", "fpgrowth", "fpmax"];
const ALGO_NAME: Record<string, string> = { apriori: "Apriori", fpgrowth: "FP-Growth", fpmax: "FP-Max (maximal)" };

export default function PatternsPage() {
  const [consequent, setConsequent] = useState<"FRAUDULENT" | "BORDERLINE" | "">("FRAUDULENT");
  const [minConf, setMinConf] = useState(0.5);
  const [minLift, setMinLift] = useState(1.5);
  const [networkOnly, setNetworkOnly] = useState(false);
  const [universe, setUniverse] = useState<"extended" | "core">("extended");
  const all = useApi<Patterns>("/patterns", { limit: 500 });
  const filtered = useApi<Patterns>("/patterns", { consequent, min_confidence: minConf, min_lift: minLift, network_only: networkOnly, limit: 100 });
  const bench = useApi<Bench>("/patterns/benchmark");

  return (
    <>
      <PageHead title="Pattern mining lab" sub="Association rules mined from discretised transactions (Phase 4). Only productive rules are kept: each beats every simpler rule it contains. Labels appear only as consequents." />
      <div className="grid g2">
        <Card title="All rules: support vs confidence" aside="bubble size = lift" note="Rules with a network item (pass-through, quick inflow, hub receiver) reach the highest confidence; they exist only in the v1.1 dataset with injected structure.">
          <Async state={all}>{(p) => (
            <Plot height={340} data={() => ["FRAUDULENT", "BORDERLINE"].map((c) => {
              const r = p.items.filter((x) => x.consequent === c);
              return { type: "scatter", mode: "markers", name: "→ " + c.toLowerCase(), x: r.map((x) => x.support), y: r.map((x) => x.confidence),
                marker: { size: r.map((x) => 6 + 2.2 * x.lift), color: css(c === "FRAUDULENT" ? "--s2" : "--s3"), opacity: 0.75, line: { color: css("--surface"), width: 1.5 } },
                text: r.map((x) => x.rule), customdata: r.map((x) => x.lift),
                hovertemplate: "%{text}<br>support %{x:.2%} · confidence <b>%{y:.0%}</b> · lift %{customdata:.2f}<extra></extra>" };
            })} layout={() => ({ xaxis: { type: "log", title: { text: "support (share of all transactions, log)" }, tickformat: ".1%" },
              yaxis: { title: { text: "confidence" }, tickformat: ".0%" }, legend: { orientation: "h", y: 1.1, x: 0 } })} />
          )}</Async>
        </Card>
        <Card title="Apriori vs FP-Growth" aside={<Seg value={universe} onChange={setUniverse} options={[{ value: "extended", label: "49 items" }, { value: "core", label: "22 items" }]} />}
              note="Identical itemsets from every algorithm (checked). Apriori's candidate generation explodes memory as min_support falls; FP-Growth's compressed tree stays small.">
          <Async state={bench}>{(b) => {
            const runs = b.runs.filter((r) => r.items === universe);
            const trace = (y: "seconds" | "peak_mb") => ALGOS.map((a, i) => {
              const r = runs.filter((x) => x.algorithm === a).sort((u, v) => u.min_support - v.min_support);
              return { type: "scatter", mode: "lines+markers", name: ALGO_NAME[a], x: r.map((x) => x.min_support), y: r.map((x) => x[y]),
                line: { color: css(`--s${i + 1}`), width: 2 }, marker: { size: 7 }, customdata: r.map((x) => x.itemsets),
                hovertemplate: `${ALGO_NAME[a]} · min_support %{x}<br><b>%{y:.3g}</b> ${y === "seconds" ? "s" : "MB"} · %{customdata:,} itemsets<extra></extra>` };
            });
            const sup = [...new Set(runs.map((r) => r.min_support))].sort((a, b) => b - a);
            const lay = (t: string) => () => ({ xaxis: { type: "log", autorange: "reversed", tickvals: sup, ticktext: sup.map(String), title: { text: "min_support (log; harder →)" } },
              yaxis: { type: "log", dtick: 1, title: { text: t } }, legend: { orientation: "h", y: -0.32, x: 0 }, margin: { l: 10, r: 10, t: 6, b: 10 } });
            return (<div className="grid g2"><Plot height={300} data={() => trace("seconds")} layout={lay("runtime (s, log)")} /><Plot height={300} data={() => trace("peak_mb")} layout={lay("peak memory (MB, log)")} /></div>);
          }}</Async>
        </Card>
      </div>
      <div className="filters" style={{ marginTop: 14 }}>
        <label className="field">Consequent
          <Seg value={consequent} onChange={setConsequent} options={[{ value: "FRAUDULENT", label: "Fraudulent" }, { value: "BORDERLINE", label: "Borderline" }, { value: "", label: "Both" }]} /></label>
        <label className="field">Min confidence <input type="number" step={0.05} min={0} max={1} value={minConf} onChange={(e) => setMinConf(Number(e.target.value))} /></label>
        <label className="field">Min lift <input type="number" step={0.5} min={0} value={minLift} onChange={(e) => setMinLift(Number(e.target.value))} /></label>
        <label className="check"><input type="checkbox" checked={networkOnly} onChange={(e) => setNetworkOnly(e.target.checked)} /> network items only</label>
      </div>
      <div className="grid g3">
        <Card className="span2" title="Rules" aside={filtered.data ? `${filtered.data.total} match` : ""}>
          <Async state={filtered}>{(p) => (
            <Table rows={p.items} columns={[
              { key: "antecedents", label: "If (all of)", render: (r) => r.antecedents.map((a) => <span key={a} className={"chip" + (p.network_items.includes(a) ? " on" : "")} title={p.item_definitions[a]}>{a}</span>) },
              { key: "consequent", label: "Then", render: (r) => <LabelBadge label={r.consequent} /> },
              { key: "support", label: "Support", right: true, render: (r) => fmt.pct(r.support, 2) },
              { key: "confidence", label: "Confidence", right: true, render: (r) => fmt.pct(r.confidence, 0) },
              { key: "lift", label: "Lift", right: true, render: (r) => r.lift.toFixed(2) },
            ]} max={60} />
          )}</Async>
        </Card>
        <Card title="Items" note="Each transaction becomes a basket of these true/false items. Outlined chips in the table are network items (Phase 3 features).">
          <Async state={all}>{(p) => (
            <Table rows={Object.entries(p.item_definitions).map(([item, rule]) => ({ item, rule }))} columns={[
              { key: "item", label: "Item", render: (r) => <b className="small">{r.item}</b> }, { key: "rule", label: "Definition", render: (r) => <code>{r.rule}</code> },
            ]} />
          )}</Async>
        </Card>
      </div>
    </>
  );
}
