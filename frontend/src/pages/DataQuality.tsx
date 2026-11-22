import { fmt, useApi } from "../api";
import { css, Plot } from "../components/Plot";
import { Async, Card, PageHead, Table } from "../components/ui";

type DQ = {
  quality: {
    tables: { table: string; rows: number; columns: number; completeness: number; duplicate_rows: number; constant_columns: string[] }[];
    referential_integrity: { fk: string; rows: number; matched_rate: number; orphans: number }[];
    leakage_scan: { feature: string; auc: number; flag: string }[];
    label_proxies: { column: string; nonzero_rows: number; fraud_rate_when_nonzero: number; fraud_rate_when_zero: number }[];
    cleaning_log: { step: string; rows_removed?: number; reason?: string }[];
    redundant_pairs: { a: string; b: string }[];
  };
  eda: { rq1_feature_ranking_v1_1: { feature: string; auc: number; mutual_info: number; group: string }[];
         flags: { flag: string; support: number; fraud_rate_given_flag: number; lift: number }[] };
  sanctions: { entities: number; name_variants: number; by_source: Record<string, number>;
               customer_screening_threshold_sweep: { threshold: number; alerts: number; false_positive_rate: number }[] };
  items: { item: string; rule: string; count: number; support: number; fraud_rate: number | null }[];
};

const GROUPS = ["A transaction", "B behavioural", "C network"];

export default function DataQualityPage() {
  const st = useApi<DQ>("/data-quality");
  return (
    <>
      <PageHead title="Data quality & audit" sub="KDD steps 1–3: what was loaded, how clean it is, and which columns had to be excluded because they leak the label." />
      <Async state={st}>{(d) => {
        const leak = [...d.quality.leakage_scan].sort((a, b) => b.auc - a.auc).slice(0, 15).reverse();
        const rank = [...d.eda.rq1_feature_ranking_v1_1].slice(0, 15).reverse();
        return (
          <div className="stack">
            <div className="grid g2">
              <Card title="Tables" note="All seven files load with full completeness; one constant column (status) and one 1:1 recoding (transaction_type ≡ channel) were dropped.">
                <Table rows={d.quality.tables} columns={[
                  { key: "table", label: "Table" }, { key: "rows", label: "Rows", right: true, render: (r) => fmt.int(r.rows) },
                  { key: "columns", label: "Cols", right: true }, { key: "completeness", label: "Complete", right: true, render: (r) => fmt.pct(r.completeness) },
                  { key: "duplicate_rows", label: "Duplicates", right: true },
                  { key: "constant_columns", label: "Constant columns", render: (r) => r.constant_columns.join(", ") || "—" },
                ]} />
              </Card>
              <Card title="Referential integrity" note="Every foreign key matches; the beneficiary pair-join matched only 8 rows, so beneficiaries are aggregated per sender instead.">
                <Table rows={d.quality.referential_integrity} columns={[
                  { key: "fk", label: "Relationship" }, { key: "rows", label: "Rows", right: true, render: (r) => fmt.int(r.rows) },
                  { key: "matched_rate", label: "Matched", right: true, render: (r) => fmt.pct(r.matched_rate, 2) },
                  { key: "orphans", label: "Orphans", right: true },
                ]} />
              </Card>
            </div>
            <div className="grid g2">
              <Card title="Leakage scan: single-feature AUC" note="▲ SUSPECT = a column that alone predicts the label almost perfectly (an alert flag set only on fraud and borderline). Blacklisted and enforced in code.">
                <Plot height={400} data={() => [{
                  type: "bar", orientation: "h", y: leak.map((r) => (r.flag ? "▲ " : "") + fmt.feature(r.feature)), x: leak.map((r) => r.auc),
                  marker: { color: leak.map((r) => (r.flag ? css("--critical") : css("--mark"))) },
                  text: leak.map((r) => r.auc.toFixed(3) + (r.flag ? "  SUSPECT" : "")), textposition: "outside", cliponaxis: false,
                  hovertemplate: "%{y}<br>AUC <b>%{x:.3f}</b><extra></extra>",
                }]} layout={() => ({ showlegend: false, xaxis: { range: [0.45, 1.08], title: { text: "AUC of the column alone vs is_fraud" } }, margin: { l: 10, r: 60, t: 6, b: 10 } })} />
              </Card>
              <Card title="RQ1: strongest individual signals (v1.1)" note="Single-feature AUC, coloured by feature group. Amount context and payee recency lead; network features matter in combination (Phase 5).">
                <Plot height={400} data={() => GROUPS.map((g, gi) => {
                  const rows = rank.filter((r) => r.group === g);
                  return { type: "bar", orientation: "h", name: g, y: rows.map((r) => fmt.feature(r.feature)), x: rows.map((r) => r.auc),
                    marker: { color: css(`--s${gi + 1}`) }, customdata: rows.map((r) => r.mutual_info),
                    hovertemplate: "%{y}<br>AUC <b>%{x:.3f}</b> · MI %{customdata:.4f}<extra>" + g + "</extra>" };
                })} layout={() => ({ barmode: "overlay", yaxis: { categoryorder: "array", categoryarray: rank.map((r) => fmt.feature(r.feature)) },
                  xaxis: { range: [0.45, 0.9], title: { text: "single-feature AUC" } }, legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0 }, margin: { l: 10, r: 14, t: 44, b: 10 } })} />
              </Card>
            </div>
            <div className="grid g2">
              <Card title="Behaviour flags" note="Support = share of all transactions; lift = fraud rate given the flag ÷ base rate (13.5%).">
                <Table rows={d.eda.flags} columns={[
                  { key: "flag", label: "Flag" }, { key: "support", label: "Support", right: true, render: (r) => fmt.pct(r.support) },
                  { key: "fraud_rate_given_flag", label: "Fraud rate | flag", right: true, render: (r) => fmt.pct(r.fraud_rate_given_flag) },
                  { key: "lift", label: "Lift", right: true, render: (r) => r.lift.toFixed(2) },
                ]} />
              </Card>
              <Card title="Sanctions screening threshold" aside={`${fmt.int(d.sanctions.entities)} entities · ${fmt.int(d.sanctions.name_variants)} names`}
                    note="Every customer name is synthetic (not sanctioned), so each alert is a false positive. ≥ 90 keeps false alarms at 0.17%, the threshold used by the risk score override.">
                <Plot height={300} data={() => {
                  const s = d.sanctions.customer_screening_threshold_sweep;
                  return [{ type: "scatter", mode: "lines+markers", x: s.map((r) => r.threshold), y: s.map((r) => r.false_positive_rate),
                    line: { color: css("--mark"), width: 2 }, marker: { size: 8 }, text: s.map((r) => `${r.alerts} alerts`),
                    hovertemplate: "threshold %{x}<br><b>%{y:.2%}</b> of customers flagged (%{text})<extra></extra>" }];
                }} layout={() => ({ showlegend: false, xaxis: { title: { text: "fuzzy-match threshold (RapidFuzz token sort ratio)" } }, yaxis: { tickformat: ".0%", title: { text: "false-positive rate" } } })} />
              </Card>
            </div>
            <div className="grid g2">
              <Card title="Label proxies excluded">
                <Table rows={d.quality.label_proxies} columns={[
                  { key: "column", label: "Column" }, { key: "nonzero_rows", label: "Non-zero rows", right: true },
                  { key: "fraud_rate_when_nonzero", label: "Fraud rate if set", right: true, render: (r) => fmt.pct(r.fraud_rate_when_nonzero) },
                  { key: "fraud_rate_when_zero", label: "Fraud rate if not", right: true, render: (r) => fmt.pct(r.fraud_rate_when_zero) },
                ]} />
              </Card>
              <Card title="Cleaning log">
                <Table rows={d.quality.cleaning_log} columns={[
                  { key: "step", label: "Step" },
                  { key: "detail", label: "Detail", render: (r) => (r.reason ? r.reason.replace(/_/g, " ") : r.rows_removed != null ? `${r.rows_removed} rows removed` : "—") },
                ]} />
              </Card>
            </div>
          </div>
        );
      }}</Async>
    </>
  );
}
