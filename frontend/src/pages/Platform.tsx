import { useEffect, useState } from "react";
import { api, API_BASE, fmt, TX_EXAMPLES, useApi } from "../api";
import { css, Plot } from "../components/Plot";
import { Async, Card, PageHead, Stat, Table } from "../components/ui";

type Route = { requests: number; errors_5xx: number; p50_ms: number; p95_ms: number; p99_ms: number; mean_ms: number };
type Status = {
  mode: string; gateway: { routes: Record<string, Route>; rss_mb: number; uptime_s: number; host: string };
  services: Record<string, Record<string, unknown>>; bundle: { version: string; created_utc: string; total_mb: number; libraries: Record<string, string> };
};
type Bench = {
  warm?: Record<string, { warm: Record<string, number>; rss_mb: Record<string, number>; mode: string }>;
  load?: Record<string, { scenario: string; rps_total: number; p50_ms: number; p95_ms: number; p99_ms: number; failures: number }>;
  images_mb?: Record<string, number | null>; startup_s?: Record<string, number>;
};

export default function PlatformPage({ wakeSeconds }: { wakeSeconds: number }) {
  const [tick, setTick] = useState(0);
  const status = useApi<Status>("/platform/status", undefined, [tick]);
  const bench = useApi<Bench>("/platform/benchmarks");
  const [auto, setAuto] = useState(true);
  const [probe, setProbe] = useState<number[]>([]);
  const [probing, setProbing] = useState(false);
  useEffect(() => {
    if (!auto) return;
    const t = setInterval(() => setTick((x) => x + 1), 10000);
    return () => clearInterval(t);
  }, [auto]);

  const runProbe = async () => {
    setProbing(true); setProbe([]);
    const ids = Object.values(TX_EXAMPLES);
    const out: number[] = [];
    for (let i = 0; i < 30; i++) {
      const t0 = performance.now();
      try { await api.post("/score-transaction", { transaction_id: ids[i % ids.length], explain: true }); } catch { /* counted as a slow sample */ }
      out.push(performance.now() - t0);
      setProbe([...out]);
    }
    setProbing(false);
    setTick((x) => x + 1);
  };
  const q = (v: number[], p: number) => { const s = [...v].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.round((p / 100) * (s.length - 1)))]; };

  return (
    <>
      <PageHead title="Platform" sub="The cloud side: which services are running, how fast they answer, how much memory they use, and what was measured under load.">
        <label className="check"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> refresh every 10 s</label>
      </PageHead>
      <Async state={status}>{(s) => {
        const routes = Object.entries(s.gateway.routes).map(([route, r]) => ({ route, ...r })).sort((a, b) => b.requests - a.requests);
        return (
          <div className="stack">
            <div className="grid g4">
              <Stat label="Deployment mode" value={s.mode} hint={s.mode === "inprocess" ? "gateway + 4 modules in one container" : "gateway + 4 microservices over HTTP"} />
              <Stat label="Backend wake-up (this visit)" value={`${wakeSeconds.toFixed(1)} s`} hint={wakeSeconds > 5 ? "the service was asleep (scale-to-zero)" : "the service was already warm"} />
              <Stat label="Gateway memory" value={`${fmt.int(s.gateway.rss_mb)} MB`} hint={`uptime ${fmt.int(s.gateway.uptime_s)} s · host ${s.gateway.host}`} />
              <Stat label="Model bundle" value={<span style={{ fontSize: "1.05rem" }}>{s.bundle.version}</span>} hint={`${s.bundle.total_mb} MB · scikit-learn ${s.bundle.libraries["scikit-learn"]} · xgboost ${s.bundle.libraries["xgboost-cpu"]}`} />
            </div>
            <div className="grid g2">
              <Card title="Services" aside={API_BASE}>
                <Table rows={Object.entries(s.services).map(([name, v]) => ({ name, ...v }) as Record<string, unknown>)} columns={[
                  { key: "name", label: "Service", render: (r) => <b>{String(r.name)}</b> },
                  { key: "status", label: "Status", render: (r) => <span className="badge">{r.status === "ok" ? "✓ ok" : "▲ " + String(r.status)}</span> },
                  { key: "rss_mb", label: "Memory", right: true, render: (r) => (r.rss_mb != null ? `${fmt.int(Number(r.rss_mb))} MB` : "shared") },
                  { key: "latency_ms", label: "Health RTT", right: true, render: (r) => (r.latency_ms != null ? fmt.ms(Number(r.latency_ms)) : "in-process") },
                  { key: "host", label: "Host", render: (r) => <span className="small muted">{String(r.host ?? "—")}</span> },
                ]} />
              </Card>
              <Card title="Gateway routes" aside="since start, last 2,000 requests per route" note="With several workers, each reports its own counters.">
                <Table rows={routes} columns={[
                  { key: "route", label: "Route", render: (r) => <code>{r.route}</code> }, { key: "requests", label: "Requests", right: true },
                  { key: "p50_ms", label: "p50", right: true, render: (r) => fmt.ms(r.p50_ms) }, { key: "p95_ms", label: "p95", right: true, render: (r) => fmt.ms(r.p95_ms) },
                  { key: "errors_5xx", label: "5xx", right: true },
                ]} max={12} />
              </Card>
            </div>
            <Card title="Live latency probe" aside={probe.length ? `p50 ${fmt.ms(q(probe, 50))} · p95 ${fmt.ms(q(probe, 95))} (n=${probe.length})` : ""}
                  note="30 sequential POST /score-transaction calls from this browser. Includes the network round trip to the API host; the target is p95 < 200 ms when warm.">
              <button className="primary" onClick={runProbe} disabled={probing}>{probing ? "Running…" : "Run 30 requests"}</button>
              {probe.length > 0 && <Plot height={220} data={() => [{ type: "bar", x: probe.map((_, i) => i + 1), y: probe, marker: { color: css("--s1") },
                hovertemplate: "request %{x}: <b>%{y:.0f} ms</b><extra></extra>" }]}
                layout={() => ({ showlegend: false, xaxis: { title: { text: "request #" } }, yaxis: { title: { text: "ms" }, rangemode: "tozero" },
                  shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 200, y1: 200, line: { color: css("--muted"), dash: "dash", width: 1 } }] })} />}
            </Card>
          </div>
        );
      }}</Async>
      <div style={{ marginTop: 14 }}>
        <Async state={bench} what="Loading benchmarks">{(b) => (!b.load ? <Card title="Recorded benchmarks"><p className="muted small">No measurements bundled yet.</p></Card> : (
          <div className="grid g2">
            <Card title="Load test (recorded, Docker Compose on one laptop)" note="Locust, remote mode. Scaling ml-service to 3 replicas lowered median latency at 30 users; no request failed.">
              <Table rows={Object.values(b.load)} columns={[
                { key: "scenario", label: "Scenario" }, { key: "rps_total", label: "req/s", right: true, render: (r) => r.rps_total.toFixed(0) },
                { key: "p50_ms", label: "p50", right: true, render: (r) => fmt.ms(r.p50_ms) }, { key: "p95_ms", label: "p95", right: true, render: (r) => fmt.ms(r.p95_ms) },
                { key: "failures", label: "Errors", right: true },
              ]} />
            </Card>
            <Card title="Images and start-up (recorded)">
              <Table rows={Object.entries(b.images_mb ?? {}).map(([image, mb]) => ({ image, mb }))} columns={[
                { key: "image", label: "Image", render: (r) => <code>{r.image.replace(":local", "")}</code> },
                { key: "mb", label: "Size (uncompressed)", right: true, render: (r) => (r.mb == null ? "—" : `${fmt.int(r.mb)} MB`) },
              ]} />
              <p className="small ink2" style={{ marginTop: 8 }}>Start-up to healthy: single container {b.startup_s?.inprocess_container ?? "—"} s; full 5-service stack ~{b.startup_s?.remote_stack_all_healthy ?? "—"} s.</p>
            </Card>
          </div>
        ))}</Async>
      </div>
    </>
  );
}
