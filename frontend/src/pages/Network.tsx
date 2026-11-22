import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { fmt, useApi } from "../api";
import { Async, Card, PageHead, Seg, Stat, Table } from "../components/ui";

type Acc = { account_id: string; in_degree: number; out_degree: number; pagerank: number; community_id: number; community_size: number; clustering: number; fraud_received: number };
type Link = { source: string; target: string; n: number; amount_inr: number; last: string; fraud: number };
type Net = {
  account_id: string; metrics: Record<string, number>; community_members: number;
  activity: { sent: number; received: number; amount_out_inr: number; amount_in_inr: number; distinct_receivers: number; distinct_senders: number; fraud_involved: number };
  ego: { nodes: { id: string; role: string; pagerank: number | null; community_id: number | null }[]; links: Link[] };
};

function Ego({ net, onPick }: { net: Net; onPick: (id: string) => void }) {
  const W = 760, H = 560, cx = W / 2, cy = H / 2, R = 230;
  const focus = net.account_id;
  const layout = useMemo(() => {
    const ins = net.ego.links.filter((l) => l.target === focus).map((l) => l.source);
    const outs = net.ego.links.filter((l) => l.source === focus).map((l) => l.target).filter((id) => !ins.includes(id));
    const pos: Record<string, [number, number]> = { [focus]: [cx, cy] };
    const place = (ids: string[], a0: number, a1: number) => ids.forEach((id, i) => {
      const a = ids.length === 1 ? (a0 + a1) / 2 : a0 + ((a1 - a0) * i) / (ids.length - 1);
      pos[id] = [cx + R * Math.cos(a), cy + R * 0.9 * Math.sin(a)];
    });
    place(ins, (Math.PI * 5) / 8, (Math.PI * 11) / 8);    // senders on the left
    place(outs, (-Math.PI * 3) / 8, (Math.PI * 3) / 8);   // receivers on the right
    return pos;
  }, [net, focus, cx, cy]);
  const maxN = Math.max(1, ...net.ego.links.map((l) => l.n));
  const pr = (p: number | null) => 5 + Math.sqrt((p ?? 0) * 4000);
  const node = Object.fromEntries(net.ego.nodes.map((n) => [n.id, n]));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={`Ego network of ${focus}`} style={{ maxHeight: 560 }}>
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted)" /></marker>
        <marker id="arrf" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="var(--s2)" /></marker>
      </defs>
      <text x={60} y={22} fill="var(--muted)" fontSize="12">senders →</text>
      <text x={W - 60} y={22} fill="var(--muted)" fontSize="12" textAnchor="end">→ receivers</text>
      {net.ego.links.map((l, i) => {
        const [x1, y1] = layout[l.source] ?? [cx, cy], [x2, y2] = layout[l.target] ?? [cx, cy];
        const fraud = l.fraud > 0;
        const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1, r2 = pr(node[l.target]?.pagerank ?? null) + 3;
        return (
          <line key={i} x1={x1} y1={y1} x2={x2 - (dx / len) * r2} y2={y2 - (dy / len) * r2} stroke={fraud ? "var(--s2)" : "var(--baseline)"}
                strokeWidth={1 + (3 * l.n) / maxN} markerEnd={fraud ? "url(#arrf)" : "url(#arr)"} opacity={0.9}>
            <title>{`${l.source} → ${l.target}\n${l.n} transactions · ${fmt.inr(l.amount_inr)}${fraud ? ` · ${l.fraud} fraudulent` : ""}`}</title>
          </line>
        );
      })}
      {Object.entries(layout).map(([id, [x, y]]) => {
        const n = node[id];
        const isFocus = id === focus;
        return (
          <g key={id} onClick={() => !isFocus && onPick(id)} style={{ cursor: isFocus ? "default" : "pointer" }}>
            <circle cx={x} cy={y} r={Math.max(12, pr(n?.pagerank ?? null) + 6)} fill="transparent" />{/* larger hit target */}
            <circle cx={x} cy={y} r={pr(n?.pagerank ?? null)} fill={isFocus ? "var(--accent)" : "var(--surface-2)"} stroke={isFocus ? "var(--surface)" : "var(--ink-2)"} strokeWidth={2} />
            {/* labels point outwards (left of senders, right of receivers) so a crowded arc stays readable */}
            <text x={isFocus ? x : x < cx ? x - pr(n?.pagerank ?? null) - 5 : x + pr(n?.pagerank ?? null) + 5} y={isFocus ? y + pr(n?.pagerank ?? null) + 15 : y + 4}
                  textAnchor={isFocus ? "middle" : x < cx ? "end" : "start"} fontSize={isFocus ? 12 : 10} fill={isFocus ? "var(--ink)" : "var(--ink-2)"} fontWeight={isFocus ? 700 : 400}>{id.replace("ACC_IND_", "")}</text>
            <title>{`${id}${n?.pagerank != null ? `\nPageRank ${n.pagerank.toFixed(5)}` : ""}${n?.community_id != null ? ` · community ${n.community_id}` : ""}${isFocus ? "" : "\nclick to explore"}`}</title>
          </g>
        );
      })}
    </svg>
  );
}

export default function NetworkPage() {
  const [params, setParams] = useSearchParams();
  const [sort, setSort] = useState<"pagerank" | "in_degree" | "out_degree">("pagerank");
  const [query, setQuery] = useState("");
  const top = useApi<{ items: Acc[] }>("/accounts", { sort, limit: 15, query: query || undefined });
  const [acc, setAcc] = useState<string | null>(params.get("account"));
  useEffect(() => { if (!acc && top.data?.items.length) setAcc(top.data.items[0].account_id); }, [acc, top.data]);
  const pick = (id: string) => { setAcc(id); setParams({ account: id }, { replace: true }); };
  const net = useApi<Net>(acc ? `/network/${acc}` : null, { limit: 30 }, [acc]);
  return (
    <>
      <PageHead title="Network explorer" sub="Accounts as nodes, payments as directed edges. Organised fraud (rings, mules, layering) is visible here even when each payment looks normal." />
      <div className="grid explorer">
        <Card title="Accounts" aside={<Seg value={sort} onChange={setSort} options={[{ value: "pagerank", label: "PageRank" }, { value: "in_degree", label: "In" }, { value: "out_degree", label: "Out" }]} />}>
          <div className="filters"><input type="text" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search account id…" aria-label="Search account id" /></div>
          <Async state={top}>{(t) => (
            <Table rows={t.items} rowKey="account_id" selected={acc ?? undefined} onRow={(r) => pick(r.account_id)} columns={[
              { key: "account_id", label: "Account", render: (r) => <span className="small">{r.account_id}</span> },
              { key: "pagerank", label: "PageRank", right: true, render: (r) => r.pagerank.toFixed(4) },
              { key: "in_degree", label: "In", right: true }, { key: "out_degree", label: "Out", right: true },
              { key: "fraud_received", label: "Fraud in", right: true },
            ]} />
          )}</Async>
        </Card>
        <div className="stack">
          {!acc && <Card title="Network"><p className="empty">Select an account.</p></Card>}
          {acc && <Async state={net}>{(n) => (
            <div className="stack">
              <div className="grid g2">
                <Stat label="Account" value={<span style={{ fontSize: "1.3rem" }}>{n.account_id}</span>} hint={`community ${n.metrics.community_id} · ${n.community_members} members`} />
                <Stat label="PageRank" value={n.metrics.pagerank.toFixed(4)} hint={`in ${n.metrics.in_degree} · out ${n.metrics.out_degree}`} />
                <Stat label="Money in / out" value={<span style={{ fontSize: "1.3rem" }}>{fmt.inr(n.activity.amount_in_inr)} / {fmt.inr(n.activity.amount_out_inr)}</span>}
                      hint={`${n.activity.distinct_senders} senders · ${n.activity.distinct_receivers} receivers`} />
                <Stat label="Fraudulent transactions" value={fmt.int(n.activity.fraud_involved)} hint={`of ${fmt.int(n.activity.sent + n.activity.received)} involving this account`} />
              </div>
              <Card title="Ego network" note={<>Orange edges carried at least one fraudulent payment; width = number of payments; node size = PageRank. Click a peer to move the focus. Hover for details.</>}>
                <Ego net={n} onPick={pick} />
              </Card>
              <Card title="Counterparties">
                <Table rows={n.ego.links} columns={[
                  { key: "dir", label: "Direction", render: (r) => (r.source === n.account_id ? "→ sent to" : "← received from") },
                  { key: "peer", label: "Account", render: (r) => { const id = r.source === n.account_id ? r.target : r.source; return <a href="#" onClick={(e) => { e.preventDefault(); pick(id); }}>{id}</a>; } },
                  { key: "n", label: "Payments", right: true }, { key: "amount_inr", label: "Amount", right: true, render: (r) => fmt.inr(r.amount_inr) },
                  { key: "fraud", label: "Fraudulent", right: true }, { key: "last", label: "Last", render: (r) => fmt.date(r.last) },
                ]} max={30} tall />
              </Card>
            </div>
          )}</Async>}
        </div>
      </div>
    </>
  );
}
