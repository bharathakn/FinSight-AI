import { ReactNode, useEffect, useState } from "react";
import { ApiError, API_BASE, api } from "../api";

export function Card({ title, aside, children, note, className = "" }: { title?: ReactNode; aside?: ReactNode; children: ReactNode; note?: ReactNode; className?: string }) {
  return (
    <section className={"card " + className}>
      {title && <h2><span>{title}</span>{aside && <span className="small muted">{aside}</span>}</h2>}
      {children}
      {note && <div className="note">{note}</div>}
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}

export function PageHead({ title, sub, children }: { title: string; sub?: ReactNode; children?: ReactNode }) {
  return (
    <div className="topbar">
      <div><h1>{title}</h1>{sub && <div className="sub">{sub}</div>}</div>
      {children}
    </div>
  );
}

export function Loading({ what = "Loading" }: { what?: string }) {
  return <div className="loading" role="status">{what}…</div>;
}

export function ErrorBox({ error, retry }: { error: ApiError | Error; retry?: () => void }) {
  const rid = (error as ApiError).requestId;
  return (
    <div className="error" role="alert">
      <b>Request failed.</b> {error.message}
      {rid && <span className="muted"> · request id {rid}</span>}
      {retry && <> · <button onClick={retry}>Retry</button></>}
    </div>
  );
}

/** Renders children once data exists; dims them while a refetch is running; shows errors inline. */
export function Async<T>({ state, children, what }: { state: { data: T | null; error: ApiError | null; loading: boolean; reload: () => void }; children: (d: T) => ReactNode; what?: string }) {
  if (state.error && !state.data) return <ErrorBox error={state.error} retry={state.reload} />;
  if (!state.data) return <Loading what={what} />;
  return <div className={state.loading ? "refetch" : ""}>{state.error && <ErrorBox error={state.error} retry={state.reload} />}{children(state.data)}</div>;
}

export const LABEL_VAR: Record<string, string> = { LEGITIMATE: "var(--s1)", FRAUDULENT: "var(--s2)", BORDERLINE: "var(--s3)" };

export function LabelBadge({ label }: { label: string }) {
  return <span className="badge"><span className="dot" style={{ background: LABEL_VAR[label] ?? "var(--muted)" }} />{label.charAt(0) + label.slice(1).toLowerCase()}</span>;
}

const TIER_ICON: Record<string, string> = { LOW: "○", MEDIUM: "◐", HIGH: "●", CRITICAL: "▲" };
export function TierBadge({ tier }: { tier: string }) {
  return <span className={"badge tier " + tier} title={`Risk tier ${tier}`}>{TIER_ICON[tier]} {tier}</span>;
}

export type Column<T> = { key: string; label: string; right?: boolean; render?: (row: T) => ReactNode };
export function Table<T extends Record<string, unknown>>({ rows, columns, onRow, selected, rowKey, max }: {
  rows: T[]; columns: Column<T>[]; onRow?: (r: T) => void; selected?: string; rowKey?: string; max?: number;
}) {
  const shown = max ? rows.slice(0, max) : rows;
  return (
    <div className="tablewrap">
      <table className="t">
        <thead><tr>{columns.map((c) => <th key={c.key} className={c.right ? "r" : ""}>{c.label}</th>)}</tr></thead>
        <tbody>
          {shown.map((r, i) => (
            <tr key={rowKey ? String(r[rowKey]) : i} className={(onRow ? "click " : "") + (selected && rowKey && r[rowKey] === selected ? "sel" : "")}
                onClick={onRow ? () => onRow(r) : undefined} tabIndex={onRow ? 0 : undefined}
                onKeyDown={onRow ? (e) => { if (e.key === "Enter") onRow(r); } : undefined}>
              {columns.map((c) => <td key={c.key} className={c.right ? "r" : ""}>{c.render ? c.render(r) : String(r[c.key] ?? "—")}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      {max && rows.length > max && <div className="note muted small">Showing {max} of {rows.length}.</div>}
    </div>
  );
}

export function Seg<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div className="seg" role="group">
      {options.map((o) => <button key={o.value} className={o.value === value ? "on" : ""} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>{o.label}</button>)}
    </div>
  );
}

/**
 * Render free web services sleep after 15 min idle and take up to ~1 min to wake (D-023). Instead of failing,
 * the app polls /health and explains the wait; the measured wake time is shown on the Platform page.
 */
export function WakeGate({ children }: { children: (health: Record<string, unknown>) => ReactNode }) {
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [attempts, setAttempts] = useState(0);
  useEffect(() => {
    let alive = true;
    const t0 = performance.now();
    const tick = setInterval(() => setElapsed((performance.now() - t0) / 1000), 250);
    const poll = async () => {
      while (alive) {
        setAttempts((a) => a + 1);
        try {
          const h = await api.get<Record<string, unknown>>("/health", undefined, 70000);
          if (alive) { setHealth({ ...h, wake_seconds: (performance.now() - t0) / 1000 }); clearInterval(tick); }
          return;
        } catch {
          await new Promise((r) => setTimeout(r, 2000));
        }
      }
    };
    poll();
    return () => { alive = false; clearInterval(tick); };
  }, []);
  if (health) return <>{children(health)}</>;
  return (
    <div className="wake">
      <div className="box">
        <div className="spinner" aria-hidden />
        <h1>Waking the backend…</h1>
        <p className="ink2">The API runs on a free cloud instance that sleeps after 15 minutes without traffic. The first request starts it again, which usually takes under a minute.</p>
        <p className="muted small num">{elapsed.toFixed(0)} s · attempt {attempts} · {API_BASE}</p>
      </div>
    </div>
  );
}
