// Thin client for the FinSight gateway. The base URL comes from VITE_API_BASE_URL at build time
// (Vercel env var → the Render service URL); the SPA calls the gateway directly with CORS (D-023).
import { useCallback, useEffect, useRef, useState } from "react";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") || "http://localhost:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string, public requestId?: string) {
    super(message);
  }
}

type Params = Record<string, string | number | boolean | undefined | null>;

function url(path: string, params?: Params) {
  const u = new URL(API_BASE + path);
  Object.entries(params ?? {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") u.searchParams.set(k, String(v));
  });
  return u.toString();
}

async function request<T>(method: "GET" | "POST", path: string, opts: { params?: Params; body?: unknown; timeoutMs?: number } = {}): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), opts.timeoutMs ?? 45000);
  try {
    const r = await fetch(url(path, opts.params), {
      method,
      signal: ctrl.signal,
      headers: opts.body ? { "Content-Type": "application/json" } : undefined,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    });
    const rid = r.headers.get("x-request-id") ?? undefined;
    if (!r.ok) {
      let detail = r.statusText;
      try {
        const j = await r.json();
        detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      } catch { /* not JSON */ }
      throw new ApiError(r.status, detail, rid);
    }
    return (await r.json()) as T;
  } catch (e) {
    if (e instanceof ApiError) throw e;
    if ((e as Error).name === "AbortError") throw new ApiError(0, "The API did not answer in time.");
    throw new ApiError(0, "Cannot reach the API at " + API_BASE + ".");
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  get: <T,>(path: string, params?: Params, timeoutMs?: number) => request<T>("GET", path, { params, timeoutMs }),
  post: <T,>(path: string, body: unknown) => request<T>("POST", path, { body }),
};

/** Fetch on mount / when `deps` change. While refetching, the previous data stays (rendered dimmed). */
export function useApi<T>(path: string | null, params?: Params, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState<boolean>(!!path);
  const seq = useRef(0);
  const key = JSON.stringify(params ?? {});
  const load = useCallback(() => {
    if (!path) return;
    const n = ++seq.current;
    setLoading(true);
    api.get<T>(path, params)
      .then((d) => { if (n === seq.current) { setData(d); setError(null); } })
      .catch((e: ApiError) => { if (n === seq.current) setError(e); })
      .finally(() => { if (n === seq.current) setLoading(false); });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, key, ...deps]);
  useEffect(load, [load]);
  return { data, error, loading, reload: load };
}

// ---------------------------------------------------------------- formatting helpers
export const fmt = {
  int: (v: number | null | undefined) => (v == null ? "—" : Math.round(v).toLocaleString("en-IN")),
  num: (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d)),
  pct: (v: number | null | undefined, d = 1) => (v == null ? "—" : (100 * v).toFixed(d) + "%"),
  inr: (v: number | null | undefined) => (v == null ? "—" : "₹" + Math.round(v).toLocaleString("en-IN")),
  ms: (v: number | null | undefined) => (v == null ? "—" : v < 10 ? v.toFixed(1) + " ms" : Math.round(v) + " ms"),
  date: (s: string | null | undefined) => (s ? new Date(s).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" }) : "—"),
  short: (s: string | null | undefined) => (s ? new Date(s).toLocaleString("en-IN", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hour12: false }) : "—"),
  scenarioOrder: (a: string, b: string) => parseInt(a.slice(1)) - parseInt(b.slice(1)),
  feature: (f: string) => FEATURE_NAMES[f] ?? f.replace(/_/g, " "),
};

// Human-readable feature names for reasons / importance charts
export const FEATURE_NAMES: Record<string, string> = {
  log_amount: "Amount (log)", amount_inr: "Amount (₹)", amount_ratio_vs_30d_avg: "Amount vs 30-day average",
  time_since_beneficiary_added_min: "Minutes since payee added", distance_from_home_km: "Distance from home (km)",
  sec_since_sender_last_inflow: "Seconds since money last received", is_pass_through: "Pass-through (forwarded quickly)",
  forward_ratio: "Forwarded share of last inflow", sender_tx_10m: "Sender transactions in 10 min", sender_tx_1h: "Sender transactions in 1 h",
  sender_tx_24h: "Sender transactions in 24 h", receiver_distinct_senders_24h: "Receiver's distinct senders (24 h)",
  receiver_distinct_senders_7d: "Receiver's distinct senders (7 d)", receiver_tx_in_24h: "Receiver's incoming tx (24 h)",
  is_new_device: "New device", is_new_beneficiary: "New beneficiary", device_is_registered: "Registered device",
  device_age_days: "Device age (days)", time_since_last_tx_sec: "Seconds since sender's last tx", channel_UPI: "Channel = UPI",
  channel_NEFT: "Channel = NEFT", channel_RTGS: "Channel = RTGS", receiver_pagerank: "Receiver PageRank", sender_pagerank: "Sender PageRank",
  same_community: "Same Louvain community", receiver_community_size: "Receiver community size", sender_community_size: "Sender community size",
  account_balance_before: "Balance before", sender_declared_annual_income_inr: "Declared annual income", hour_of_day: "Hour of day",
  device_associated_accounts: "Accounts on device", device_distinct_accounts: "Distinct accounts on device", chain_depth: "Pass-through chain depth",
  sender_bene_count: "Sender's payees", sender_bene_added_7d: "Payees added in 7 days", tx_frequency_24h: "Declared 24 h frequency",
};

export const LABELS = ["LEGITIMATE", "FRAUDULENT", "BORDERLINE"] as const;
export type Label = (typeof LABELS)[number];
export const TX_EXAMPLES = {
  fraud: "TXN_IND_202608_001768",       // S8 mule pass-through
  borderline: "TXN_IND_202608_006264",  // S5 new device (BORDERLINE false alarm)
  legit: "TXN_IND_202608_000001",
};
