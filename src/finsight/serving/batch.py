"""Live analysis of a transaction batch (Phase 9b): upload a CSV, the demo dataset, or a simulated attack.

Pipeline (KDD on the uploaded data, reusing the training-time code so features are computed identically):
  1. validate     required columns, types, timestamps (naive = IST), positive amounts, duplicate ids, size limit
  2. defaults     optional columns missing from the file are filled from the bank's reference data or a neutral
                  value, and reported (D-037)
  3. context      bank history (the stored 10,000 transactions, minus any ids that appear in the batch) + batch,
                  so windows and PageRank see the same context the model was trained with; the demo dataset
                  therefore reproduces the stored features exactly
  4. features     finsight.data.preprocessor.build_master (7-source merge) + finsight.graph.network_features
  5. impute       accounts/devices unknown to the bank → training medians / modes (counted, reported)
  6. score        ml (P(fraud), anomaly, cluster) · graph risk · rule matching · sanctions (batched) → RiskScorer
  7. summarise    data quality, KPIs, tiers, suspicious transactions + account groups, rules, anomalies,
                  evaluation against labels if the file has them (labels are never model inputs)
Results live in memory for 30 minutes (stateless containers: fine for one instance; a shared store such as Redis
would be needed behind a load balancer — reported as a CCT trade-off).
"""
from __future__ import annotations

import io
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from finsight import config
from finsight.data.loader import IndiaDataset, load_india_table
from finsight.serving.store import CATEGORICAL, FEATURES

MAX_ROWS = 25_000
MAX_BYTES = 10 * 1024 * 1024
CHUNK = 1000
EXPLAIN_TOP = 50
TTL_SECONDS = 30 * 60
MAX_STORED = 2                 # results kept per instance (memory: Render free = 512 MB)
RSS_SOFT_LIMIT_MB = 420        # above this, older analyses are evicted before storing a new one
TZ = config.TIMEZONE

REQUIRED = {
    "transaction_id": "unique id of the payment",
    "timestamp": "ISO date-time; without a time zone it is read as IST (Asia/Kolkata)",
    "amount_inr": "amount in rupees, > 0",
    "channel": "UPI, NEFT or RTGS",
    "sender_account_id": "paying account (e.g. ACC_IND_000513)",
    "receiver_account_id": "receiving account",
}
OPTIONAL = {      # column → (default used when the column is missing, description)
    "sender_customer_id": ("from the bank's account records", "customer owning the sender account"),
    "receiver_customer_id": ("from the bank's account records", "customer owning the receiver account"),
    "device_id": ("the sender customer's first registered device", "device used for the payment"),
    "ip_address": ("one address per sender (never shared)", "IP address of the session"),
    "tx_city": ("sender's home city", "city of the transaction"),
    "tx_state": ("sender's home state", "state of the transaction"),
    "tx_lat": ("sender's home latitude", "latitude of the transaction"),
    "tx_long": ("sender's home longitude", "longitude of the transaction"),
    "distance_from_home_km": ("0 (paid from home)", "distance from the sender's home"),
    "is_new_device": ("0", "1 if the device was first seen for this customer"),
    "is_new_beneficiary": ("0", "1 if the payee was added for this payment"),
    "time_since_beneficiary_added_min": ("median of the bank history", "minutes since the payee was added"),
    "amount_ratio_vs_30d_avg": ("amount ÷ sender's 30-day average (account records)", "amount relative to the usual"),
    "tx_frequency_24h": ("median of the bank history", "sender's declared 24 h frequency"),
    "time_since_last_tx_sec": ("computed from history + batch", "seconds since the sender's previous payment"),
    "account_balance_before": ("sender's current balance (account records)", "balance before the payment"),
}
LABELS = {"is_fraud": "0/1 ground truth (optional, evaluation only)",
          "risk_label": "LEGITIMATE / FRAUDULENT / BORDERLINE (optional, evaluation only)",
          "scenario": "scenario name (optional, per-scenario recall)"}
CHANNEL_TYPE = {"UPI": "P2P", "NEFT": "C2B", "RTGS": "B2B"}


class AnalysisError(ValueError):
    """Fatal input problem (→ HTTP 422 with a readable message)."""


# ---------------------------------------------------------------- reference data (the bank's records)
class Reference:
    def __init__(self, root: Path):
        d = Path(root) / "data" / "reference"
        t = {n: load_india_table(n, d) for n in ("transactions", "targets", "accounts", "customers", "devices", "beneficiaries")}
        self.history, self.targets = t["transactions"], t["targets"]
        self.accounts, self.customers, self.devices, self.beneficiaries = t["accounts"], t["customers"], t["devices"], t["beneficiaries"]
        self.raw_columns = list(self.history.columns)
        self.acc = self.accounts.set_index("account_id")
        self.cust = self.customers.set_index("customer_id")
        self.first_device = self.devices.sort_values("first_seen_timestamp").groupby("customer_id", observed=True)["device_id"].first()
        self.medians = {c: float(self.history[c].median()) for c in ("time_since_beneficiary_added_min", "tx_frequency_24h")}
        store = pd.read_parquet(Path(root) / "data" / "transactions.parquet")
        self.feature_fill = {c: (store[c].astype(str).mode().iloc[0] if c in CATEGORICAL else float(store[c].median()))
                             for c in FEATURES}
        self.names = self.cust["customer_name_synthetic"]
        self.history_labels = self.targets.assign(scenario=self.targets["scenario_id"].astype(str).map(_short_scenario))
        import json

        self.test_ids = set(json.loads((Path(root) / "data" / "test_ids.json").read_text()))

    def demo(self, n: int | None = None, seed: int = 7) -> pd.DataFrame:
        """The stored dataset in upload format (with labels), optionally a stratified sample.

        Seed 7, not 42: the train/test split used 42 on same-sized arrays, which would correlate a sample with it."""
        df = self.history.merge(self.history_labels[["transaction_id", "is_fraud", "risk_label", "scenario"]], on="transaction_id")
        if n and n < len(df):     # stratified by label (explicit loop: pandas 3 apply drops the grouping column)
            parts = [g.sample(max(1, round(n * len(g) / len(df))), random_state=seed)
                     for _, g in df.groupby("risk_label", observed=True)]
            df = pd.concat(parts)
        return df.sort_values("timestamp").reset_index(drop=True)


def _short_scenario(s: str) -> str:
    # SCENARIO_8_MULE_PASS_THROUGH → "S8 mule pass through" (same names as the rest of the project)
    parts = s.split("_")
    return f"S{parts[1]} " + " ".join(p.lower() for p in parts[2:]) if len(parts) > 2 else s


# ---------------------------------------------------------------- 1–2 validation and defaults
def parse_timestamps(s: pd.Series) -> pd.Series:
    ts = pd.to_datetime(s.astype(str), errors="coerce", format="mixed")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(TZ, nonexistent="shift_forward", ambiguous="NaT")
    return ts.dt.tz_convert(TZ)


def read_csv(data: bytes) -> pd.DataFrame:
    if len(data) > MAX_BYTES:
        raise AnalysisError(f"File is {len(data) / 1e6:.1f} MB; the limit is {MAX_BYTES / 1e6:.0f} MB.")
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, na_values=[""])
    except Exception as e:                                       # noqa: BLE001 — any parse failure is the user's file
        raise AnalysisError(f"Could not read the file as CSV ({type(e).__name__}).") from None
    df.columns = [c.strip() for c in df.columns]
    if "scenario_id" in df.columns and "scenario" not in df.columns:
        df["scenario"] = df["scenario_id"].map(lambda s: _short_scenario(str(s)) if str(s).startswith("SCENARIO_") else s)
    return df


def validate(df: pd.DataFrame, ref: Reference) -> tuple[pd.DataFrame, dict]:
    if len(df) == 0:
        raise AnalysisError("The file has no rows.")
    if len(df) > MAX_ROWS:
        raise AnalysisError(f"The file has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise AnalysisError(f"Missing required column(s): {', '.join(missing)}. Download the template for the expected format.")
    n0 = len(df)
    issues = []
    df = df.copy()
    df["timestamp"] = parse_timestamps(df["timestamp"])
    df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors="coerce")
    df["channel"] = df["channel"].astype(str).str.strip().str.upper()
    for c in ("transaction_id", "sender_account_id", "receiver_account_id"):
        df[c] = df[c].astype(str).str.strip()
    checks = [("timestamp could not be parsed", df["timestamp"].isna()),
              ("amount missing, non-numeric or ≤ 0", ~(df["amount_inr"] > 0)),
              ("channel not UPI / NEFT / RTGS", ~df["channel"].isin(list(CHANNEL_TYPE))),
              ("empty transaction or account id", (df[["transaction_id", "sender_account_id", "receiver_account_id"]] == "").any(axis=1)
               | df[["transaction_id", "sender_account_id", "receiver_account_id"]].isna().any(axis=1))]
    bad = pd.Series(False, index=df.index)
    for msg, mask in checks:
        if mask.any():
            issues.append({"problem": msg, "rows": int(mask.sum())})
        bad |= mask
    df = df[~bad]
    dup = df["transaction_id"].duplicated()
    if dup.any():
        issues.append({"problem": "duplicate transaction_id (first kept)", "rows": int(dup.sum())})
        df = df[~dup]
    if len(df) == 0:
        raise AnalysisError("No valid rows remain after validation: " + "; ".join(f"{i['problem']} ({i['rows']})" for i in issues))
    known = set(ref.acc.index)
    unknown_accounts = sorted((set(df["sender_account_id"]) | set(df["receiver_account_id"])) - known)
    labels = [c for c in LABELS if c in df.columns]
    if "is_fraud" in df.columns:
        df["is_fraud"] = pd.to_numeric(df["is_fraud"], errors="coerce")
    if "risk_label" in df.columns and "is_fraud" not in df.columns:
        df["is_fraud"] = (df["risk_label"].astype(str).str.upper() == "FRAUDULENT").astype(int)
        labels.append("is_fraud (from risk_label)")
    dq = {"rows_received": n0, "rows_valid": int(len(df)), "rows_dropped": int(n0 - len(df)), "issues": issues,
          "unknown_accounts": len(unknown_accounts), "unknown_account_examples": unknown_accounts[:5],
          "labels_present": labels, "period": [df["timestamp"].min().isoformat(), df["timestamp"].max().isoformat()],
          "columns_received": list(df.columns)}
    return df.reset_index(drop=True), dq


def apply_defaults(df: pd.DataFrame, ref: Reference) -> tuple[pd.DataFrame, dict]:
    """Fill optional raw columns that the file does not provide; report which (and how many rows) were defaulted."""
    df = df.copy()
    acc, cust = ref.acc, ref.cust
    filled = {}

    def fill(col, values):
        if col not in df.columns:
            df[col] = values
            filled[col] = OPTIONAL.get(col, ("derived", ""))[0]
        else:
            na = df[col].isna()
            if na.any():
                df.loc[na, col] = pd.Series(values, index=df.index)[na]
                filled[col] = f"{int(na.sum())} empty cells → " + OPTIONAL.get(col, ("derived", ""))[0]

    snd, rcv = df["sender_account_id"], df["receiver_account_id"]
    fill("sender_customer_id", snd.map(acc["customer_id"]).fillna("CUST_UNKNOWN_" + snd))
    fill("receiver_customer_id", rcv.map(acc["customer_id"]).fillna("CUST_UNKNOWN_" + rcv))
    fill("device_id", df["sender_customer_id"].map(ref.first_device).fillna("DEV_UNKNOWN_" + snd))
    fill("ip_address", "ip-" + snd)
    home = df["sender_customer_id"].map(cust["home_city"]).astype(object)
    fill("tx_city", home.fillna("Unknown"))
    fill("tx_state", df["sender_customer_id"].map(cust["home_state"]).astype(object).fillna("Unknown"))
    fill("tx_lat", df["sender_customer_id"].map(cust["home_lat"]))
    fill("tx_long", df["sender_customer_id"].map(cust["home_long"]))
    fill("distance_from_home_km", 0.0)
    fill("is_new_device", 0)
    fill("is_new_beneficiary", 0)
    fill("time_since_beneficiary_added_min", ref.medians["time_since_beneficiary_added_min"])
    avg = snd.map(acc["historical_30d_avg_tx_amt_inr"])
    fill("amount_ratio_vs_30d_avg", (df["amount_inr"] / avg).fillna(1.0))
    fill("tx_frequency_24h", ref.medians["tx_frequency_24h"])
    fill("account_balance_before", snd.map(acc["current_balance_inr"]).fillna(df["amount_inr"] * 2))
    if "time_since_last_tx_sec" not in df.columns:
        df["time_since_last_tx_sec"] = np.nan                    # computed after history + batch are combined
        filled["time_since_last_tx_sec"] = OPTIONAL["time_since_last_tx_sec"][0]
    numeric = ["amount_inr", "distance_from_home_km", "is_new_device", "is_new_beneficiary", "time_since_beneficiary_added_min",
               "amount_ratio_vs_30d_avg", "tx_frequency_24h", "time_since_last_tx_sec", "account_balance_before", "tx_lat", "tx_long"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["account_balance_after"] = df["account_balance_before"] - df["amount_inr"]
    df["transaction_type"] = df["channel"].map(CHANNEL_TYPE)
    df["status"] = "SUCCESS"
    for c in ("sender_vpa", "receiver_vpa", "sender_ifsc", "receiver_ifsc"):
        if c not in df.columns:
            df[c] = ""
    ts = df["timestamp"]
    df["day_of_month"], df["hour_of_day"], df["day_of_week"] = ts.dt.day, ts.dt.hour, ts.dt.day_of_week
    return df, filled


# ---------------------------------------------------------------- 3–5 features in context
def batch_features(batch: pd.DataFrame, ref: Reference) -> tuple[pd.DataFrame, dict]:
    from finsight.data.preprocessor import build_master
    from finsight.graph.network_features import network_features

    t0 = time.perf_counter()
    hist = ref.history[~ref.history["transaction_id"].isin(batch["transaction_id"])]
    cols = ref.raw_columns
    combined = pd.concat([hist[cols], batch[cols]], ignore_index=True)
    for c in ("channel", "transaction_type", "status", "tx_city", "tx_state"):
        combined[c] = combined[c].astype(str).astype("category")
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True).dt.tz_convert(TZ)
    # seconds since the sender's previous payment, where the file did not provide it
    order = combined.sort_values("timestamp")
    prev = order.groupby("sender_account_id")["timestamp"].diff().dt.total_seconds().reindex(combined.index)
    combined["time_since_last_tx_sec"] = combined["time_since_last_tx_sec"].fillna(prev).fillna(30 * 86_400.0)
    targets = pd.DataFrame({"transaction_id": combined["transaction_id"], "alert_id": np.nan, "is_fraud": 0,
                            "risk_label": "UNKNOWN", "is_flagged_alert": 0, "scenario_id": "UNKNOWN"})
    edges = pd.DataFrame({"src": combined["sender_account_id"], "dst": combined["receiver_account_id"],
                          "amount_inr": combined["amount_inr"], "timestamp": combined["timestamp"]})
    ds = IndiaDataset(transactions=combined, targets=targets, accounts=ref.accounts, customers=ref.customers,
                      devices=ref.devices, beneficiaries=ref.beneficiaries, edges=edges)
    master, _ = build_master(ds)
    feats, _ = network_features(combined, edges, seed=config.SEED)
    m = master.merge(feats, on="transaction_id", how="left", validate="one_to_one")
    m = m[m["transaction_id"].isin(batch["transaction_id"])]
    m = m.set_index("transaction_id").loc[batch["transaction_id"]].reset_index()
    t_feat = time.perf_counter() - t0
    # accounts / devices the bank does not know → training medians / modes (never silently NaN)
    imputed = {}
    for c in FEATURES:
        na = m[c].isna() if c not in CATEGORICAL else (m[c].isna() | (m[c].astype(str).isin(["nan", "None"])))
        if na.any():
            imputed[c] = int(na.sum())
            m.loc[na, c] = ref.feature_fill[c]
    for c in CATEGORICAL:
        m[c] = m[c].astype(str)
    m["sender_name"] = m["sender_customer_id"].map(ref.names)
    m["receiver_name"] = m["receiver_customer_id"].map(ref.names)
    return m, {"feature_seconds": round(t_feat, 2), "history_rows_used": int(len(hist)), "imputed_features": imputed}


# ---------------------------------------------------------------- 6 scoring
def score_frame(m: pd.DataFrame, backend, scorer, rid=None) -> tuple[pd.DataFrame, dict]:
    t0 = time.perf_counter()
    keep = FEATURES + ["amount_inr", "transaction_id"]
    rows = [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()}
            for r in m[keep].to_dict(orient="records")]
    ml, gr, mi = [], [], []
    for i in range(0, len(rows), CHUNK):
        part = rows[i:i + CHUNK]
        ml += backend.ml_score(part, 0, rid=rid)
        gr += backend.graph_risk(part, rid=rid)
        mi += backend.mining_match(part, rid=rid)
    t_ml = time.perf_counter() - t0
    names = m["sender_name"].dropna().unique().tolist()
    sanc = backend.sanctions_screen_batch(names, rid=rid) if names else {}
    best = m["sender_name"].map(lambda n: (sanc.get(n) or {}).get("best_score") or 0).to_numpy(dtype=float)
    p = np.array([r["p_fraud"] for r in ml])
    a = np.array([r["anomaly"]["pct"] for r in ml])
    g = np.array([r["graph_risk"] for r in gr])
    struct = np.array([r["structuring"] for r in mi])
    pep = m["sender_pep_flag"].astype(float).to_numpy() == 1
    res = scorer.score(p, a, g, pep=pep, structuring=struct, sanctions=best)
    out = m.copy()
    out["risk_score"] = np.round(res["score"], 2)
    out["tier"] = res["tier"].astype(str)
    out["p_fraud"] = p
    out["anomaly_pct"] = a
    out["graph_risk"] = g
    out["cluster"] = [r["cluster"]["cluster"] for r in ml]
    out["network_flags"] = [",".join(k for k, v in r["flags"].items() if v) for r in gr]
    out["top_rule"] = [(r["fired_rules"][0]["rule"] if r["n_fired_fraud"] and r["fired_rules"] else "") for r in mi]
    out["n_fraud_rules"] = [r["n_fired_fraud"] for r in mi]
    out["structuring"] = struct
    out["sanctions_score"] = best
    out["overrides"] = [";".join(k for k, f in (("structuring", res["override_structuring"][i]), ("pep", res["override_pep"][i]),
                                                  ("sanctions", res["override_sanctions"][i])) if f) for i in range(len(out))]
    return out, {"score_seconds": round(t_ml, 2), "sanctions_hits": int(sum(1 for v in sanc.values() if v.get("hit"))),
                 "sanctions_names_screened": len(names)}


# ---------------------------------------------------------------- 7 summaries
def _groups(s: pd.DataFrame) -> dict:
    """Suspicious account structures among alerted (HIGH/CRITICAL) transactions."""
    f = s[s["tier"].isin(["HIGH", "CRITICAL"])]
    hubs = (f.groupby("receiver_account_id").agg(alerts=("transaction_id", "size"), senders=("sender_account_id", "nunique"),
                                                  amount_inr=("amount_inr", "sum")).query("senders >= 3")
            .sort_values("senders", ascending=False).head(10).reset_index())
    chains = (f[f["is_pass_through"] == 1].groupby("sender_account_id")
              .agg(forwards=("transaction_id", "size"), max_chain_depth=("chain_depth", "max"), amount_inr=("amount_inr", "sum"))
              .sort_values(["max_chain_depth", "forwards"], ascending=False).head(10).reset_index())
    bursts = (f[f["sender_tx_1h"] >= 3].groupby("sender_account_id")
              .agg(alerts=("transaction_id", "size"), max_tx_in_1h=("sender_tx_1h", "max"), receivers=("receiver_account_id", "nunique"))
              .sort_values("alerts", ascending=False).head(10).reset_index())
    devices = (f[f["device_distinct_accounts"] >= 3].groupby("device_id")
               .agg(alerts=("transaction_id", "size"), accounts=("device_distinct_accounts", "max"))
               .sort_values("accounts", ascending=False).head(10).reset_index())
    # graph: only the structures worth looking at (top hubs with their senders, pass-through chains, bursts)
    def links(rows, kind, limit):
        e = (rows.groupby(["sender_account_id", "receiver_account_id"]).agg(n=("transaction_id", "size"), amount_inr=("amount_inr", "sum"),
                                                                            max_risk=("risk_score", "max")).reset_index()
             .sort_values("max_risk", ascending=False).head(limit))
        return e.assign(kind=kind)
    parts = [links(f[f["receiver_account_id"] == h], "fan-in", 25) for h in hubs["receiver_account_id"].head(6)]
    pt = f[f["is_pass_through"] == 1]
    if len(pt):
        chain_acc = set(chains["sender_account_id"].head(8))
        parts.append(links(f[f["sender_account_id"].isin(chain_acc) | f["receiver_account_id"].isin(chain_acc)], "pass-through", 40))
    for snd in bursts["sender_account_id"].head(3):
        parts.append(links(f[f["sender_account_id"] == snd], "burst", 15))
    e = (pd.concat(parts, ignore_index=True).drop_duplicates(["sender_account_id", "receiver_account_id"])
         if parts else pd.DataFrame(columns=["sender_account_id", "receiver_account_id", "n", "amount_inr", "max_risk", "kind"]))
    nodes = sorted(set(e["sender_account_id"]) | set(e["receiver_account_id"]))
    hub_set = set(hubs["receiver_account_id"].head(6))
    return {"hubs": hubs.to_dict(orient="records"), "chains": chains.to_dict(orient="records"),
            "bursts": bursts.to_dict(orient="records"), "shared_devices": devices.to_dict(orient="records"),
            "graph": {"nodes": [{"id": n, "hub": n in hub_set} for n in nodes],
                      "links": e.rename(columns={"sender_account_id": "source", "receiver_account_id": "target"}).to_dict(orient="records")}}


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """PR-AUC (average precision) with sklearn's tie handling, in numpy (no sklearn in the slim gateway image)."""
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    idx = np.r_[np.flatnonzero(np.diff(s)), len(y) - 1]
    tps = np.cumsum(y)[idx]
    fps = idx + 1 - tps
    precision, recall = tps / (tps + fps), tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    """ROC-AUC via the Mann–Whitney rank statistic (average ranks for ties)."""
    ranks = pd.Series(s).rank(method="average").to_numpy()
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _evaluation(s: pd.DataFrame, cutoffs: dict, test_ids: set | None = None, _top: bool = True) -> dict | None:
    if "is_fraud" not in s.columns or s["is_fraud"].isna().all():
        return None
    y = s["is_fraud"].fillna(0).astype(int).to_numpy()
    if y.min() == y.max():
        return {"note": "labels contain a single class; ranking metrics need both", "positives": int(y.sum()), "rows": int(len(y))}
    alert = s["tier"].isin(["HIGH", "CRITICAL"]).to_numpy()
    tp, fp, fn = int((alert & (y == 1)).sum()), int((alert & (y == 0)).sum()), int((~alert & (y == 1)).sum())
    sc = s["risk_score"].to_numpy(dtype=float)
    out = {"positives": int(y.sum()), "rows": int(len(y)), "pr_auc": average_precision(y, sc),
           "roc_auc": roc_auc(y, sc), "alert_rule": f"tier HIGH or CRITICAL (score ≥ {cutoffs['HIGH']:.0f})",
           "tp": tp, "fp": fp, "fn": fn, "tn": int(len(y) - tp - fp - fn),
           "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    out["f1"] = 2 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-12)
    if "risk_label" in s.columns:
        out["tiers_by_label"] = pd.crosstab(s["risk_label"].astype(str), s["tier"]).reindex(
            columns=["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0).reset_index().to_dict(orient="records")
    if "scenario" in s.columns:
        fr = s[y == 1]
        out["recall_by_scenario"] = (fr.assign(alert=fr["tier"].isin(["HIGH", "CRITICAL"]))
                                     .groupby(fr["scenario"].astype(str)).agg(n=("alert", "size"), recall=("alert", "mean"))
                                     .reset_index().rename(columns={"scenario": "scenario"}).to_dict(orient="records"))
    if _top and test_ids:
        held = s[s["transaction_id"].isin(test_ids)]
        if len(held) >= 50:
            out["held_out"] = _evaluation(held, cutoffs, None, _top=False)
            out["held_out_note"] = (f"{len(held):,} of these rows are in the model's held-out test split (never used for "
                                    "training); the other rows were seen in training, so metrics on them are optimistic.")
    return out


def summarise(s: pd.DataFrame, scorer, dq: dict, timings: dict, source: dict, test_ids: set | None = None) -> dict:
    tiers = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    alert = s["tier"].isin(["HIGH", "CRITICAL"])
    tc = s["tier"].value_counts().reindex(tiers, fill_value=0)
    hist, edges = np.histogram(s["risk_score"], bins=20, range=(0, 100))
    by_channel = s.groupby("channel", observed=True).agg(n=("transaction_id", "size"), alerts=("tier", lambda t: int(t.isin(["HIGH", "CRITICAL"]).sum())),
                                                         amount_inr=("amount_inr", "sum")).reset_index()
    hours = s.assign(h=pd.to_datetime(s["timestamp"]).dt.floor("h"))
    timeline = hours.groupby("h").agg(n=("transaction_id", "size"), alerts=("tier", lambda t: int(t.isin(["HIGH", "CRITICAL"]).sum()))).reset_index()
    if len(timeline) > 200:
        timeline = hours.assign(h=pd.to_datetime(s["timestamp"]).dt.floor("D")).groupby("h").agg(
            n=("transaction_id", "size"), alerts=("tier", lambda t: int(t.isin(["HIGH", "CRITICAL"]).sum()))).reset_index()
    flagged = s[alert]
    rules = (flagged.loc[flagged["top_rule"] != "", "top_rule"].value_counts().head(10)
             .rename_axis("rule").reset_index(name="alerts").to_dict(orient="records"))
    flags = pd.Series([f for x in flagged["network_flags"] for f in x.split(",") if f]).value_counts().to_dict()
    anomalies = s[(s["anomaly_pct"] >= 0.99) & ~alert].sort_values("anomaly_pct", ascending=False)
    return {
        "source": source, "data_quality": dq, "timings": timings,
        "kpis": {"transactions": int(len(s)), "alerts": int(alert.sum()), "alert_rate": float(alert.mean()),
                 "critical": int(tc["CRITICAL"]), "value_at_risk_inr": float(flagged["amount_inr"].sum()),
                 "amount_total_inr": float(s["amount_inr"].sum()), "accounts": int(pd.concat([s.sender_account_id, s.receiver_account_id]).nunique())},
        "tiers": {t: int(tc[t]) for t in tiers}, "cutoffs": scorer.cutoffs, "weights": scorer.weights,
        "score_histogram": {"edges": edges.tolist(), "counts": hist.tolist()},
        "by_channel": by_channel.to_dict(orient="records"),
        "timeline": timeline.rename(columns={"h": "time"}).to_dict(orient="records"),
        "groups": _groups(s), "rules": rules, "network_flags": flags,
        "anomalies": {"count": int(len(anomalies)), "note": "anomaly percentile ≥ 99% but below the alert tiers: unusual, not predicted fraud",
                      "top": anomalies.head(10)[ROW_COLUMNS[:12]].to_dict(orient="records")},
        "evaluation": _evaluation(s, scorer.cutoffs, test_ids),
        "top": s.sort_values("risk_score", ascending=False).head(EXPLAIN_TOP)[ROW_COLUMNS].to_dict(orient="records"),
    }


ROW_COLUMNS = ["transaction_id", "timestamp", "channel", "amount_inr", "sender_account_id", "receiver_account_id",
               "sender_name", "receiver_name", "risk_score", "tier", "p_fraud", "anomaly_pct", "graph_risk", "cluster",
               "network_flags", "top_rule", "n_fraud_rules", "overrides", "sanctions_score"]
LABEL_COLUMNS = ["is_fraud", "risk_label", "scenario"]


# ---------------------------------------------------------------- store + orchestration
@dataclass
class Analysis:
    id: str
    created: float
    summary: dict
    rows: pd.DataFrame
    features: pd.DataFrame = field(repr=False)


class AnalysisStore:
    def __init__(self):
        self._d: dict[str, Analysis] = {}
        self.pinned: dict[str, Analysis] = {}      # deterministic demo results: computed once, never evicted
        self._lock = threading.Lock()

    def put(self, a: Analysis) -> None:
        from finsight.serving.common import rss_mb

        with self._lock:
            now = time.time()
            for k in [k for k, v in self._d.items() if now - v.created > TTL_SECONDS]:
                del self._d[k]
            while self._d and (len(self._d) >= MAX_STORED or rss_mb() > RSS_SOFT_LIMIT_MB):
                del self._d[min(self._d, key=lambda k: self._d[k].created)]
                release_memory()
            self._d[a.id] = a


    def pin(self, a: Analysis) -> None:
        with self._lock:
            self._d.pop(a.id, None)
            self.pinned[a.id] = a

    def get(self, aid: str) -> Analysis:
        with self._lock:
            if aid in self.pinned:
                return self.pinned[aid]
            a = self._d.get(aid)
            if a is None or time.time() - a.created > TTL_SECONDS:
                raise KeyError(f"analysis {aid} (results are kept for 30 minutes)")
            return a


def release_memory() -> None:
    """Free Python garbage and hand freed heap pages back to the OS (glibc malloc_trim), so RSS reflects live data."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:          # not glibc (e.g. macOS): nothing to trim
        pass

def run(batch: pd.DataFrame, ref: Reference, backend, scorer, store: AnalysisStore, source: dict, rid=None) -> dict:
    t0 = time.perf_counter()
    clean, dq = validate(batch, ref)
    raw, defaults = apply_defaults(clean, ref)
    dq["defaulted_columns"] = defaults
    m, fmeta = batch_features(raw, ref)
    dq["imputed_features"] = fmeta.pop("imputed_features")
    scored, smeta = score_frame(m, backend, scorer, rid=rid)
    for c in LABEL_COLUMNS:
        if c in clean.columns:
            scored[c] = clean.set_index("transaction_id").loc[scored["transaction_id"], c].to_numpy()
    timings = {**fmeta, **smeta, "total_seconds": round(time.perf_counter() - t0, 2)}
    summary = summarise(scored, scorer, dq, timings, source, ref.test_ids)
    compact = m[["transaction_id", "amount_inr"] + FEATURES].copy()      # only what a row explanation needs
    for c in CATEGORICAL:
        compact[c] = compact[c].astype("category")
    a = Analysis(id=uuid.uuid4().hex[:12], created=time.time(), summary=summary,
                 rows=scored[ROW_COLUMNS + [c for c in LABEL_COLUMNS if c in scored.columns]].copy(), features=compact)
    del m, scored, raw, clean
    summary["analysis_id"] = a.id
    summary["expires_in_s"] = TTL_SECONDS
    store.put(a)
    release_memory()
    return summary


def rows_page(a: Analysis, tier: str | None = None, q: str | None = None, sort: str = "risk_score", desc: bool = True,
              limit: int = 50, offset: int = 0) -> dict:
    r = a.rows
    if tier:
        r = r[r["tier"].isin([t.strip().upper() for t in tier.split(",")])]
    if q:
        qq = q.strip().upper()
        r = r[r["transaction_id"].str.upper().str.contains(qq, regex=False) | r["sender_account_id"].str.contains(qq, regex=False)
              | r["receiver_account_id"].str.contains(qq, regex=False)]
    if sort in r.columns:
        r = r.sort_values(sort, ascending=not desc)
    return {"total": int(len(r)), "offset": offset, "limit": limit, "items": r.iloc[offset:offset + limit].to_dict(orient="records")}


def feature_row(a: Analysis, tx_id: str) -> dict:
    m = a.features
    hit = m[m["transaction_id"] == tx_id]
    if hit.empty:
        raise KeyError(tx_id)
    return {k: (None if isinstance(v, float) and math.isnan(v) else (str(v) if k in CATEGORICAL else v))
            for k, v in hit.iloc[0].to_dict().items()}


def to_csv(a: Analysis) -> bytes:
    """Scored rows as CSV; text cells that a spreadsheet could execute (= + - @) are neutralised."""
    r = a.rows.copy()
    for c in r.columns:
        if not pd.api.types.is_numeric_dtype(r[c]) and not pd.api.types.is_bool_dtype(r[c]):   # pandas 3: "str" dtype, not object
            r[c] = r[c].map(lambda v: "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v)
    return r.to_csv(index=False).encode()


def template_csv(ref: Reference, n: int = 3) -> bytes:
    cols = list(REQUIRED) + list(OPTIONAL)
    ex = ref.history[ref.history["transaction_id"].isin(ref.history["transaction_id"].iloc[:n])].copy()
    ex["timestamp"] = ex["timestamp"].map(lambda t: t.isoformat())
    return ex[cols].to_csv(index=False).encode()


def sample_csv(ref: Reference, n: int = 1000) -> bytes:
    d = ref.demo(n)
    d["timestamp"] = d["timestamp"].map(lambda t: t.isoformat())
    return d[list(REQUIRED) + list(OPTIONAL) + ["is_fraud", "risk_label", "scenario"]].to_csv(index=False).encode()


def schema() -> dict:
    return {"limits": {"max_rows": MAX_ROWS, "max_megabytes": MAX_BYTES / 1e6},
            "required": [{"column": c, "description": d} for c, d in REQUIRED.items()],
            "optional": [{"column": c, "description": d, "default_if_missing": dflt} for c, (dflt, d) in OPTIONAL.items()],
            "labels": [{"column": c, "description": d} for c, d in LABELS.items()],
            "notes": ["Accounts, customers and devices are looked up in the bank's records (1,500 accounts); unknown ones get typical values and are reported.",
                      "Network features (bursts, pass-through, fan-in, PageRank) are computed over the bank's history plus your file.",
                      "Label columns are only used to evaluate the results; they never influence a score."]}


__all__ = ["Reference", "AnalysisStore", "AnalysisError", "run", "rows_page", "feature_row", "to_csv", "template_csv",
           "sample_csv", "schema", "read_csv"]
