"""Phase 1b — seeded graph-structure injection: Custom India v1.0 → v1.1 (Plan §6.3).

Why: the v1.0 generator *labelled* network-fraud scenarios (rings, mules, layering,
bursts, structuring, shared devices) but never wired them into the transaction graph —
every network/velocity feature had AUC ≈ 0.50. This module rewires ONLY the rows of
those scenarios into real structure, plus a layer of legitimate "decoy" structure
(merchant hubs, family devices, benign forwards, salary bursts) so the graph signal is
realistic rather than a free give-away.

Invariants preserved (verified on v1.0 by the audit):
  * account → customer, VPA number, IFSC                     (looked up, never invented)
  * device belongs to the sender's customer                  (except injected shared devices)
  * amount_ratio_vs_30d_avg = amount / sender 30d avg amount (recomputed on change)
  * balance_after = balance_before − amount                  (recomputed on change)
  * calendar columns derive from timestamp                   (recomputed for all rows)
Labels (targets file) are never modified. Every change is logged in injection_log.csv.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from finsight import config
from finsight.data.loader import INDIA_FILES, load_india

S = {
    "legit": "SCENARIO_1_NORMAL_LEGITIMATE",
    "burst": "SCENARIO_3_RAPID_BURST",
    "struct": "SCENARIO_6_STRUCTURING_SPLITTING",
    "layer": "SCENARIO_7_LAYERING_NETWORK",
    "mule": "SCENARIO_8_MULE_PASS_THROUGH",
    "ring": "SCENARIO_9_FRAUD_RING_HUB",
    "shared": "SCENARIO_11_SHARED_DEVICE_IP",
}

# Organised fraud usually looks normal one transaction at a time. v1.0 gave every network
# scenario obvious per-transaction fingerprints, so behavioural features alone caught 100% of it.
# A seeded share of those FRAUDULENT rows gets its behavioural columns copied from real
# legitimate rows ("camouflage"), so only graph structure can expose them (Plan §6.3).
CAMOUFLAGE_FRACTION = 0.5
CAMOUFLAGE_SCENARIOS = ("burst", "struct", "layer", "mule", "ring", "shared")
CAMOUFLAGE_COLUMNS = ["is_new_device", "is_new_beneficiary", "time_since_beneficiary_added_min",
                      "distance_from_home_km"]

RAPID_FORWARD_WINDOW_SEC = 3600
RAPID_FORWARD_MIN_RATIO = 0.8


@dataclass
class _Ctx:
    tx: pd.DataFrame                    # working copy, index = row position
    accounts: pd.DataFrame
    devices: pd.DataFrame
    rng: np.random.Generator
    t_min: pd.Timestamp
    t_max: pd.Timestamp
    log: list[dict] = field(default_factory=list)
    acc_ifsc: dict = field(default_factory=dict)
    cust_devices: dict = field(default_factory=dict)
    acc_cust: dict = field(default_factory=dict)
    acc_avg_amt: dict = field(default_factory=dict)
    used_receivers: set = field(default_factory=set)

    # ---- helpers -------------------------------------------------------------
    def note(self, idx, scenario, group, role, changed):
        self.log.append({"transaction_id": self.tx.at[idx, "transaction_id"], "scenario_id": scenario,
                         "group_id": group, "role": role, "fields_changed": ",".join(changed)})

    def anchor(self, span: pd.Timedelta, preferred: pd.Timestamp | None = None) -> pd.Timestamp:
        """A start time such that [start, start+span] fits inside the dataset window."""
        latest = self.t_max - span
        if preferred is not None and self.t_min <= preferred <= latest:
            return preferred
        offset = self.rng.uniform(0, (latest - self.t_min).total_seconds())
        return (self.t_min + pd.Timedelta(seconds=offset)).floor("s")

    def _vpa(self, acc: str, old_vpa: str) -> str:
        bank = old_vpa.split("@", 1)[1] if "@" in old_vpa else "oksbi"
        return f"user{acc[-6:]}@{bank}"

    def set_receiver(self, idx, acc) -> list[str]:
        t = self.tx
        if t.at[idx, "receiver_account_id"] == acc or t.at[idx, "sender_account_id"] == acc:
            return []                                   # unchanged, or would be a self-transfer
        t.at[idx, "receiver_account_id"] = acc
        t.at[idx, "receiver_customer_id"] = self.acc_cust[acc]
        t.at[idx, "receiver_vpa"] = self._vpa(acc, t.at[idx, "receiver_vpa"])
        t.at[idx, "receiver_ifsc"] = self.acc_ifsc[acc]
        return ["receiver"]

    def set_sender(self, idx, acc) -> list[str]:
        t = self.tx
        if t.at[idx, "sender_account_id"] == acc or t.at[idx, "receiver_account_id"] == acc:
            return []                                   # unchanged, or would be a self-transfer
        cust = self.acc_cust[acc]
        t.at[idx, "sender_account_id"] = acc
        t.at[idx, "sender_customer_id"] = cust
        t.at[idx, "sender_vpa"] = self._vpa(acc, t.at[idx, "sender_vpa"])
        t.at[idx, "sender_ifsc"] = self.acc_ifsc[acc]
        t.at[idx, "device_id"] = self.cust_devices[cust][0]      # keep device↔sender invariant
        self._recompute_ratio(idx)
        return ["sender", "device_id", "amount_ratio_vs_30d_avg"]

    def set_amount(self, idx, amount: float) -> list[str]:
        t = self.tx
        amount = round(float(amount), 2)
        t.at[idx, "amount_inr"] = amount
        t.at[idx, "account_balance_after"] = round(t.at[idx, "account_balance_before"] - amount, 2)
        if t.at[idx, "channel"] == "RTGS" and amount < 200_000:      # RTGS minimum is ₹2 lakh
            t.at[idx, "channel"] = "NEFT"
            t.at[idx, "transaction_type"] = "C2B"
        self._recompute_ratio(idx)
        return ["amount_inr", "account_balance_after", "amount_ratio_vs_30d_avg"]

    def set_time(self, idx, ts: pd.Timestamp) -> list[str]:
        self.tx.at[idx, "timestamp"] = ts
        return ["timestamp"]

    def _recompute_ratio(self, idx):
        avg = self.acc_avg_amt[self.tx.at[idx, "sender_account_id"]]
        self.tx.at[idx, "amount_ratio_vs_30d_avg"] = round(self.tx.at[idx, "amount_inr"] / avg, 4)

    def fresh_accounts(self, n: int, exclude: set) -> list[str]:
        """Accounts not yet used as an injected receiver (avoids accidental mega-hubs)."""
        pool = [a for a in self.accounts["account_id"] if a not in exclude and a not in self.used_receivers]
        pick = list(self.rng.choice(pool, size=n, replace=False))
        self.used_receivers.update(pick)
        return pick

    def rows(self, scenario: str) -> np.ndarray:
        return self.tx.index[self.tx["scenario_id"] == scenario].to_numpy()


def _groups(rng, idx: np.ndarray, size: int) -> list[np.ndarray]:
    idx = rng.permutation(idx)
    return [idx[i:i + size] for i in range(0, len(idx), size) if len(idx[i:i + size]) == size]


# ---------------------------------------------------------------- fraud structures

def _camouflage(c: _Ctx):
    scen = [S[k] for k in CAMOUFLAGE_SCENARIOS]
    rows = c.tx.index[c.tx["scenario_id"].isin(scen) & (c.tx["risk_label"] == "FRAUDULENT")].to_numpy()
    pick = np.sort(c.rng.choice(rows, size=int(len(rows) * CAMOUFLAGE_FRACTION), replace=False))
    legit = c.tx.index[c.tx["risk_label"] == "LEGITIMATE"].to_numpy()
    donors = c.rng.choice(legit, size=len(pick), replace=True)
    c.tx.loc[pick, CAMOUFLAGE_COLUMNS] = c.tx.loc[donors, CAMOUFLAGE_COLUMNS].to_numpy()
    for i, d in zip(pick, donors):
        c.note(i, c.tx.at[i, "scenario_id"], "camouflage", f"donor={c.tx.at[d, 'transaction_id']}",
               CAMOUFLAGE_COLUMNS)


def _ring_hubs(c: _Ctx, n_rings: int = 10):
    rows = c.rows(S["ring"])
    for g, members in enumerate(np.array_split(c.rng.permutation(rows), n_rings)):
        senders = set(c.tx.loc[members, "sender_account_id"])
        hub = c.fresh_accounts(1, exclude=senders)[0]
        start = c.anchor(pd.Timedelta(hours=48))
        for i in members:
            ch = c.set_receiver(i, hub)
            ch += c.set_time(i, start + pd.Timedelta(seconds=int(c.rng.uniform(0, 48 * 3600))))
            c.note(i, S["ring"], f"ring_{g:02d}", "ring_member→hub", ch)


def _mules(c: _Ctx):
    """X→M then M→Y within 2–30 min at 90–98% of the amount (M = tx2's sender)."""
    for g, (a, b) in enumerate(_groups(c.rng, c.rows(S["mule"]), 2)):
        mule = c.tx.at[b, "sender_account_id"]
        start = c.anchor(pd.Timedelta(hours=1), c.tx.at[a, "timestamp"])
        ch_a = c.set_receiver(a, mule) + c.set_time(a, start)
        ch_b = c.set_time(b, start + pd.Timedelta(seconds=int(c.rng.uniform(120, 1800))))
        ch_b += c.set_amount(b, c.tx.at[a, "amount_inr"] * c.rng.uniform(0.90, 0.98))
        c.note(a, S["mule"], f"mule_{g:03d}", "inflow_to_mule", ch_a)
        c.note(b, S["mule"], f"mule_{g:03d}", "mule_outflow", ch_b)


def _layering(c: _Ctx, hops: int = 4):
    """Chains A→B→C→D→E: tx_i.receiver := tx_{i+1}.sender, 5–60 min apart, 2–5% decay."""
    for g, chain in enumerate(_groups(c.rng, c.rows(S["layer"]), hops)):
        t = c.anchor(pd.Timedelta(hours=hops), c.tx.at[chain[0], "timestamp"])
        amount = c.tx.at[chain[0], "amount_inr"]
        for k, i in enumerate(chain):
            ch = c.set_time(i, t)
            if k > 0:
                amount *= 1 - c.rng.uniform(0.02, 0.05)
                ch += c.set_amount(i, amount)
            if k < hops - 1:
                ch += c.set_receiver(i, c.tx.at[chain[k + 1], "sender_account_id"])
            c.note(i, S["layer"], f"layer_{g:03d}", f"hop_{k + 1}", ch)
            t = t + pd.Timedelta(seconds=int(c.rng.uniform(300, 3600)))


def _bursts(c: _Ctx, size: int = 5):
    for g, grp in enumerate(_groups(c.rng, c.rows(S["burst"]), size)):
        sender = c.tx.at[grp[0], "sender_account_id"]
        start = c.anchor(pd.Timedelta(minutes=10), c.tx.at[grp[0], "timestamp"])
        for i in grp:
            ch = c.set_sender(i, sender)
            ch += c.set_time(i, start + pd.Timedelta(seconds=int(c.rng.uniform(0, 600))))
            c.note(i, S["burst"], f"burst_{g:02d}", "burst_tx", ch)


def _structuring(c: _Ctx, size: int = 5):
    """5 transfers of ₹45,000–49,999 from one sender within 24h (just under ₹50k)."""
    for g, grp in enumerate(_groups(c.rng, c.rows(S["struct"]), size)):
        sender = c.tx.at[grp[0], "sender_account_id"]
        start = c.anchor(pd.Timedelta(hours=24), c.tx.at[grp[0], "timestamp"])
        for i in grp:
            ch = c.set_sender(i, sender)
            ch += c.set_amount(i, c.rng.uniform(45_000, 49_999))
            ch += c.set_time(i, start + pd.Timedelta(seconds=int(c.rng.uniform(0, 86_400))))
            c.note(i, S["struct"], f"struct_{g:02d}", "split_tx", ch)


def _shared_devices(c: _Ctx, n_devices: int = 15):
    rows = c.rows(S["shared"])
    for g, members in enumerate(np.array_split(c.rng.permutation(rows), n_devices)):
        dev = c.tx.at[members[0], "device_id"]
        ip = c.tx.at[members[0], "ip_address"]
        for i in members:
            c.tx.at[i, "device_id"] = dev
            c.tx.at[i, "ip_address"] = ip
            c.tx.at[i, "is_new_device"] = int(c.tx.at[i, "sender_account_id"] != c.tx.at[members[0], "sender_account_id"])
            c.note(i, S["shared"], f"shared_dev_{g:02d}", "device_user", ["device_id", "ip_address", "is_new_device"])


# ---------------------------------------------------------------- legitimate decoys

def _decoys(c: _Ctx):
    legit = c.rng.permutation(c.rows(S["legit"]))
    take = iter(np.split(legit, np.cumsum([200, 120, 60, 40]))[:4])

    # 10 merchant hubs × 20 customers (half of them inside a 48h "sale" window)
    for g, members in enumerate(np.array_split(next(take), 10)):
        hub = c.fresh_accounts(1, exclude=set(c.tx.loc[members, "sender_account_id"]))[0]
        clustered = g % 2 == 0
        start = c.anchor(pd.Timedelta(hours=48))
        for i in members:
            ch = c.set_receiver(i, hub)
            if clustered:
                ch += c.set_time(i, start + pd.Timedelta(seconds=int(c.rng.uniform(0, 48 * 3600))))
            c.note(i, S["legit"], f"decoy_merchant_{g:02d}", "customer→merchant", ch)

    # 60 benign forwards: A→B then B→C 10–180 min later at 30–100% of the amount
    for g, (a, b) in enumerate(_groups(c.rng, next(take), 2)):
        start = c.anchor(pd.Timedelta(hours=4), c.tx.at[a, "timestamp"])
        ch_a = c.set_receiver(a, c.tx.at[b, "sender_account_id"]) + c.set_time(a, start)
        ch_b = c.set_time(b, start + pd.Timedelta(seconds=int(c.rng.uniform(600, 10_800))))
        ch_b += c.set_amount(b, c.tx.at[a, "amount_inr"] * c.rng.uniform(0.3, 1.0))
        c.note(a, S["legit"], f"decoy_forward_{g:02d}", "inflow", ch_a)
        c.note(b, S["legit"], f"decoy_forward_{g:02d}", "forward", ch_b)

    # 20 family devices × 3 senders
    for g, members in enumerate(_groups(c.rng, next(take), 3)):
        dev, ip = c.tx.at[members[0], "device_id"], c.tx.at[members[0], "ip_address"]
        for i in members:
            c.tx.at[i, "device_id"], c.tx.at[i, "ip_address"] = dev, ip
            c.note(i, S["legit"], f"decoy_family_dev_{g:02d}", "device_user", ["device_id", "ip_address"])

    # 10 salary-day bursts: one sender, 4 payments within 30 min
    for g, grp in enumerate(_groups(c.rng, next(take), 4)):
        sender = c.tx.at[grp[0], "sender_account_id"]
        start = c.anchor(pd.Timedelta(minutes=30), c.tx.at[grp[0], "timestamp"])
        for i in grp:
            ch = c.set_sender(i, sender)
            ch += c.set_time(i, start + pd.Timedelta(seconds=int(c.rng.uniform(0, 1800))))
            c.note(i, S["legit"], f"decoy_salary_{g:02d}", "burst_tx", ch)


# ---------------------------------------------------------------- consistency pass

def epoch_seconds(ts: pd.Series) -> np.ndarray:
    """tz-aware timestamps → int64 Unix seconds (independent of pandas' datetime unit)."""
    return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.int64)


def _recompute_derived(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    ts = tx["timestamp"]
    tx["day_of_month"], tx["hour_of_day"], tx["day_of_week"] = ts.dt.day, ts.dt.hour, ts.dt.day_of_week

    # The given velocity columns describe the account's whole activity (the 10K rows are a
    # sample of the ecosystem), so they can only be *raised* by what we observe, never lowered.
    tx["_s"] = epoch_seconds(ts)
    obs_24h = np.zeros(len(tx), dtype=int)
    obs_gap = np.full(len(tx), np.iinfo(np.int64).max)
    for _, g in tx.groupby("sender_account_id", sort=False):
        s = g["_s"].to_numpy()
        lo = np.searchsorted(s, s - 86_400, side="left")
        obs_24h[g.index] = np.arange(len(s)) - lo + 1                # incl. the current tx
        gaps = np.diff(s, prepend=np.iinfo(np.int64).min // 2)
        obs_gap[g.index] = np.where(np.arange(len(s)) > 0, gaps, np.iinfo(np.int64).max)
    tx["tx_frequency_24h"] = np.maximum(tx["tx_frequency_24h"].to_numpy(), obs_24h)
    tx["time_since_last_tx_sec"] = np.minimum(tx["time_since_last_tx_sec"].to_numpy(), obs_gap)
    return tx


def rapid_forward_delta(tx: pd.DataFrame) -> np.ndarray:
    """Per edge: seconds until the receiver forwarded ≥80% of it onward (within 1h), else 0.
    Recomputed from the real timeline — replaces the generator's label-proxy column."""
    out = np.zeros(len(tx), dtype=np.int64)
    by_sender = {k: g[["_s", "amount_inr"]].to_numpy() for k, g in tx.groupby("sender_account_id", sort=False)}
    for i, (dst, s, amt) in enumerate(tx[["receiver_account_id", "_s", "amount_inr"]].itertuples(index=False)):
        nxt = by_sender.get(dst)
        if nxt is None:
            continue
        lo = np.searchsorted(nxt[:, 0], s, side="right")
        hi = np.searchsorted(nxt[:, 0], s + RAPID_FORWARD_WINDOW_SEC, side="right")
        hits = nxt[lo:hi]
        hits = hits[hits[:, 1] >= RAPID_FORWARD_MIN_RATIO * amt]
        if len(hits):
            out[i] = int(hits[0, 0] - s)
    return out


# ---------------------------------------------------------------- entry point

def inject(src_dir: Path | None = None, out_dir: Path | None = None, seed: int = config.SEED) -> dict:
    src_dir = Path(src_dir or config.INDIA_V1_0_DIR)
    out_dir = Path(out_dir or config.INDIA_V1_1_DIR)
    ds = load_india(src_dir, typed=False)
    raw_tx = ds.transactions
    tx = raw_tx.merge(ds.targets[["transaction_id", "scenario_id", "risk_label"]], on="transaction_id", validate="one_to_one")
    tx["timestamp"] = pd.to_datetime(tx["timestamp"]).dt.tz_convert(config.TIMEZONE)

    acc = ds.accounts
    ifsc = pd.concat([
        raw_tx[["sender_account_id", "sender_ifsc"]].set_axis(["acc", "ifsc"], axis=1),
        raw_tx[["receiver_account_id", "receiver_ifsc"]].set_axis(["acc", "ifsc"], axis=1),
    ]).drop_duplicates("acc").set_index("acc")["ifsc"].to_dict()
    for a, code in zip(acc["account_id"], acc["ifsc_code_synthetic"]):
        ifsc.setdefault(a, code)

    c = _Ctx(
        tx=tx, accounts=acc, devices=ds.devices, rng=np.random.default_rng(seed),
        t_min=tx["timestamp"].min().floor("D"), t_max=tx["timestamp"].max(),
        acc_ifsc=ifsc,
        cust_devices=ds.devices.groupby("customer_id")["device_id"].apply(list).to_dict(),
        acc_cust=dict(zip(acc["account_id"], acc["customer_id"])),
        acc_avg_amt=dict(zip(acc["account_id"], acc["historical_30d_avg_tx_amt_inr"])),
    )
    for step in (_camouflage, _ring_hubs, _mules, _layering, _bursts, _structuring, _shared_devices, _decoys):
        step(c)

    tx = _recompute_derived(c.tx)
    rf = rapid_forward_delta(tx)

    # ---- write v1.1 with the exact v1.0 schemas
    out_dir.mkdir(parents=True, exist_ok=True)
    iso = tx["timestamp"].map(lambda t: t.isoformat())
    feats = tx.assign(timestamp=iso)[raw_tx.columns]
    feats.to_csv(out_dir / INDIA_FILES["transactions"], index=False)
    ds.targets.set_index("transaction_id").loc[feats["transaction_id"]].reset_index() \
        .to_csv(out_dir / INDIA_FILES["targets"], index=False)

    edges = pd.DataFrame({
        ":START_ID": tx["sender_account_id"], ":END_ID": tx["receiver_account_id"],
        "amount_inr": tx["amount_inr"], "timestamp": iso, "channel": tx["channel"],
        "rapid_forward_delta_sec": rf, "scenario_id": tx["scenario_id"], ":TYPE": "TRANSFERRED_TO",
    })
    edges.to_csv(out_dir / INDIA_FILES["edges"], index=False)

    dev = ds.devices.copy()
    observed = tx.groupby("device_id")["sender_account_id"].nunique()
    dev["associated_accounts_count"] = np.maximum(
        dev["associated_accounts_count"], dev["device_id"].map(observed).fillna(0).astype(int))
    dev.to_csv(out_dir / INDIA_FILES["devices"], index=False)
    for name in ("accounts", "customers", "beneficiaries"):
        getattr(ds, name).to_csv(out_dir / INDIA_FILES[name], index=False)

    log = pd.DataFrame(c.log)
    log.to_csv(out_dir / "injection_log.csv", index=False)

    meta = json.loads((src_dir / "dataset_metadata.json").read_text())
    meta.update({
        "dataset_name": "Custom Synthetic India Dataset (graph-structure injected)",
        "version": "1.1.0",
        "derived_from": "1.0.0",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "injection": {
            "module": "finsight.data.inject",
            "rows_modified": int(log["transaction_id"].nunique()),
            "rows_by_group_type": log.assign(kind=log["group_id"].str.replace(r"_\d+$", "", regex=True))
                                     .groupby("kind")["transaction_id"].nunique().to_dict(),
            "labels_modified": False,
            "camouflage": {"fraction": CAMOUFLAGE_FRACTION, "columns": CAMOUFLAGE_COLUMNS,
                           "rows": int((log["group_id"] == "camouflage").sum())},
            "rapid_forward_delta_sec": f"recomputed: ≥{RAPID_FORWARD_MIN_RATIO:.0%} forwarded within "
                                       f"{RAPID_FORWARD_WINDOW_SEC}s",
        },
    })
    (out_dir / "dataset_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta
