"""KDD steps 2–3 (Cleaning + Integration): 7 files → one analytical table.

The audit (Plan §0) drives the decisions here:
  * `transaction_type` is a 1:1 recoding of `channel`  → dropped (redundant)
  * `status` is constant ("SUCCESS")                    → dropped (zero variance)
  * beneficiaries match only 8/10,000 (sender, receiver) pairs → aggregated per
    sender account, point-in-time, instead of a pair join
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from finsight.data.loader import IndiaDataset

DROP_REDUNDANT = ["transaction_type", "status"]
TARGET_COLS = ["is_fraud", "risk_label", "scenario_id", "alert_id", "is_flagged_alert"]


def clean_transactions(tx: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Return a cleaned copy plus a log of every cleaning decision (for the DQ report)."""
    log: list[dict] = []
    out = tx.copy()

    n0 = len(out)
    out = out.drop_duplicates(subset="transaction_id")
    log.append({"step": "dedupe_transaction_id", "rows_removed": n0 - len(out)})

    n_bad_ts = int(out["timestamp"].isna().sum())
    out = out[out["timestamp"].notna()]
    log.append({"step": "drop_unparseable_timestamp", "rows_removed": n_bad_ts})

    for col in DROP_REDUNDANT:
        if col in out.columns:
            reason = "constant" if out[col].nunique() <= 1 else "redundant_with_channel"
            out = out.drop(columns=col)
            log.append({"step": f"drop_{col}", "reason": reason})

    # Re-derive calendar parts from the timestamp so they can never disagree with it.
    ts = out["timestamp"]
    out["day_of_month"] = ts.dt.day.astype("int16")
    out["hour_of_day"] = ts.dt.hour.astype("int16")
    out["day_of_week"] = ts.dt.day_of_week.astype("int16")
    out["log_amount"] = np.log1p(out["amount_inr"])
    log.append({"step": "derive_calendar_and_log_amount"})
    return out.reset_index(drop=True), log


def beneficiary_aggregates(tx: pd.DataFrame, bene: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time beneficiary stats of the *sender* account at transaction time."""
    pairs = tx[["transaction_id", "sender_account_id", "timestamp"]].merge(
        bene[["sender_account_id", "beneficiary_added_timestamp", "prior_tx_count"]],
        on="sender_account_id",
        how="left",
    )
    known = pairs["beneficiary_added_timestamp"] <= pairs["timestamp"]
    recent = known & (pairs["timestamp"] - pairs["beneficiary_added_timestamp"] <= pd.Timedelta(days=7))
    pairs = pairs.assign(_known=known, _recent=recent,
                         _prior=pairs["prior_tx_count"].where(known))
    agg = pairs.groupby("transaction_id", sort=False).agg(
        sender_bene_count=("_known", "sum"),
        sender_bene_added_7d=("_recent", "sum"),
        sender_bene_min_prior_tx=("_prior", "min"),
    )
    agg["sender_bene_min_prior_tx"] = agg["sender_bene_min_prior_tx"].fillna(-1)
    return agg.reset_index()


def build_master(ds: IndiaDataset) -> tuple[pd.DataFrame, list[dict]]:
    """Merge all sources into `master_transactions` (one row per transaction)."""
    tx, log = clean_transactions(ds.transactions)

    acc = ds.accounts.drop(columns=["account_number_synthetic", "ifsc_code_synthetic", "customer_id"])
    sender_acc = acc.add_prefix("sender_").rename(columns={"sender_account_id": "sender_account_id"})
    sender_acc = sender_acc.drop(columns=["sender_account_open_date"])
    recv_acc = acc[["account_id", "account_type", "account_age_days"]].add_prefix("receiver_")

    cust = ds.customers.drop(columns=["customer_name_synthetic", "home_lat", "home_long", "customer_creation_date"])
    sender_cust = cust.add_prefix("sender_")

    dev = ds.devices[["device_id", "device_type", "is_registered", "associated_accounts_count", "first_seen_timestamp"]]
    dev = dev.rename(columns={"associated_accounts_count": "device_associated_accounts",
                              "is_registered": "device_is_registered"})

    m = (
        tx.merge(sender_acc, on="sender_account_id", how="left", validate="many_to_one")
        .merge(recv_acc, on="receiver_account_id", how="left", validate="many_to_one")
        .merge(sender_cust, on="sender_customer_id", how="left", validate="many_to_one")
        .merge(dev, on="device_id", how="left", validate="many_to_one")
        .merge(beneficiary_aggregates(tx, ds.beneficiaries), on="transaction_id", how="left", validate="one_to_one")
        .merge(ds.targets, on="transaction_id", how="left", validate="one_to_one")
    )
    m["device_age_days"] = (m["timestamp"] - m["first_seen_timestamp"]).dt.total_seconds() / 86400
    m = m.drop(columns=["first_seen_timestamp"])
    m["is_fraud"] = m["is_fraud"].astype("int8")
    log.append({"step": "merge_7_sources", "rows": len(m), "columns": m.shape[1]})
    return m.sort_values("timestamp").reset_index(drop=True), log
