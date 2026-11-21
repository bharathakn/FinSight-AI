"""KDD step 4 for mining: numeric/categorical features → boolean *items* (Plan §6.4).

Thresholds come from the Phase 0/2 audit counts (docs/DATA.md §5, reports/eda_summary.json).
Label items (FRAUDULENT / BORDERLINE) are appended separately and may only appear as rule
consequents (enforced in finsight.mining.association).
"""
from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------- India (Custom synthetic)

INDIA_ITEM_RULES: dict[str, str] = {
    # amount (transaction)
    "HIGH_AMOUNT_SPIKE": "amount_ratio_vs_30d_avg > 5",
    "MODERATE_AMOUNT_SPIKE": "2 < amount_ratio_vs_30d_avg <= 5",
    "HIGH_VALUE": "amount_inr >= 50000",
    "NEAR_THRESHOLD": "45000 <= amount_inr < 50000",
    # behaviour
    "NEW_DEVICE": "is_new_device == 1",
    "NEW_BENEFICIARY": "is_new_beneficiary == 1",
    "RECENT_BENEFICIARY": "time_since_beneficiary_added_min < 1440",
    "FAR_FROM_HOME": "distance_from_home_km > 100",
    "EXTREME_LOCATION": "distance_from_home_km > 500",
    "ODD_HOUR": "hour_of_day < 6 or hour_of_day > 22",
    # context
    "CH_UPI": "channel == 'UPI'",
    "CH_NEFT": "channel == 'NEFT'",
    "CH_RTGS": "channel == 'RTGS'",
    "PEP_CUSTOMER": "sender_pep_flag == 1",
    "KYC_NOT_VERIFIED": "sender_kyc_status != 'VERIFIED'",
    # network (point-in-time, Phase 3)
    "PASS_THROUGH": "is_pass_through == 1",
    "QUICK_INFLOW": "sec_since_sender_last_inflow < 3600",
    "HUB_RECEIVER": "receiver_distinct_senders_24h >= 4",
    "BURST_SENDER": "sender_tx_24h >= 2",
    "SHARED_DEVICE": "device_distinct_accounts >= 3",
}
NETWORK_ITEMS = ["PASS_THROUGH", "QUICK_INFLOW", "HUB_RECEIVER", "BURST_SENDER", "SHARED_DEVICE"]
LABEL_ITEMS = ["FRAUDULENT", "BORDERLINE"]
# Dropped after the audit: NEW_ACCOUNT (0 rows), IMMEDIATE_REPEAT (2 rows),
# RAPID_BURST via tx_frequency_24h (no signal → replaced by computed BURST_SENDER).


def india_items(m: pd.DataFrame, with_labels: bool = True) -> pd.DataFrame:
    """One boolean column per item. `m` = master_features table (Phase 3 output)."""
    df = m.assign(channel=m["channel"].astype(str), sender_kyc_status=m["sender_kyc_status"].astype(str))
    items = pd.DataFrame({name: df.eval(expr).astype(bool) for name, expr in INDIA_ITEM_RULES.items()},
                         index=m.index)
    if with_labels:
        label = m["risk_label"].astype(str)
        for lab in LABEL_ITEMS:
            items[lab] = label == lab
    return items


def india_items_extended(m: pd.DataFrame, with_labels: bool = True) -> pd.DataFrame:
    """22 core items + 27 one-hot context items (city, device, hour band, account and customer type).

    Used only by the benchmark to show how the item-universe size drives Apriori's candidate
    explosion (memory) versus FP-Growth's compressed tree.
    """
    hour_band = pd.cut(m["hour_of_day"], [-1, 5, 11, 17, 23], labels=["NIGHT", "MORNING", "AFTERNOON", "EVENING"])
    extra = [
        pd.get_dummies(m["tx_city"].astype(str), prefix="CITY"),
        pd.get_dummies(m["device_type"].astype(str), prefix="DEV"),
        pd.get_dummies(hour_band.astype(str), prefix="HOUR"),
        pd.get_dummies(m["sender_account_type"].astype(str), prefix="ACCT"),
        pd.get_dummies(m["sender_customer_type"].astype(str), prefix="CUST"),
    ]
    return pd.concat([india_items(m, with_labels)] + [e.astype(bool) for e in extra], axis=1)


def to_baskets(items: pd.DataFrame) -> list[list[str]]:
    """Transaction → list of present items (the classic market-basket view, for display)."""
    cols = items.columns.to_numpy()
    return [list(cols[row]) for row in items.to_numpy()]


def item_table(items: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    """Support and fraud rate per item."""
    y = m["is_fraud"].to_numpy()
    rows = []
    for c in items.columns:
        on = items[c].to_numpy()
        rows.append({"item": c, "rule": INDIA_ITEM_RULES.get(c, "risk_label == " + c),
                     "count": int(on.sum()), "support": round(on.mean(), 4),
                     "fraud_rate": round(y[on].mean(), 4) if on.any() else None})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- PaySim (6.36M rows, Polars)

def paysim_items_lazy(lf):
    """PaySim → boolean items as a lazy Polars frame (never materialised in pandas until mined).

    `isFlaggedFraud` is excluded: it's the simulator's own rule (TRANSFER > 200k), not behaviour.
    """
    import polars as pl

    c = pl.col
    exprs = [(c("type") == t).alias(f"TYPE_{t}") for t in ("PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT")]
    exprs += [
        (c("amount") > 200_000).alias("AMT_LARGE"),
        (c("amount") < 1_000).alias("AMT_SMALL"),
        ((c("oldbalanceOrg") > 0) & (c("newbalanceOrig") == 0)).alias("ORIG_EMPTIED"),
        (c("oldbalanceOrg") == 0).alias("ORIG_ZERO_BEFORE"),
        ((c("amount") - c("oldbalanceOrg")).abs() < 1).alias("ORIG_FULL_AMOUNT"),
        (c("nameDest").str.starts_with("M")).alias("DEST_MERCHANT"),
        ((c("oldbalanceDest") == 0) & ~c("nameDest").str.starts_with("M")).alias("DEST_ZERO_BEFORE"),
        (((c("oldbalanceDest") + c("amount") - c("newbalanceDest")).abs() > 1)
         & ~c("nameDest").str.starts_with("M")).alias("DEST_BALANCE_MISMATCH"),
        ((c("step") % 24) < 6).alias("NIGHT"),
        (c("isFraud") == 1).alias("FRAUD"),
    ]
    return lf.select(exprs)


def paysim_items(n_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Materialise PaySim items as a pandas bool frame (optionally a seeded random sample)."""
    from finsight.data.loader import scan_paysim

    lf = paysim_items_lazy(scan_paysim())
    df = lf.collect()
    if n_rows is not None and n_rows < df.height:
        df = df.sample(n=n_rows, seed=seed)
    return df.to_pandas()
