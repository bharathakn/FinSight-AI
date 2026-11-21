"""Transaction store: the served copy of the v1.1 master table (bundle data/transactions.parquet).

Scoring input comes from here, in one of three forms (D-033):
  * transaction_id            → the stored, point-in-time feature row (exactly what the model was evaluated on)
  * transaction_id + overrides → "what-if": change e.g. amount_inr or is_new_device; derived columns are recomputed
  * features                  → a complete feature row supplied by the caller (all Exp C columns)
Computing network features for a brand-new, unseen transaction needs a live feature store (streaming graph
updates); that is out of scope and documented as future work.
Labels (is_fraud, risk_label, scenario) are kept only for display in the dashboard; they are never features
(feature_columns() is checked against the leakage blacklist at import).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from finsight.ml.feature_sets import EXPERIMENTS, feature_columns

FEATURES = feature_columns("C")
CATEGORICAL = set(EXPERIMENTS["C"]["categorical"])
EXTRA_INPUTS = {"amount_inr"}                       # raw input from which log_amount / amount_ratio are derived
PROFILE_COLUMNS = ["transaction_id", "timestamp", "channel", "amount_inr", "sender_account_id", "receiver_account_id",
                   "sender_name", "receiver_name", "tx_city", "tx_state", "device_type", "risk_label", "is_fraud", "scenario"]


class TransactionStore:
    def __init__(self, root: Path):
        df = pd.read_parquet(Path(root) / "data" / "transactions.parquet")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        self.df = df.set_index("transaction_id", drop=False).rename_axis(None)   # unnamed index: column stays sortable

    def __len__(self):
        return len(self.df)

    def get(self, tx_id: str) -> dict:
        if tx_id not in self.df.index:
            raise KeyError(tx_id)
        return self.df.loc[tx_id].to_dict()

    # ---------------------------------------------------------------- scoring input
    def feature_row(self, transaction_id: str | None = None, features: dict | None = None,
                    overrides: dict | None = None) -> tuple[dict, dict]:
        """Return (row, info). `row` has every Exp C feature plus amount_inr and display context."""
        overrides = dict(overrides or {})
        if transaction_id is not None:
            row = self.get(transaction_id)
        else:
            missing = [f for f in FEATURES if f not in (features or {})]
            if missing:
                raise ValueError(f"features is missing {len(missing)} required columns, e.g. {missing[:5]}")
            row = dict(features)
            row.setdefault("transaction_id", "ad-hoc")
            if "amount_inr" not in row and row.get("log_amount") is not None:
                row["amount_inr"] = float(np.expm1(row["log_amount"]))
        unknown = sorted(set(overrides) - set(FEATURES) - EXTRA_INPUTS)
        if unknown:
            raise ValueError(f"cannot override unknown or non-feature columns: {unknown}")
        derived = []
        row.update(overrides)
        if "amount_inr" in overrides:
            amt = float(overrides["amount_inr"])
            if amt <= 0:
                raise ValueError("amount_inr must be positive")
            row["log_amount"] = math.log1p(amt)
            avg = row.get("sender_historical_30d_avg_tx_amt_inr")
            if avg and avg > 0 and "amount_ratio_vs_30d_avg" not in overrides:
                row["amount_ratio_vs_30d_avg"] = amt / avg
            derived += ["log_amount", "amount_ratio_vs_30d_avg"]
        for c in CATEGORICAL:
            if row.get(c) is not None:
                row[c] = str(row[c])
        return row, {"source": "store" if transaction_id else "features", "overrides": sorted(overrides),
                     "derived": derived}

    # ---------------------------------------------------------------- browsing (dashboard)
    def query(self, limit: int = 50, offset: int = 0, label: str | None = None, scenario: str | None = None,
              channel: str | None = None, account_id: str | None = None, min_amount: float | None = None,
              sort: str = "timestamp", desc: bool = True) -> dict:
        d = self.df
        if label:
            d = d[d["risk_label"] == label.upper()]
        if scenario:
            d = d[d["scenario"].str.startswith(scenario)]
        if channel:
            d = d[d["channel"].astype(str) == channel.upper()]
        if account_id:
            d = d[(d["sender_account_id"] == account_id) | (d["receiver_account_id"] == account_id)]
        if min_amount is not None:
            d = d[d["amount_inr"] >= min_amount]
        if sort in d.columns:
            d = d.sort_values(sort, ascending=not desc)
        page = d.iloc[offset: offset + limit][PROFILE_COLUMNS]
        return {"total": int(len(d)), "offset": offset, "limit": limit, "items": page.to_dict(orient="records")}

    def overview(self) -> dict:
        d = self.df
        lab = d["risk_label"].value_counts().to_dict()
        by_channel = d.groupby(d["channel"].astype(str)).agg(n=("is_fraud", "size"), fraud_rate=("is_fraud", "mean"),
                                                             amount_median=("amount_inr", "median"))
        daily = d.groupby(d["timestamp"].dt.strftime("%Y-%m-%d")).agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum"))
        scen = d.groupby("scenario").agg(n=("is_fraud", "size"), fraud=("is_fraud", "sum")).reset_index()
        return {"transactions": int(len(d)), "accounts": int(pd.concat([d.sender_account_id, d.receiver_account_id]).nunique()),
                "customers": int(d["sender_customer_id"].nunique()), "labels": lab,
                "fraud_rate": float(d["is_fraud"].mean()), "amount_total_inr": float(d["amount_inr"].sum()),
                "period": [d["timestamp"].min().isoformat(), d["timestamp"].max().isoformat()],
                "by_channel": by_channel.reset_index().to_dict(orient="records"),
                "daily": daily.reset_index().rename(columns={"timestamp": "day"}).to_dict(orient="records"),
                "scenarios": scen.to_dict(orient="records")}
