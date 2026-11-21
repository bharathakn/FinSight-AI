"""Experiment A / B / C feature definitions (Plan §7.3, docs/DATA.md §4).

Every list is checked against the leakage blacklist and the ID list at import time,
so a leaking column cannot silently enter a model.
"""
from __future__ import annotations

import pandas as pd

from finsight import config
from finsight.graph.feature_names import GLOBAL_FEATURES, POINT_IN_TIME_FEATURES

TARGET = "is_fraud"

TRANSACTION_NUMERIC = [
    "log_amount", "hour_of_day", "day_of_week", "account_balance_before",
    "sender_account_age_days", "sender_historical_30d_avg_tx_amt_inr",
    "sender_historical_30d_tx_count", "sender_historical_30d_avg_balance_inr",
    "receiver_account_age_days",
]
TRANSACTION_CATEGORICAL = ["channel", "sender_account_type", "receiver_account_type"]

BEHAVIOURAL_NUMERIC = [
    "is_new_device", "is_new_beneficiary", "time_since_beneficiary_added_min",
    "amount_ratio_vs_30d_avg", "distance_from_home_km", "tx_frequency_24h",
    "time_since_last_tx_sec", "device_is_registered", "device_associated_accounts",
    "device_age_days", "sender_bene_count", "sender_bene_added_7d", "sender_bene_min_prior_tx",
    "sender_pep_flag", "sender_declared_annual_income_inr",
]
BEHAVIOURAL_CATEGORICAL = ["device_type", "sender_kyc_status", "sender_customer_type"]

NETWORK_NUMERIC = POINT_IN_TIME_FEATURES + GLOBAL_FEATURES

EXPERIMENTS: dict[str, dict[str, list[str]]] = {
    "A": {"numeric": TRANSACTION_NUMERIC, "categorical": TRANSACTION_CATEGORICAL},
    "B": {"numeric": TRANSACTION_NUMERIC + BEHAVIOURAL_NUMERIC,
          "categorical": TRANSACTION_CATEGORICAL + BEHAVIOURAL_CATEGORICAL},
    "C": {"numeric": TRANSACTION_NUMERIC + BEHAVIOURAL_NUMERIC + NETWORK_NUMERIC,
          "categorical": TRANSACTION_CATEGORICAL + BEHAVIOURAL_CATEGORICAL},
}

EXPERIMENT_LABELS = {"A": "A · transaction", "B": "B · + behavioural", "C": "C · + network"}


def feature_columns(exp: str) -> list[str]:
    e = EXPERIMENTS[exp]
    return e["numeric"] + e["categorical"]


def assert_no_leakage(columns) -> None:
    bad = (set(columns) & config.LEAKAGE_COLUMNS) | (set(columns) & config.ID_COLUMNS)
    if bad:
        raise ValueError(f"Leakage/ID columns in feature set: {sorted(bad)}")


def get_xy(df: pd.DataFrame, exp: str) -> tuple[pd.DataFrame, pd.Series]:
    cols = feature_columns(exp)
    assert_no_leakage(cols)
    return df[cols].copy(), df[TARGET].astype(int)


for _exp in EXPERIMENTS:
    assert_no_leakage(feature_columns(_exp))
