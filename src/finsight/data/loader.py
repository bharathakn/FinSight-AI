"""KDD step 1 (Selection): typed loaders for every source dataset."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

from finsight import config

INDIA_FILES = {
    "transactions": "transactions_features.csv",
    "targets": "transactions_targets.csv",
    "accounts": "accounts.csv",
    "customers": "customers.csv",
    "devices": "devices.csv",
    "beneficiaries": "beneficiaries.csv",
    "edges": "graph_edges.csv",
}

_DATETIME_COLS = {
    "transactions": ["timestamp"],
    "accounts": ["account_open_date"],
    "customers": ["customer_creation_date"],
    "devices": ["first_seen_timestamp"],
    "beneficiaries": ["beneficiary_added_timestamp"],
    "edges": ["timestamp"],
}

_CATEGORY_COLS = {
    "transactions": ["channel", "transaction_type", "status", "tx_city", "tx_state"],
    "targets": ["risk_label", "scenario_id"],
    "accounts": ["account_type"],
    "customers": ["customer_type", "kyc_status", "home_city", "home_state"],
    "devices": ["device_type", "os_version", "app_version"],
    "beneficiaries": ["beneficiary_status"],
    "edges": ["channel", "scenario_id"],
}

# neo4j-style headers in graph_edges.csv → plain names
_EDGE_RENAME = {":START_ID": "src", ":END_ID": "dst", ":TYPE": "rel_type"}


@dataclass
class IndiaDataset:
    transactions: pd.DataFrame
    targets: pd.DataFrame
    accounts: pd.DataFrame
    customers: pd.DataFrame
    devices: pd.DataFrame
    beneficiaries: pd.DataFrame
    edges: pd.DataFrame

    def items(self):
        return ((f.name, getattr(self, f.name)) for f in fields(self))


def _to_datetime(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    return out.dt.tz_convert(config.TIMEZONE)


def load_india_table(name: str, directory: Path | None = None, typed: bool = True) -> pd.DataFrame:
    directory = Path(directory or config.INDIA_V1_0_DIR)
    df = pd.read_csv(directory / INDIA_FILES[name])
    if name == "edges":
        df = df.rename(columns=_EDGE_RENAME)
    if not typed:
        return df
    for col in _DATETIME_COLS.get(name, []):
        df[col] = _to_datetime(df[col])
    for col in _CATEGORY_COLS.get(name, []):
        df[col] = df[col].astype("category")
    return df


def load_india(directory: Path | None = None, typed: bool = True) -> IndiaDataset:
    """Load all 7 Custom India files (v1.0 by default; pass INDIA_V1_1_DIR for v1.1)."""
    return IndiaDataset(**{name: load_india_table(name, directory, typed) for name in INDIA_FILES})


# ---------------------------------------------------------------- auxiliary corpora

def paysim_path() -> Path:
    return next((config.RAW_DIR / "paysim").glob("*.csv"))


def scan_paysim():
    """Lazy Polars scan of the 6.36M-row PaySim CSV (never materialise it in pandas)."""
    import polars as pl

    return pl.scan_csv(paysim_path())


def load_ieee(split: str = "train", nrows: int | None = None) -> pd.DataFrame:
    """IEEE-CIS transaction ⋈ identity (left join on TransactionID)."""
    d = config.RAW_DIR / "ieee_cis"
    tx = pd.read_csv(d / f"{split}_transaction.csv", nrows=nrows)
    idn = pd.read_csv(d / f"{split}_identity.csv")
    # test_identity uses id-01 style names; normalise to train's id_01
    idn.columns = [c.replace("-", "_") for c in idn.columns]
    return tx.merge(idn, on="TransactionID", how="left", validate="one_to_one")
