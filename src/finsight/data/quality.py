"""Data Quality Report (Plan §6.2): completeness, duplicates, timestamps, RI,
redundancy and a single-feature leakage scan."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from finsight.data.loader import IndiaDataset

# (child table, child column, parent table, parent column)
INDIA_FOREIGN_KEYS = [
    ("transactions", "sender_account_id", "accounts", "account_id"),
    ("transactions", "receiver_account_id", "accounts", "account_id"),
    ("transactions", "sender_customer_id", "customers", "customer_id"),
    ("transactions", "receiver_customer_id", "customers", "customer_id"),
    ("transactions", "device_id", "devices", "device_id"),
    ("targets", "transaction_id", "transactions", "transaction_id"),
    ("accounts", "customer_id", "customers", "customer_id"),
    ("devices", "customer_id", "customers", "customer_id"),
    ("beneficiaries", "sender_account_id", "accounts", "account_id"),
    ("beneficiaries", "receiver_account_id", "accounts", "account_id"),
    ("edges", "src", "accounts", "account_id"),
    ("edges", "dst", "accounts", "account_id"),
]


def table_profile(df: pd.DataFrame, name: str) -> dict:
    n_cells = df.size or 1
    nulls = df.isna().sum()
    ts_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.DatetimeTZDtype)
               or pd.api.types.is_datetime64_any_dtype(df[c])]
    return {
        "table": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "completeness": round(1 - nulls.sum() / n_cells, 6),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rate": round(df.duplicated().mean(), 6) if len(df) else 0.0,
        "null_columns": {c: int(v) for c, v in nulls[nulls > 0].items()},
        "row_feature_coverage_mean": round(float(df.notna().mean(axis=1).mean()), 6),
        "valid_timestamps": {c: round(float(df[c].notna().mean()), 6) for c in ts_cols},
        "constant_columns": [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
    }


def referential_integrity(ds: IndiaDataset) -> list[dict]:
    tables = dict(ds.items())
    out = []
    for child, ccol, parent, pcol in INDIA_FOREIGN_KEYS:
        values = tables[child][ccol]
        ok = values.isin(tables[parent][pcol])
        out.append({"fk": f"{child}.{ccol} → {parent}.{pcol}",
                    "rows": int(len(values)), "matched_rate": round(float(ok.mean()), 6),
                    "orphans": int((~ok).sum())})
    # pair-level match that motivated the beneficiary aggregation decision
    pairs = set(zip(ds.beneficiaries.sender_account_id, ds.beneficiaries.receiver_account_id))
    hit = sum((s, r) in pairs for s, r in zip(ds.transactions.sender_account_id, ds.transactions.receiver_account_id))
    out.append({"fk": "transactions.(sender,receiver) → beneficiaries.(sender,receiver)",
                "rows": int(len(ds.transactions)), "matched_rate": round(hit / len(ds.transactions), 6),
                "orphans": int(len(ds.transactions) - hit)})
    return out


def redundant_pairs(df: pd.DataFrame, max_card: int = 50) -> list[dict]:
    """Categorical column pairs that are 1:1 recodings of each other."""
    cats = [c for c in df.columns if df[c].dtype.name in ("category", "object", "str", "string")
            and 1 < df[c].nunique() <= max_card]
    out = []
    for a, b in itertools.combinations(cats, 2):
        joint = df[[a, b]].drop_duplicates()
        if len(joint) == df[a].nunique() == df[b].nunique():
            out.append({"a": a, "b": b, "mapping": {str(k): str(v) for k, v in joint.values}})
    return out


def leakage_scan(df: pd.DataFrame, target: str = "is_fraud", threshold: float = 0.95) -> pd.DataFrame:
    """Single-feature ROC-AUC per numeric column. AUC≈1 ⇒ label proxy (or a very strong signal)."""
    y = df[target]
    rows = []
    for col in df.select_dtypes(include=[np.number, "bool"]).columns:
        if col == target:
            continue
        x = df[col].astype(float)
        if x.nunique(dropna=True) < 2:
            continue
        x = x.fillna(x.median())
        auc = roc_auc_score(y, x)
        rows.append({"feature": col, "auc": round(max(auc, 1 - auc), 4)})
    res = pd.DataFrame(rows).sort_values("auc", ascending=False, ignore_index=True)
    res["flag"] = np.where(res["auc"] >= threshold, "SUSPECT", "")
    return res


def label_proxy_check(edges: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """The rapid_forward_delta_sec finding: is a column non-zero only for fraud?"""
    j = edges.assign(is_fraud=targets["is_fraud"].values)
    nz = j["rapid_forward_delta_sec"] > 0
    return {
        "column": "graph_edges.rapid_forward_delta_sec",
        "nonzero_rows": int(nz.sum()),
        "fraud_rate_when_nonzero": round(float(j.loc[nz, "is_fraud"].mean()) if nz.any() else 0.0, 4),
        "fraud_rate_when_zero": round(float(j.loc[~nz, "is_fraud"].mean()), 4),
        "scenarios_when_nonzero": sorted(map(str, j.loc[nz, "scenario_id"].unique())),
    }


def india_quality_report(ds: IndiaDataset, master: pd.DataFrame | None = None) -> dict:
    report = {
        "tables": [table_profile(df, name) for name, df in ds.items()],
        "referential_integrity": referential_integrity(ds),
        "redundant_pairs": redundant_pairs(ds.transactions),
        "label_balance": {
            "risk_label": ds.targets["risk_label"].value_counts(normalize=True).round(4).astype(float).to_dict(),
            "is_fraud": ds.targets["is_fraud"].value_counts(normalize=True).round(4).astype(float).to_dict(),
        },
        "label_crosstab": pd.crosstab(ds.targets["risk_label"], ds.targets["is_flagged_alert"]).to_dict(),
        "label_proxies": [label_proxy_check(ds.edges, ds.targets)],
    }
    if master is not None:
        report["leakage_scan"] = leakage_scan(master).to_dict(orient="records")
    return report


def missingness_profile(df: pd.DataFrame, top: int = 40) -> pd.DataFrame:
    """Per-column missing rate (used for the IEEE-CIS showcase)."""
    miss = df.isna().mean().sort_values(ascending=False)
    out = miss.rename("missing_rate").reset_index().rename(columns={"index": "column"})
    out["group"] = out["column"].str.extract(r"^([A-Za-z]+)")[0]
    return out.head(top) if top else out
