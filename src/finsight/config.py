"""Central configuration. Every path/URI is overridable through env vars (12-factor)."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = Path(os.getenv("FINSIGHT_DATASET_DIR", PROJECT_ROOT / "dataset"))
RAW_DIR = DATASET_DIR / "raw"
INDIA_V1_0_DIR = DATASET_DIR / "processed" / "custom_india"
INDIA_V1_1_DIR = DATASET_DIR / "processed" / "custom_india_v1_1"
CURATED_DIR = Path(os.getenv("FINSIGHT_CURATED_DIR", DATASET_DIR / "curated"))

ARTIFACTS_DIR = Path(os.getenv("FINSIGHT_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
REPORTS_DIR = Path(os.getenv("FINSIGHT_REPORTS_DIR", PROJECT_ROOT / "reports"))

# Object-storage root used by services (local path, s3://bucket/prefix, gs://bucket/prefix).
STORAGE_URI = os.getenv("FINSIGHT_STORAGE_URI", str(ARTIFACTS_DIR))

SEED = int(os.getenv("FINSIGHT_SEED", "42"))
TIMEZONE = "Asia/Kolkata"

# Columns that must never be model features (Plan §18). Enforced by ml.feature_sets.
LEAKAGE_COLUMNS = frozenset(
    {
        "is_fraud",
        "risk_label",
        "scenario_id",
        "alert_id",
        "is_flagged_alert",
        "rapid_forward_delta_sec",
    }
)

# Identifiers: useful for joins/graphs, never as raw features.
ID_COLUMNS = frozenset(
    {
        "transaction_id",
        "sender_account_id",
        "receiver_account_id",
        "sender_customer_id",
        "receiver_customer_id",
        "sender_vpa",
        "receiver_vpa",
        "sender_ifsc",
        "receiver_ifsc",
        "device_id",
        "ip_address",
    }
)
