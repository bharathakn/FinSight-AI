"""Artifact bundle = the "model registry" unit (D-024): everything the services load, nothing else.

    python -m finsight.serving.bundle build      # → artifacts/bundle/current/ + finsight-artifacts.tar.gz
    python -m finsight.serving.bundle verify     # sha256 of every file against manifest.json

Layout (paths are what the services read):
  models/final/        production_model.joblib, risk_scorer.json, metrics.json
  models/anomaly/      prep_expC.joblib, isolation_forest_expC.joblib, ocsvm_expC.joblib, autoencoder_expC.onnx,
                       reference_scores_expC.npz, anomaly_meta.json
  models/clustering/   kmeans_expC.joblib
  mining/              rules_v1_1.parquet
  graph/               account_metrics_v1_1.parquet
  sanctions/           entities.parquet
  data/                transactions.parquet (v1.1 master + synthetic customer names; labels kept for display only)
  reports/             result tables the API serves (model comparison, benchmark, cluster profiles, …)
  manifest.json        version, created, library versions, sha256 + size per file
The bundle is built offline (needs the dataset); the services never touch dataset/.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from finsight import config

BUNDLE_ROOT = config.ARTIFACTS_DIR / "bundle"
CURRENT = BUNDLE_ROOT / "current"
TARBALL = BUNDLE_ROOT / "finsight-artifacts.tar.gz"

COPY = {
    "models/final/production_model.joblib": "models/final/production_model.joblib",
    "models/final/risk_scorer.json": "models/final/risk_scorer.json",
    "models/final/metrics.json": "models/final/metrics.json",
    "models/anomaly/prep_expC.joblib": "models/anomaly/prep_expC.joblib",
    "models/anomaly/isolation_forest_expC.joblib": "models/anomaly/isolation_forest_expC.joblib",
    "models/anomaly/ocsvm_expC.joblib": "models/anomaly/ocsvm_expC.joblib",
    "models/anomaly/autoencoder_expC.onnx": "models/anomaly/autoencoder_expC.onnx",
    "models/anomaly/reference_scores_expC.npz": "models/anomaly/reference_scores_expC.npz",
    "models/anomaly/anomaly_meta.json": "models/anomaly/anomaly_meta.json",
    "models/clustering/kmeans_expC.joblib": "models/clustering/kmeans_expC.joblib",
    "mining/rules_v1_1.parquet": "mining/rules_v1_1.parquet",
    "graph/account_metrics_v1_1.parquet": "graph/account_metrics_v1_1.parquet",
    "sanctions/entities.parquet": "sanctions/entities.parquet",
}
REPORTS = ["ml_experiment_results.csv", "ml_paired_tests.csv", "final_model_comparison.csv", "final_tuning_summary.csv",
           "final_risk_tiers.csv", "final_shap_importance.csv", "mining_benchmark.csv", "mining_summary.json",
           "unsup_clusters.csv", "anomaly_results.csv", "eda_summary.json", "data_quality_india_v1_1.json",
           "sanctions_summary.json", "unsup_kmeans_sweep.csv", "final_per_scenario.csv", "final_learning_curves.csv",
           "mining_items_v1_1.csv"]
OPTIONAL_REPORTS = ["platform_summary.json"]          # measurements; present once Phase 8 has been measured
LIBS = ["scikit-learn", "xgboost-cpu", "imbalanced-learn", "onnxruntime", "pandas", "numpy", "pyarrow"]
STORE_DROP = ["alert_id", "is_flagged_alert", "scenario_id"]        # label proxies are never served
REFERENCE_FILES = ["transactions_features.csv", "transactions_targets.csv", "accounts.csv", "customers.csv",
                   "devices.csv", "beneficiaries.csv"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def transaction_store():
    """v1.1 master features + synthetic customer names (for sanctions screening and display)."""
    import pandas as pd

    from finsight.eda import load_master

    m = load_master("v1_1").drop(columns=STORE_DROP, errors="ignore")
    cust = pd.read_csv(config.INDIA_V1_1_DIR / "customers.csv", usecols=["customer_id", "customer_name_synthetic"])
    names = cust.set_index("customer_id")["customer_name_synthetic"]
    m["sender_name"] = m["sender_customer_id"].map(names)
    m["receiver_name"] = m["receiver_customer_id"].map(names)
    m["risk_label"] = m["risk_label"].astype(str)
    return m


def embedding(store) -> "pd.DataFrame":
    """2-D PCA of the scaled Exp C features (the Phase 6 K-Means space) for the dashboard scatter."""
    import joblib
    import pandas as pd
    from sklearn.decomposition import PCA

    from finsight.ml.feature_sets import feature_columns

    kb = joblib.load(config.ARTIFACTS_DIR / "models" / "clustering" / "kmeans_expC.joblib")
    X = kb["prep"].transform(store[feature_columns("C")])
    pca = PCA(n_components=2, random_state=config.SEED).fit(X)
    P = pca.transform(X)
    return pd.DataFrame({"transaction_id": store["transaction_id"].to_numpy(), "pc1": P[:, 0].astype("float32"),
                         "pc2": P[:, 1].astype("float32"), "cluster": kb["kmeans"].predict(X).astype("int8"),
                         "risk_label": store["risk_label"].to_numpy(), "scenario": store["scenario"].to_numpy()}), \
        [float(v) for v in pca.explained_variance_ratio_]


def build(out: Path = CURRENT) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for src, dst in COPY.items():
        (out / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.ARTIFACTS_DIR / src, out / dst)
    (out / "data").mkdir()
    ref = out / "data" / "reference"                       # bank history + reference tables for live analysis (Phase 9b)
    ref.mkdir()
    for name in REFERENCE_FILES:
        shutil.copy2(config.INDIA_V1_1_DIR / name, ref / name)
    store = transaction_store()
    store.to_parquet(out / "data" / "transactions.parquet", index=False)
    from finsight.ml.splits import stratified_split       # the Phase 5–7 held-out test rows (for honest live evaluation)

    sp = stratified_split(store)
    (out / "data" / "test_ids.json").write_text(json.dumps(store["transaction_id"].iloc[sp.test].tolist()))
    emb, var = embedding(store)
    emb.to_parquet(out / "data" / "embedding.parquet", index=False)
    (out / "reports").mkdir()
    for r in REPORTS + [r for r in OPTIONAL_REPORTS if (config.REPORTS_DIR / r).exists()]:
        shutil.copy2(config.REPORTS_DIR / r, out / "reports" / r)
    (out / "reports" / "embedding_meta.json").write_text(json.dumps({"method": "PCA", "explained_variance": var}))
    files = {str(p.relative_to(out)): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted(out.rglob("*")) if p.is_file()}
    digest = hashlib.sha256("".join(v["sha256"] for v in files.values()).encode()).hexdigest()[:10]
    manifest = {
        "name": "finsight-artifacts",
        "version": f"{datetime.now(timezone.utc):%Y.%m.%d}-{digest}",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "libraries": {lib: _version(lib) for lib in LIBS},
        "total_mb": round(sum(v["bytes"] for v in files.values()) / 1e6, 2),
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with tarfile.open(TARBALL, "w:gz") as tar:
        tar.add(out, arcname=".")
    manifest["tarball_mb"] = round(TARBALL.stat().st_size / 1e6, 2)
    return manifest


def _version(lib: str) -> str | None:
    try:
        return metadata.version(lib)
    except metadata.PackageNotFoundError:
        return None


def verify(root: Path = CURRENT) -> list[str]:
    """Paths whose sha256 differs from the manifest (empty list = intact)."""
    man = json.loads((root / "manifest.json").read_text())
    return [p for p, v in man["files"].items() if not (root / p).exists() or sha256(root / p) != v["sha256"]]


def main(argv=None):
    cmd = (argv or sys.argv[1:] or ["build"])[0]
    if cmd == "build":
        m = build()
        print(json.dumps({k: m[k] for k in ("version", "total_mb", "tarball_mb")} | {"files": len(m["files"]),
                                                                                   "tarball": str(TARBALL)}, indent=2))
    elif cmd == "verify":
        bad = verify()
        print("bundle OK" if not bad else f"MISMATCH: {bad}")
        sys.exit(1 if bad else 0)
    else:
        sys.exit(f"unknown command {cmd!r} (build | verify)")


if __name__ == "__main__":
    main()
