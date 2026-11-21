"""End-to-end batch pipeline (runs locally, in compose, or as a Cloud Run Job).

    python -m finsight.pipeline                       # all implemented stages
    python -m finsight.pipeline --stages inject,wrangle
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from finsight import config

log = logging.getLogger("finsight.pipeline")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


# ---------------------------------------------------------------- stages

def stage_inject():
    from finsight.data.inject import inject
    meta = inject()
    log.info("v1.1 written: %s rows modified", meta["injection"]["rows_modified"])


def stage_wrangle():
    """Merge 7 files (v1.0 and v1.1) → master tables + Data Quality Reports."""
    from finsight.data.loader import load_india
    from finsight.data.preprocessor import build_master
    from finsight.data.quality import india_quality_report

    config.CURATED_DIR.mkdir(parents=True, exist_ok=True)
    for tag, d in (("v1_0", config.INDIA_V1_0_DIR), ("v1_1", config.INDIA_V1_1_DIR)):
        ds = load_india(d)
        master, clean_log = build_master(ds)
        master.to_parquet(config.CURATED_DIR / f"master_transactions_{tag}.parquet", index=False)
        report = india_quality_report(ds, master)
        report["cleaning_log"] = clean_log
        write_json(report, config.REPORTS_DIR / f"data_quality_india_{tag}.json")
        log.info("master %s: %s", tag, master.shape)


def stage_graph():
    """Point-in-time + global network features → master_features_{tag}.parquet."""
    from finsight.data.loader import load_india
    from finsight.graph.network_features import network_features

    out = config.ARTIFACTS_DIR / "graph"
    out.mkdir(parents=True, exist_ok=True)
    for tag, d in (("v1_0", config.INDIA_V1_0_DIR), ("v1_1", config.INDIA_V1_1_DIR)):
        ds = load_india(d)
        feats, metrics = network_features(ds.transactions, ds.edges, seed=config.SEED)
        master = pd.read_parquet(config.CURATED_DIR / f"master_transactions_{tag}.parquet")
        master = master.merge(feats, on="transaction_id", how="left", validate="one_to_one")
        master.to_parquet(config.CURATED_DIR / f"master_features_{tag}.parquet", index=False)
        metrics.to_parquet(out / f"account_metrics_{tag}.parquet", index=False)
        log.info("graph %s: %d accounts, %d communities", tag, len(metrics), metrics.community_id.nunique())


def stage_sanctions():
    from finsight.data.loader import load_india_table
    from finsight.sanctions.parser import build_entities
    from finsight.sanctions.screener import SanctionsScreener, threshold_sweep

    ents = build_entities()
    out = config.ARTIFACTS_DIR / "sanctions"
    out.mkdir(parents=True, exist_ok=True)
    ents.to_parquet(out / "entities.parquet", index=False)
    customers = load_india_table("customers")
    sweep = threshold_sweep(SanctionsScreener(ents), customers["customer_name_synthetic"])
    write_json({
        "entities": int(ents.entity_uid.nunique()),
        "name_variants": int(len(ents)),
        "by_source": ents.groupby("source").entity_uid.nunique().to_dict(),
        "by_kind": ents.name_kind.value_counts().to_dict(),
        "customer_screening_threshold_sweep": sweep.to_dict(orient="records"),
    }, config.REPORTS_DIR / "sanctions_summary.json")
    log.info("sanctions: %d entities / %d name variants", ents.entity_uid.nunique(), len(ents))


def stage_eda():
    """Phase 2 figures (reports/figures/eda_*.png) + reports/eda_summary.json."""
    import warnings

    from finsight.eda import run_all

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = run_all()
    log.info("eda: %d figures; RQ1 top-3 = %s", len(s["figures"]), s["rq1_top10"][:3])


def stage_mining():
    """Phase 4: Apriori vs FP-Growth benchmark, rules (v1.0/v1.1/PaySim), emerging patterns, figures."""
    import warnings

    from finsight.mining.report import run_all

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = run_all()
    log.info("mining: v1.1 productive rules=%d (network=%d); PaySim rules=%d; %.0fs",
             s["v1_1"]["productive"], s["v1_1"]["productive_fraud_with_network_item"], s["paysim"]["rules"],
             s["runtime_seconds"])


def stage_train_supervised():
    """Phase 5: 6 models × Exp A/B/C × v1.0/v1.1 (5×2 CV + hold-out), imbalance, temporal, rule baseline."""
    import warnings

    from finsight.ml.supervised import run_all

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = run_all()
    b = s["best_v1_1"]
    log.info("supervised: best v1.1 = %s/Exp %s test PR-AUC %.3f F1 %.3f; %.0fs", b["model"], b["exp"],
             b["test_pr_auc"], b["test_f1"], s["runtime_seconds"])


def stage_unsupervised():
    """Phase 6: K-Means / DBSCAN / PCA / t-SNE, IF · OC-SVM · AE (ONNX) on legit rows, PaySim IF scale track."""
    import warnings

    from finsight.ml.unsupervised import run_all

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = run_all()
    b = s["anomaly"]["best_detector_expC"]
    log.info("unsupervised: K=%d (ARI %.3f); best detector %s PR-AUC %.3f; ONNX parity %s; %.0fs",
             s["clustering"]["C"]["k"], s["clustering"]["C"]["kmeans_vs_label"]["ari"], b["detector"], b["pr_auc"],
             s["onnx_parity"]["pass"], s["runtime_seconds"])


def stage_train_final():
    """Phase 7: tuning (Grid/Randomized, 5-fold CV), stacking + isotonic calibration, risk score, SHAP."""
    import warnings

    from finsight.ml.final_model import run_all

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = run_all()
    log.info("final: production=%s test PR-AUC %.3f F1 %.3f; risk weights %s; reload ok=%s; %.0fs",
             s["production_model"], s["test"]["test_pr_auc"], s["test"]["test_f1"], s["risk_scorer"]["weights"],
             s["reload_check"]["pass"], s["runtime_seconds"])


STAGES = {
    "inject": stage_inject,
    "wrangle": stage_wrangle,
    "graph": stage_graph,
    "sanctions": stage_sanctions,
    "eda": stage_eda,
    "mining": stage_mining,
    "train_supervised": stage_train_supervised,
    "unsupervised": stage_unsupervised,
    "train_final": stage_train_final,
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default=",".join(STAGES), help=f"comma list from: {', '.join(STAGES)}")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    for name in [s.strip() for s in args.stages.split(",") if s.strip()]:
        t0 = time.perf_counter()
        STAGES[name]()
        log.info("stage %-10s done in %.1fs", name, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
