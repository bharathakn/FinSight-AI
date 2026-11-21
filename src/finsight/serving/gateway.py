"""api-gateway orchestration (Plan §5 "Scoring", §8.3): one public request → four domain services → one answer.

/score-transaction
  1. resolve the feature row (store lookup, optional what-if overrides, or a full feature row)
  2. in parallel: ml.score (P(fraud), reasons, anomaly_pct, cluster) · graph.risk (network flags, graph_risk)
                  mining.match (fired rules, structuring) · sanctions.screen (sender + optional counterparty)
  3. RiskScorer (Phase 7 weights, cut-offs, overrides) → 0–100 score + tier
Calls run concurrently in a thread pool, so in remote mode the latency is the slowest service, not the sum.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from finsight.scoring.risk_scorer import RiskScorer
from finsight.serving.common import jsonable
from finsight.serving.store import TransactionStore

CONTEXT = ["timestamp", "channel", "amount_inr", "sender_account_id", "receiver_account_id", "sender_name",
           "receiver_name", "tx_city", "device_type", "sender_pep_flag", "risk_label", "scenario"]


class Gateway:
    def __init__(self, root: Path, backend):
        root = Path(root)
        self.store = TransactionStore(root)
        self.backend = backend
        self.scorer = RiskScorer.load(root / "models" / "final" / "risk_scorer.json")
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.version = self.manifest["version"]
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="fanout")

    def _timed(self, fn, *a, **kw):
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        return r, round((time.perf_counter() - t0) * 1000, 2)

    def score(self, req, rid: str | None = None) -> dict:
        t0 = time.perf_counter()
        row, info = self.store.feature_row(req.transaction_id, req.features, req.overrides)
        names = [n for n in (row.get("sender_name"), req.counterparty_name) if n]
        futs = {
            "ml": self.pool.submit(self._timed, self.backend.ml_score, [row], 6 if req.explain else 0, rid=rid),
            "graph": self.pool.submit(self._timed, self.backend.graph_risk, [row], rid=rid),
            "mining": self.pool.submit(self._timed, self.backend.mining_match, [row], rid=rid),
            "sanctions": [self.pool.submit(self._timed, self.backend.sanctions_screen, n, rid=rid) for n in names],
        }
        (ml,), t_ml = futs["ml"].result()
        (gr,), t_gr = futs["graph"].result()
        (mi,), t_mi = futs["mining"].result()
        sanc = [f.result() for f in futs["sanctions"]]
        t_sa = max([t for _, t in sanc], default=0.0)
        sanc = [s for s, _ in sanc]
        best_sanction = max([s["best_score"] or 0 for s in sanc], default=0)

        pep = bool(row.get("sender_pep_flag") == 1)
        r = self.scorer.score(np.array([ml["p_fraud"]]), np.array([ml["anomaly"]["pct"]]), np.array([gr["graph_risk"]]),
                              pep=[pep], structuring=[mi["structuring"]], sanctions=[best_sanction])
        applied = [k for k, flag in (("structuring +10", r["override_structuring"][0]), ("PEP floor → MEDIUM", r["override_pep"][0]),
                                     ("sanctions ≥ 90 → CRITICAL", r["override_sanctions"][0])) if flag]
        return jsonable({
            "transaction_id": row.get("transaction_id", "ad-hoc"),
            "risk_score": round(float(r["score"][0]), 2), "tier": str(r["tier"][0]), "base_score": round(float(r["base"][0]), 2),
            "components": {"model_probability": ml["p_fraud"], "anomaly_pct": ml["anomaly"]["pct"], "graph_risk": gr["graph_risk"]},
            "weights": self.scorer.weights, "cutoffs": self.scorer.cutoffs, "overrides_applied": applied,
            "decision": {"model_flagged": ml["flagged"], "model_threshold": ml["threshold"],
                         "alert": str(r["tier"][0]) in ("HIGH", "CRITICAL")},
            "reasons": ml.get("reasons", []), "base_log_odds": ml.get("base_log_odds"),
            "network": gr, "rules": mi, "cluster": ml["cluster"], "anomaly": ml["anomaly"], "sanctions": sanc,
            "input": info, "context": {k: row.get(k) for k in CONTEXT},
            "timings_ms": {"ml": t_ml, "graph": t_gr, "mining": t_mi, "sanctions": t_sa,
                           "total": round((time.perf_counter() - t0) * 1000, 2)},
            "mode": self.backend.mode, "model_version": self.version,
        })

    def anomaly(self, req, rid=None) -> dict:
        row, info = self.store.feature_row(req.transaction_id, req.features, req.overrides)
        (res,) = self.backend.ml_anomaly([row], rid=rid)
        return {"transaction_id": row.get("transaction_id"), "detectors": res, "input": info}
