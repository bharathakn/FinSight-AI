"""0–100 risk score with tiers and rule overrides (Plan §5 "Scoring", §6.6).

    risk = 100 · (w_model · P(fraud) + w_anomaly · anomaly_pct + w_graph · graph_risk)

Components
  P(fraud)      calibrated probability from the production model (Phase 7)
  anomaly_pct   percentile of the autoencoder's reconstruction error among legitimate validation rows (Phase 6)
  graph_risk    transparent network score: the training-split fraud rate of the strongest network red flag
                that fires (PASS_THROUGH, QUICK_INFLOW, HUB_RECEIVER, BURST_SENDER, SHARED_DEVICE; Phase 4
                items); 0 if none fires (D-030)
Fitting (validation split only; never test)
  weights       grid over the simplex (step 0.05), maximising PR-AUC of the combined score
  cut-offs      MEDIUM ("review") = highest score that still keeps recall ≥ 0.98;
                HIGH ("alert")    = max-F1 threshold;
                CRITICAL          = lowest score whose precision ≥ 0.95 (above HIGH)
Overrides (applied after the weighted sum)
  sanctions match ≥ 90   → at least CRITICAL
  PEP customer           → at least MEDIUM
  structuring rule fires → +10 points  (NEAR_THRESHOLD amount and BURST_SENDER)

Serving needs only numpy + this module: the fitted scorer is plain JSON (`to_dict` / `from_dict`).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from finsight.mining.discretiser import INDIA_ITEM_RULES, NETWORK_ITEMS

TIERS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
COMPONENTS = ["model", "anomaly", "graph"]
STRUCTURING_ITEMS = ["NEAR_THRESHOLD", "BURST_SENDER"]
FLAG_COLUMNS = ["is_pass_through", "sec_since_sender_last_inflow", "receiver_distinct_senders_24h",
                "sender_tx_24h", "device_distinct_accounts", "amount_inr"]


def network_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean network red flags (+ NEAR_THRESHOLD for the structuring override), same rules as Phase 4."""
    names = NETWORK_ITEMS + ["NEAR_THRESHOLD"]
    return pd.DataFrame({n: df.eval(INDIA_ITEM_RULES[n]).astype(bool).to_numpy() for n in names}, index=df.index)


def flag_rates(flags: pd.DataFrame, y) -> dict[str, float]:
    """Fraud rate among rows where each network flag fires (fit on the training split)."""
    y = np.asarray(y)
    return {n: float(y[flags[n].to_numpy()].mean()) if flags[n].any() else 0.0 for n in NETWORK_ITEMS}


def average_precision(y, s) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def simplex(step: float = 0.05):
    k = round(1 / step)
    for i, j in itertools.product(range(k + 1), repeat=2):
        if i + j <= k:
            yield np.array([i, j, k - i - j]) / k


@dataclass
class RiskScorer:
    weights: dict = field(default_factory=lambda: {"model": 1.0, "anomaly": 0.0, "graph": 0.0})
    cutoffs: dict = field(default_factory=lambda: {"MEDIUM": 30.0, "HIGH": 60.0, "CRITICAL": 85.0})
    flag_rates: dict = field(default_factory=dict)
    structuring_bonus: float = 10.0
    sanctions_critical: float = 90.0
    fit_info: dict = field(default_factory=dict)

    # ---- components
    def graph_risk(self, flags: pd.DataFrame) -> np.ndarray:
        r = np.zeros(len(flags))
        for n in NETWORK_ITEMS:
            r = np.where(flags[n].to_numpy(), np.maximum(r, self.flag_rates.get(n, 0.0)), r)
        return r

    def combine(self, p, a, g) -> np.ndarray:
        w = self.weights
        return 100 * (w["model"] * np.asarray(p) + w["anomaly"] * np.asarray(a) + w["graph"] * np.asarray(g))

    # ---- tiers + overrides
    def tier(self, score) -> np.ndarray:
        s = np.asarray(score)
        out = np.full(s.shape, "LOW", dtype=object)
        for t in ("MEDIUM", "HIGH", "CRITICAL"):
            out[s >= self.cutoffs[t]] = t
        return out

    def score(self, p, a, g, pep=None, structuring=None, sanctions=None) -> dict:
        """Vectorised scoring. Returns base score, final score (after overrides), tier and override flags."""
        base = self.combine(p, a, g)
        final = base.copy()
        n = len(final)
        structuring = np.zeros(n, bool) if structuring is None else np.asarray(structuring, bool)
        pep = np.zeros(n, bool) if pep is None else np.asarray(pep, bool)
        sanc = np.zeros(n) if sanctions is None else np.asarray(sanctions, float)
        final = final + self.structuring_bonus * structuring
        final = np.where(pep, np.maximum(final, self.cutoffs["MEDIUM"]), final)
        final = np.where(sanc >= self.sanctions_critical, np.maximum(final, self.cutoffs["CRITICAL"]), final)
        final = np.clip(final, 0, 100)
        return {"base": base, "score": final, "tier": self.tier(final),
                "override_structuring": structuring, "override_pep": pep & (base < self.cutoffs["MEDIUM"]),
                "override_sanctions": sanc >= self.sanctions_critical}

    # ---- fitting (validation only)
    def fit_weights(self, p, a, g, y, step: float = 0.05) -> pd.DataFrame:
        rows = []
        X = np.column_stack([p, a, g])
        for w in simplex(step):
            rows.append({"w_model": w[0], "w_anomaly": w[1], "w_graph": w[2], "pr_auc": average_precision(y, X @ w)})
        grid = pd.DataFrame(rows)
        # best PR-AUC; ties broken towards the larger model weight (simplest explanation)
        best = grid.sort_values(["pr_auc", "w_model"], ascending=[False, False]).iloc[0]
        self.weights = {"model": float(best.w_model), "anomaly": float(best.w_anomaly), "graph": float(best.w_graph)}
        self.fit_info["weights_val_pr_auc"] = float(best.pr_auc)
        self.fit_info["model_only_val_pr_auc"] = float(grid.query("w_model == 1").pr_auc.iloc[0])
        return grid

    def fit_cutoffs(self, score, y, review_recall: float = 0.98, critical_precision: float = 0.95) -> dict:
        from sklearn.metrics import precision_recall_curve

        y = np.asarray(y).astype(int)
        prec, rec, thr = precision_recall_curve(y, score)
        prec, rec = prec[:-1], rec[:-1]
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        high = float(thr[int(np.argmax(f1))])
        ok = np.where(rec >= review_recall)[0]
        medium = float(thr[ok.max()]) if len(ok) else float(thr[0])
        medium = min(medium, high)
        crit = np.where((prec >= critical_precision) & (thr > high))[0]
        critical = float(thr[crit.min()]) if len(crit) else float(min(100.0, np.percentile(score, 99.5)))
        self.cutoffs = {"MEDIUM": medium, "HIGH": high, "CRITICAL": max(critical, high)}
        self.fit_info["cutoff_rules"] = {"MEDIUM": f"recall >= {review_recall}", "HIGH": "max F1",
                                         "CRITICAL": f"precision >= {critical_precision}"}
        return self.cutoffs

    # ---- persistence
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RiskScorer":
        return cls(**d)

    def save(self, path: Path) -> Path:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "RiskScorer":
        return cls.from_dict(json.loads(Path(path).read_text()))


def structuring_fired(flags: pd.DataFrame) -> np.ndarray:
    return flags[STRUCTURING_ITEMS].all(axis=1).to_numpy()
