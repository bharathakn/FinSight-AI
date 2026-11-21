"""Service cores for mining, graph and sanctions: plain Python objects that load bundle artifacts and
answer requests. The FastAPI apps (finsight.serving.apps) are thin wrappers, and the gateway can call
these objects directly (inprocess mode) or over HTTP (remote mode) with identical results.
Only pandas/numpy/rapidfuzz are needed here; the ML core lives in ml_core.py (sklearn, xgboost, onnxruntime).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from finsight.mining.discretiser import INDIA_ITEM_RULES, NETWORK_ITEMS
from finsight.scoring import risk_scorer as RS
from finsight.serving.fastpath import items_for_record


# ---------------------------------------------------------------- mining-service (DMW)
class MiningCore:
    """Association rules from Phase 4 applied to live transactions (rule matching = transparent detector)."""

    def __init__(self, root: Path):
        root = Path(root)
        self.rules = pd.read_parquet(root / "mining" / "rules_v1_1.parquet")
        self.rules["antecedents"] = self.rules["antecedents"].map(list)
        self.benchmark_df = pd.read_csv(root / "reports" / "mining_benchmark.csv")
        self.summary = json.loads((root / "reports" / "mining_summary.json").read_text())
        self.item_names = list(INDIA_ITEM_RULES)
        ranked = self.rules.sort_values(["confidence", "lift"], ascending=False)
        keep = ["rule", "consequent", "support", "confidence", "lift", "has_network_item"]
        self._ranked = [(frozenset(a), rec) for a, rec in zip(ranked["antecedents"], ranked[keep].to_dict(orient="records"))]

    def match(self, rows: list[dict], top: int = 5) -> list[dict]:
        """Rules whose antecedent items all fire, best first (pre-sorted plain records: ~0.1 ms per row)."""
        out = []
        for row in rows:
            fired_items = items_for_record(row)
            hit = [rec for ante, rec in self._ranked if ante <= fired_items]
            fraud = [h for h in hit if h["consequent"] == "FRAUDULENT"]
            out.append({"items": sorted(fired_items), "fired_rules": hit[:top], "n_fired": len(hit),
                        "n_fired_fraud": len(fraud), "max_fraud_confidence": max((h["confidence"] for h in fraud), default=0.0),
                        "structuring": bool({"NEAR_THRESHOLD", "BURST_SENDER"} <= fired_items)})
        return out

    def patterns(self, consequent: str | None = None, min_confidence: float = 0.0, min_lift: float = 0.0,
                 network_only: bool = False, limit: int = 50) -> dict:
        r = self.rules
        if consequent:
            r = r[r["consequent"] == consequent.upper()]
        r = r[(r["confidence"] >= min_confidence) & (r["lift"] >= min_lift)]
        if network_only:
            r = r[r["has_network_item"]]
        r = r.sort_values(["lift", "confidence"], ascending=False)
        cols = ["rule", "antecedents", "consequent", "n_antecedent", "support", "confidence", "lift", "leverage",
                "conviction", "has_network_item"]
        return {"total": int(len(r)), "items": r.head(limit)[cols].to_dict(orient="records"),
                "item_definitions": INDIA_ITEM_RULES, "network_items": NETWORK_ITEMS}

    def benchmark(self) -> dict:
        return {"runs": self.benchmark_df.to_dict(orient="records"),
                "hardest_point": self.summary.get("benchmark_hardest_point"),
                "apriori_equals_fpgrowth": self.summary.get("apriori_equals_fpgrowth"),
                "paysim": self.summary.get("paysim")}

    def health(self) -> dict:
        return {"rules": int(len(self.rules)), "items": len(self.item_names)}


# ---------------------------------------------------------------- graph-service (ML + DMW)
class GraphCore:
    """Account-level network metrics, ego networks and the transparent graph_risk component (D-030)."""

    def __init__(self, root: Path):
        root = Path(root)
        self.metrics = pd.read_parquet(root / "graph" / "account_metrics_v1_1.parquet").set_index("account_id")
        tx = pd.read_parquet(root / "data" / "transactions.parquet",
                             columns=["transaction_id", "timestamp", "sender_account_id", "receiver_account_id",
                                      "amount_inr", "channel", "risk_label"])
        self.edges = tx
        self.scorer = RS.RiskScorer.load(root / "models" / "final" / "risk_scorer.json")

    def risk(self, rows: list[dict]) -> list[dict]:
        """Same definition as RiskScorer.graph_risk / structuring_fired (D-030), evaluated per row dict."""
        out = []
        for row in rows:
            items = items_for_record(row)
            fired = [n for n in NETWORK_ITEMS if n in items]
            rates = {n: self.scorer.flag_rates.get(n, 0.0) for n in fired}
            out.append({"graph_risk": float(max(rates.values(), default=0.0)),
                        "flags": {n: n in items for n in NETWORK_ITEMS}, "flag_rates": rates,
                        "structuring": set(RS.STRUCTURING_ITEMS) <= items})
        return out

    def network(self, account_id: str, limit: int = 40) -> dict:
        if account_id not in self.metrics.index:
            raise KeyError(account_id)
        m = self.metrics.loc[account_id].to_dict()
        e = self.edges
        out = e[e["sender_account_id"] == account_id]
        inc = e[e["receiver_account_id"] == account_id]

        def agg(d, other):
            g = d.groupby(other).agg(n=("transaction_id", "size"), amount_inr=("amount_inr", "sum"),
                                     last=("timestamp", "max"), fraud=("risk_label", lambda s: int((s == "FRAUDULENT").sum())))
            return g.sort_values("n", ascending=False).head(limit).reset_index().rename(columns={other: "account_id"})

        peers_out, peers_in = agg(out, "receiver_account_id"), agg(inc, "sender_account_id")
        nodes = {account_id: {"id": account_id, "role": "focus", **{k: m[k] for k in ("pagerank", "community_id")}}}
        for _, r in pd.concat([peers_out, peers_in]).iterrows():
            pm = self.metrics.loc[r.account_id] if r.account_id in self.metrics.index else None
            nodes.setdefault(r.account_id, {"id": r.account_id, "role": "peer",
                                            "pagerank": None if pm is None else float(pm["pagerank"]),
                                            "community_id": None if pm is None else int(pm["community_id"])})
        links = ([{"source": account_id, "target": r.account_id, **r.drop("account_id").to_dict()} for _, r in peers_out.iterrows()]
                 + [{"source": r.account_id, "target": account_id, **r.drop("account_id").to_dict()} for _, r in peers_in.iterrows()])
        return {"account_id": account_id, "metrics": m,
                "activity": {"sent": int(len(out)), "received": int(len(inc)),
                             "amount_out_inr": float(out["amount_inr"].sum()), "amount_in_inr": float(inc["amount_inr"].sum()),
                             "distinct_receivers": int(out["receiver_account_id"].nunique()),
                             "distinct_senders": int(inc["sender_account_id"].nunique()),
                             "fraud_involved": int(((out["risk_label"] == "FRAUDULENT").sum() + (inc["risk_label"] == "FRAUDULENT").sum()))},
                "community_members": int((self.metrics["community_id"] == m["community_id"]).sum()),
                "ego": {"nodes": list(nodes.values()), "links": links}}

    def accounts(self, sort: str = "pagerank", limit: int = 25, query: str | None = None) -> dict:
        """Accounts ranked by a network metric (for picking an account in the Network Explorer)."""
        m = self.metrics.reset_index()
        if query:
            m = m[m["account_id"].str.contains(query.upper(), regex=False)]
        if sort not in m.columns:
            raise ValueError(f"sort must be one of {sorted(c for c in m.columns if c != 'account_id')}")
        e = self.edges
        fraud_in = e[e["risk_label"] == "FRAUDULENT"].groupby("receiver_account_id").size()
        top = m.sort_values(sort, ascending=False).head(limit).copy()
        top["fraud_received"] = top["account_id"].map(fraud_in).fillna(0).astype(int)
        return {"sort": sort, "total": int(len(m)), "items": top.to_dict(orient="records")}

    def health(self) -> dict:
        return {"accounts": int(len(self.metrics)), "edges": int(len(self.edges))}


# ---------------------------------------------------------------- sanctions-service (DMW)
class SanctionsCore:
    """Fuzzy OFAC + UN screening (RapidFuzz token_sort_ratio); ≥ 90 forces CRITICAL in the risk score (D-007)."""

    def __init__(self, root: Path):
        from finsight.sanctions.screener import SanctionsScreener

        ent = pd.read_parquet(Path(root) / "sanctions" / "entities.parquet")
        self.screener = SanctionsScreener(ent)
        self.n_entities = int(ent["entity_uid"].nunique())
        self.n_names = int(len(ent))
        self.sources = ent.groupby("source")["entity_uid"].nunique().to_dict()
        from functools import lru_cache

        # screening is a pure function of (name, threshold, limit); names repeat across transactions, and a
        # fuzzy scan of 27K names costs ~10 ms of CPU, so cache the most recent 4,096 queries (D-035)
        self.screen = lru_cache(maxsize=4096)(self._screen)

    def cache_info(self) -> dict:
        c = self.screen.cache_info()
        return {"hits": c.hits, "misses": c.misses, "size": c.currsize}

    def screen_batch(self, names: list[str], threshold: float = 90.0) -> dict[str, dict]:
        """Best match per distinct name in one vectorised RapidFuzz pass (cdist, all cores) — used by batch analysis."""
        import numpy as np
        from rapidfuzz import fuzz, process

        from finsight.sanctions.text import normalise_name

        uniq = sorted({n for n in names if isinstance(n, str) and n.strip()})
        if not uniq:
            return {}
        q = [normalise_name(n) for n in uniq]
        S = process.cdist(q, self.screener._choices, scorer=fuzz.token_sort_ratio, score_cutoff=threshold,
                          dtype=np.uint8, workers=-1)
        best = S.argmax(axis=1)
        out = {}
        for i, n in enumerate(uniq):
            sc = float(S[i, best[i]])
            if sc >= threshold and len(q[i]) >= 3:
                r = self.screener.entities.iloc[int(best[i])]
                out[n] = {"hit": True, "best_score": sc, "matched_name": r["name"], "source": r["source"], "program": r["program"]}
            else:
                out[n] = {"hit": False, "best_score": None}
        return out

    def _screen(self, name: str, threshold: float = 90.0, limit: int = 5) -> dict:
        from dataclasses import asdict

        from finsight.sanctions.text import normalise_name

        matches = [asdict(m) for m in self.screener.screen(name, threshold=threshold, limit=limit)]
        return {"name": name, "normalised": normalise_name(name), "threshold": threshold, "hit": bool(matches),
                "best_score": matches[0]["score"] if matches else None, "matches": matches}

    def health(self) -> dict:
        return {"entities": self.n_entities, "names": self.n_names, "sources": self.sources, "cache": self.cache_info()}
