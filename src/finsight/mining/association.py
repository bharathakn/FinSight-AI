"""KDD steps 5–6 for DMW: frequent itemsets, association rules, pattern evaluation (Plan §6.5).

* `mine`           — Apriori / FP-Growth / FP-Max with runtime + peak-memory instrumentation
* `benchmark`      — algorithm × min_support grid; asserts Apriori and FP-Growth agree exactly
* `closed_itemsets`, `fpmax` — condensed representations (closed ⊇ maximal)
* `target_rules`   — rules whose consequent is a single label item (FRAUDULENT / BORDERLINE)
* `productive_rules` — drop rules that don't beat every simpler sub-rule (rule redundancy pruning)
* `emerging_patterns` — itemsets frequent in fraud but rare in legitimate (growth rate)
* `rule_detector`  — use the rule set as a transparent classifier (coverage/precision baseline)
"""
from __future__ import annotations

import gc
import time
import tracemalloc
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth, fpmax

from finsight.mining.discretiser import LABEL_ITEMS

ALGORITHMS = {"apriori": apriori, "fpgrowth": fpgrowth, "fpmax": fpmax}


@dataclass
class MiningResult:
    algorithm: str
    min_support: float
    itemsets: pd.DataFrame
    seconds: float
    peak_mb: float

    @property
    def n_itemsets(self) -> int:
        return len(self.itemsets)


def mine(items: pd.DataFrame, algorithm: str = "fpgrowth", min_support: float = 0.01,
         max_len: int | None = None, track_memory: bool = True) -> MiningResult:
    """Run one algorithm.

    Measurement protocol (see D-017): `track_memory=True` traces allocations with tracemalloc
    (deterministic, counts numpy + Python), but tracing slows pure-Python FP-Growth ~10×, so
    **timings must come from untraced runs** — `benchmark()` does exactly that.
    """
    fn = ALGORITHMS[algorithm]
    gc.collect()
    if track_memory:
        tracemalloc.start()
    t0 = time.perf_counter()
    res = fn(items, min_support=min_support, use_colnames=True, max_len=max_len)
    secs = time.perf_counter() - t0
    peak = 0.0
    if track_memory:
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
    res = res.assign(length=res["itemsets"].map(len)).sort_values(["length", "support"], ascending=[True, False])
    return MiningResult(algorithm, min_support, res.reset_index(drop=True), secs, peak)


def _as_set(itemsets: pd.DataFrame) -> dict[frozenset, float]:
    return {frozenset(s): round(float(v), 10) for s, v in zip(itemsets["itemsets"], itemsets["support"])}


def benchmark(items: pd.DataFrame, supports=(0.05, 0.02, 0.01, 0.005, 0.002), repeats: int = 3,
              label: str = "", memory_cap_mb: float = 2000) -> pd.DataFrame:
    """Runtime (median of `repeats`) + peak memory per algorithm and min_support.

    Asserts Apriori and FP-Growth return identical itemsets+supports (correctness check).
    Apriori is skipped at lower supports once its peak memory passes `memory_cap_mb`.
    """
    rows, apriori_blown = [], False
    for s in sorted(supports, reverse=True):
        found = {}
        for algo in ("apriori", "fpgrowth", "fpmax"):
            if algo == "apriori" and apriori_blown:
                rows.append({"items": label, "n_items": items.shape[1], "n_rows": len(items), "algorithm": algo,
                             "min_support": s, "skipped": f"peak memory > {memory_cap_mb:.0f} MB at higher support"})
                continue
            runs = [mine(items, algo, s, track_memory=False) for _ in range(repeats)]
            mem = mine(items, algo, s, track_memory=True)
            found[algo] = mem.itemsets
            rows.append({"items": label, "n_items": items.shape[1], "n_rows": len(items), "algorithm": algo,
                         "min_support": s, "itemsets": mem.n_itemsets,
                         "seconds": float(np.median([r.seconds for r in runs])), "peak_mb": round(mem.peak_mb, 1)})
            if algo == "apriori" and mem.peak_mb > memory_cap_mb:
                apriori_blown = True
        if "apriori" in found:
            assert _as_set(found["apriori"]) == _as_set(found["fpgrowth"]), f"Apriori ≠ FP-Growth at {s}"
        closed = closed_itemsets(found["fpgrowth"])
        rows.append({"items": label, "n_items": items.shape[1], "n_rows": len(items), "algorithm": "closed (from fpgrowth)",
                     "min_support": s, "itemsets": len(closed)})
    return pd.DataFrame(rows)


def closed_itemsets(itemsets: pd.DataFrame) -> pd.DataFrame:
    """Itemsets with no proper superset of equal support (lossless condensed representation)."""
    sets = [frozenset(s) for s in itemsets["itemsets"]]
    sup = itemsets["support"].round(10).to_numpy()
    by_support: dict[float, list[int]] = {}
    for i, s in enumerate(sup):
        by_support.setdefault(s, []).append(i)
    keep = np.ones(len(sets), dtype=bool)
    for idx in by_support.values():
        group = sorted(idx, key=lambda i: len(sets[i]))
        for a_pos, a in enumerate(group):
            for b in group[a_pos + 1:]:
                if len(sets[b]) > len(sets[a]) and sets[a] < sets[b]:
                    keep[a] = False
                    break
    return itemsets[keep].reset_index(drop=True)


# ---------------------------------------------------------------- rules

def _fmt(s) -> str:
    return "{" + ", ".join(sorted(s)) + "}"


def target_rules(itemsets: pd.DataFrame, n_transactions: int, targets=LABEL_ITEMS,
                 min_confidence: float = 0.5) -> pd.DataFrame:
    """Rules X → {label}: single label consequent, no label items in the antecedent."""
    rules = association_rules(itemsets, num_itemsets=n_transactions, metric="confidence",
                              min_threshold=min_confidence)
    labels = set(LABEL_ITEMS)
    cons_ok = rules["consequents"].map(lambda c: len(c) == 1 and next(iter(c)) in set(targets))
    ante_ok = rules["antecedents"].map(lambda a: not (set(a) & labels))
    r = rules[cons_ok & ante_ok].copy()
    r["consequent"] = r["consequents"].map(lambda c: next(iter(c)))
    r["rule"] = r["antecedents"].map(_fmt) + " → " + r["consequent"]
    r["n_antecedent"] = r["antecedents"].map(len)
    cols = ["rule", "antecedents", "consequent", "n_antecedent", "antecedent support", "support",
            "confidence", "lift", "leverage", "conviction"]
    return r[cols].sort_values(["consequent", "lift", "support"], ascending=[True, False, False]).reset_index(drop=True)


def filter_rules(rules: pd.DataFrame, min_lift: float = 1.5, min_confidence: float = 0.6) -> pd.DataFrame:
    return rules[(rules["lift"] > min_lift) & (rules["confidence"] > min_confidence)].reset_index(drop=True)


def productive_rules(rules: pd.DataFrame, min_improvement: float = 0.01) -> pd.DataFrame:
    """Keep X → c only if conf(X → c) beats conf(Y → c) by `min_improvement` for every proper
    subset Y ⊂ X that is itself a rule (Bayardo's *improvement*). Removes rules that merely add
    an irrelevant item to a stronger simpler rule."""
    conf = {(frozenset(a), c): v for a, c, v in zip(rules["antecedents"], rules["consequent"], rules["confidence"])}
    keep = []
    for a, c, v in zip(rules["antecedents"], rules["consequent"], rules["confidence"]):
        a = frozenset(a)
        best_sub = max((cv for (sa, sc), cv in conf.items() if sc == c and sa < a), default=-1.0)
        keep.append(v - best_sub >= min_improvement)
    return rules[np.array(keep)].reset_index(drop=True)


def top_rules(rules: pd.DataFrame, consequent: str = "FRAUDULENT", k: int = 10,
              by: tuple = ("lift", "support")) -> pd.DataFrame:
    r = rules[rules["consequent"] == consequent]
    return r.sort_values(list(by), ascending=False).head(k).reset_index(drop=True)


# ---------------------------------------------------------------- contrast / emerging patterns

def emerging_patterns(items: pd.DataFrame, is_target: pd.Series, min_support: float = 0.05,
                      min_growth: float = 3.0, minimal: bool = True) -> pd.DataFrame:
    """Itemsets frequent within the target group (e.g. fraud) and their growth rate vs the rest.

    growth = support_in_target / support_in_rest  (∞ when absent from the rest)
    """
    feat = items.drop(columns=[c for c in LABEL_ITEMS if c in items.columns])
    tgt, rest = feat[is_target.to_numpy()], feat[~is_target.to_numpy()]
    fs = mine(tgt, "fpgrowth", min_support, track_memory=False).itemsets
    rest_np = rest.to_numpy()
    col = {c: i for i, c in enumerate(rest.columns)}
    rest_sup = [rest_np[:, [col[c] for c in s]].all(axis=1).mean() for s in fs["itemsets"]]
    out = fs.rename(columns={"support": "support_target"}).assign(support_rest=rest_sup)
    out["growth"] = np.where(out["support_rest"] > 0, out["support_target"] / out["support_rest"].clip(lower=1e-12), np.inf)
    out["pattern"] = out["itemsets"].map(_fmt)
    out = out[out["growth"] >= min_growth]
    if minimal:
        # keep a pattern only if no proper subset already reaches the same (or higher) growth
        sets = [frozenset(x) for x in out["itemsets"]]
        g = out["growth"].to_numpy()
        keep = [not any(sets[j] < sets[i] and g[j] >= g[i] for j in range(len(sets))) for i in range(len(sets))]
        out = out[np.array(keep, dtype=bool)]
    return out.sort_values(["growth", "support_target"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- rules as a detector

def rule_matches(items: pd.DataFrame, rules: pd.DataFrame) -> np.ndarray:
    """Boolean matrix (transactions × rules): does each rule's antecedent fire?"""
    col = {c: i for i, c in enumerate(items.columns)}
    X = items.to_numpy()
    return np.column_stack([X[:, [col[c] for c in a]].all(axis=1) for a in rules["antecedents"]]) \
        if len(rules) else np.zeros((len(items), 0), dtype=bool)


def rule_detector(items: pd.DataFrame, rules: pd.DataFrame, y: pd.Series,
                  risk_label: pd.Series | None = None) -> dict:
    """Flag a transaction if ANY rule fires. Precision / recall / F1 on is_fraud (a transparent baseline)."""
    fired = rule_matches(items, rules).any(axis=1)
    yv = y.to_numpy().astype(bool)
    tp = int((fired & yv).sum()); fp = int((fired & ~yv).sum()); fn = int((~fired & yv).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {"n_rules": len(rules), "flagged": int(fired.sum()), "tp": tp, "fp": fp, "fn": fn,
           "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    if risk_label is not None:
        out["fp_borderline"] = int((fired & (risk_label.astype(str).to_numpy() == "BORDERLINE")).sum())
    return out
