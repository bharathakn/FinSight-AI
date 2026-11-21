"""Phase 3 — graph features (Plan §7.8).

Point-in-time features use ONLY transactions strictly before the current one, so they are
leakage-free and identical to what the graph-service can compute online.
Global features (PageRank, Louvain community, clustering) are label-free but computed on the
full graph — documented as transductive.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from finsight.data.inject import epoch_seconds

PASS_THROUGH_WINDOW_SEC = 30 * 60
PASS_THROUGH_MIN_RATIO = 0.8
PASS_THROUGH_MAX_RATIO = 1.0

from finsight.graph.feature_names import GLOBAL_FEATURES, NETWORK_FEATURES, POINT_IN_TIME_FEATURES  # noqa: E402,F401


def _window_counts(keys: np.ndarray, secs: np.ndarray, window: int,
                   other: np.ndarray | None = None) -> np.ndarray:
    """For each row: # earlier rows with the same key within `window` seconds
    (or # distinct `other` values among them). Rows must be sorted by time."""
    out = np.zeros(len(keys), dtype=np.int32)
    order = pd.Series(np.arange(len(keys))).groupby(keys, sort=False).indices
    for idx in order.values():
        s = secs[idx]
        lo = np.searchsorted(s, s - window, side="left")
        hi = np.searchsorted(s, s, side="left")            # strictly before (ties excluded)
        if other is None:
            out[idx] = hi - lo
        else:
            o = other[idx]
            out[idx] = [len(set(o[a:b])) for a, b in zip(lo, hi)]
    return out


def _distinct_to_date(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """# distinct `values` seen with this key so far, including the current row."""
    seen: dict = {}
    out = np.empty(len(keys), dtype=np.int32)
    for i, (k, v) in enumerate(zip(keys, values)):
        s = seen.setdefault(k, set())
        s.add(v)
        out[i] = len(s)
    return out


def point_in_time_features(tx: pd.DataFrame) -> pd.DataFrame:
    """`tx` needs transaction_id, timestamp, sender/receiver_account_id, amount_inr, device_id, ip_address."""
    t = tx.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)
    secs = epoch_seconds(t["timestamp"])
    snd = t["sender_account_id"].to_numpy()
    rcv = t["receiver_account_id"].to_numpy()

    f = pd.DataFrame({"transaction_id": t["transaction_id"]})
    f["sender_tx_10m"] = _window_counts(snd, secs, 600)
    f["sender_tx_1h"] = _window_counts(snd, secs, 3600)
    f["sender_tx_24h"] = _window_counts(snd, secs, 86_400)
    f["sender_distinct_receivers_24h"] = _window_counts(snd, secs, 86_400, rcv)
    f["receiver_tx_in_24h"] = _window_counts(rcv, secs, 86_400)
    f["receiver_distinct_senders_24h"] = _window_counts(rcv, secs, 86_400, snd)
    f["receiver_distinct_senders_7d"] = _window_counts(rcv, secs, 7 * 86_400, snd)

    # last inflow into the *sender* strictly before this tx (pass-through / mule / layering)
    inflow = pd.DataFrame({"acc": rcv, "in_s": secs, "in_amt": t["amount_inr"].to_numpy(),
                           "in_row": np.arange(len(t))})
    out = pd.DataFrame({"acc": snd, "s": secs, "row": np.arange(len(t))})
    j = pd.merge_asof(out.sort_values("s"), inflow.sort_values("in_s"), left_on="s", right_on="in_s",
                      by="acc", direction="backward", allow_exact_matches=False).sort_values("row")
    gap = (j["s"] - j["in_s"]).to_numpy(dtype=float)
    ratio = (t["amount_inr"].to_numpy() / j["in_amt"].to_numpy())
    f["sec_since_sender_last_inflow"] = np.nan_to_num(gap, nan=30 * 86_400.0)
    f["forward_ratio"] = np.nan_to_num(ratio, nan=0.0)
    pt = (gap <= PASS_THROUGH_WINDOW_SEC) & (ratio >= PASS_THROUGH_MIN_RATIO) & (ratio <= PASS_THROUGH_MAX_RATIO)
    f["is_pass_through"] = np.nan_to_num(pt, nan=0).astype(np.int8)

    depth = np.zeros(len(t), dtype=np.int16)                     # rows are in time order
    in_row = j["in_row"].to_numpy()
    for i in np.flatnonzero(f["is_pass_through"].to_numpy()):
        depth[i] = depth[int(in_row[i])] + 1
    f["chain_depth"] = depth

    f["device_distinct_accounts"] = _distinct_to_date(t["device_id"].to_numpy(), snd)
    f["ip_distinct_accounts"] = _distinct_to_date(t["ip_address"].to_numpy(), snd)
    return f


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Weighted directed account graph from `edges` (src, dst, amount_inr)."""
    agg = edges.groupby(["src", "dst"], as_index=False).agg(weight=("amount_inr", "sum"), n=("amount_inr", "size"))
    g = nx.DiGraph()
    g.add_weighted_edges_from(agg[["src", "dst", "weight"]].itertuples(index=False))
    nx.set_edge_attributes(g, dict(zip(zip(agg.src, agg.dst), agg.n)), "n_tx")
    return g


def account_metrics(g: nx.DiGraph, seed: int = 42) -> pd.DataFrame:
    ug = g.to_undirected()
    pr = nx.pagerank(g, weight=None)
    comms = nx.community.louvain_communities(ug, seed=seed)
    comm_id = {n: i for i, c in enumerate(comms) for n in c}
    comm_size = {n: len(c) for c in comms for n in c}
    clust = nx.clustering(ug)
    nodes = list(g.nodes)
    return pd.DataFrame({
        "account_id": nodes,
        "in_degree": [g.in_degree(n) for n in nodes],
        "out_degree": [g.out_degree(n) for n in nodes],
        "pagerank": [pr[n] for n in nodes],
        "community_id": [comm_id[n] for n in nodes],
        "community_size": [comm_size[n] for n in nodes],
        "clustering": [clust[n] for n in nodes],
    })


def global_features(tx: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    m = metrics.set_index("account_id")
    s, r = tx["sender_account_id"], tx["receiver_account_id"]
    return pd.DataFrame({
        "transaction_id": tx["transaction_id"].to_numpy(),
        "sender_pagerank": s.map(m["pagerank"]).to_numpy(),
        "receiver_pagerank": r.map(m["pagerank"]).to_numpy(),
        "sender_community_size": s.map(m["community_size"]).to_numpy(),
        "receiver_community_size": r.map(m["community_size"]).to_numpy(),
        "same_community": (s.map(m["community_id"]) == r.map(m["community_id"])).astype(np.int8).to_numpy(),
        "receiver_clustering": r.map(m["clustering"]).to_numpy(),
    })


def network_features(tx: pd.DataFrame, edges: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-transaction network features, per-account metrics)."""
    metrics = account_metrics(build_graph(edges), seed=seed)
    feats = point_in_time_features(tx).merge(global_features(tx, metrics), on="transaction_id", validate="one_to_one")
    return feats, metrics
