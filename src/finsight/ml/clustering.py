"""Module 2: clustering (Plan §7.4, RQ5).

The input is the scaled feature matrix for one experiment: the same preprocessing as the
distance-based classifiers (signed-log1p → StandardScaler on numerics, one-hot categoricals).
Labels are never used for fitting. They are only used afterwards, to describe the clusters
(fraud rate, scenario mix, ARI / NMI against `risk_label`).

* K-Means for K = 2..8: inertia (elbow), silhouette, Davies–Bouldin. K = the lowest Davies–Bouldin
  among the K values whose silhouette is within 0.005 of the best (silhouette is flat here; D-026)
* DBSCAN: ε from the knee of the k-distance curve (k = min_samples); noise points compared with labels
* 2-D projections: PCA (linear) and t-SNE (non-linear), coloured by cluster and by label
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from finsight import config
from finsight.ml.classifiers import preprocessor
from finsight.ml.feature_sets import feature_columns

K_RANGE = range(2, 9)


def scaled_matrix(m: pd.DataFrame, exp: str, fit_rows: np.ndarray | None = None):
    """Return (X_scaled, fitted preprocessor). Fit on `fit_rows` (default: all rows; clustering has no labels to leak)."""
    cols = feature_columns(exp)
    prep = preprocessor(exp, scaled=True)
    prep.fit(m[cols].iloc[fit_rows] if fit_rows is not None else m[cols])
    return prep.transform(m[cols]).astype(np.float64), prep


def kmeans_sweep(X: np.ndarray, ks=K_RANGE, seed: int = config.SEED) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_, random_state=seed),
                     "davies_bouldin": davies_bouldin_score(X, km.labels_)})
    return pd.DataFrame(rows)


def pick_k(sweep: pd.DataFrame, tol: float = 0.005) -> int:
    """Silhouette plateau (within `tol` of the best) → lowest Davies–Bouldin inside it."""
    plateau = sweep[sweep["silhouette"] >= sweep["silhouette"].max() - tol]
    return int(plateau.sort_values(["davies_bouldin", "k"]).iloc[0]["k"])


def fit_kmeans(X: np.ndarray, k: int, seed: int = config.SEED) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)


def cluster_profile(km: KMeans, feature_names, top: int = 4) -> dict[int, list[tuple[str, float]]]:
    """Features with the largest |centroid| (in standard-deviation units) per cluster: what defines it."""
    c = pd.DataFrame(km.cluster_centers_, columns=list(feature_names))
    return {int(i): [(f, round(float(c.loc[i, f]), 2)) for f in c.loc[i].abs().sort_values(ascending=False).index[:top]]
            for i in c.index}


def k_distance(X: np.ndarray, k: int) -> np.ndarray:
    """Sorted distance of every point to its k-th nearest neighbour (excluding itself)."""
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    return np.sort(d[:, -1])


def knee(y: np.ndarray) -> int:
    """Index of maximum distance to the chord joining the first and last point (the Kneedle idea)."""
    y = np.asarray(y, float)
    x = np.linspace(0, 1, len(y))
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    return int(np.argmax(x - yn)) if yn[len(yn) // 2] < x[len(x) // 2] else int(np.argmax(yn - x))


def fit_dbscan(X: np.ndarray, min_samples: int) -> tuple[DBSCAN, float, np.ndarray]:
    kd = k_distance(X, min_samples)
    eps = float(kd[knee(kd)])
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X), eps, kd


def describe_clusters(m: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Size, fraud rate, label shares and dominant non-normal scenario per cluster (post hoc only)."""
    d = pd.DataFrame({"cluster": labels, "risk_label": m["risk_label"].astype(str).to_numpy(),
                      "is_fraud": m["is_fraud"].to_numpy(), "scenario": m["scenario"].to_numpy()})
    out = d.groupby("cluster").agg(n=("is_fraud", "size"), fraud_rate=("is_fraud", "mean"))
    shares = pd.crosstab(d["cluster"], d["risk_label"], normalize="index")
    out = out.join(shares.add_prefix("share_"))
    nonnormal = d[~d["scenario"].str.startswith("S1 ")]
    top = nonnormal.groupby("cluster")["scenario"].agg(lambda s: s.value_counts().index[0] if len(s) else "")
    out["top_scenario"] = top.reindex(out.index).fillna("")
    return out.reset_index()


def agreement(m: pd.DataFrame, labels: np.ndarray) -> dict:
    y = m["risk_label"].astype(str).to_numpy()
    return {"ari": float(adjusted_rand_score(y, labels)), "nmi": float(normalized_mutual_info_score(y, labels))}


def best_fraud_cluster_scores(m: pd.DataFrame, labels: np.ndarray) -> dict:
    """If an analyst flagged every cluster whose fraud rate beats the base rate, what precision/recall would they get?"""
    y = m["is_fraud"].to_numpy()
    rate = pd.Series(y).groupby(labels).mean()
    flagged = rate[rate > y.mean()].index
    pred = np.isin(labels, flagged)
    tp = int((pred & (y == 1)).sum())
    return {"flagged_clusters": [int(c) for c in flagged],
            "precision": tp / max(int(pred.sum()), 1), "recall": tp / max(int(y.sum()), 1)}


def projections(X: np.ndarray, seed: int = config.SEED, tsne: bool = True) -> dict[str, np.ndarray]:
    pca = PCA(n_components=2, random_state=seed).fit(X)
    out = {"pca": pca.transform(X), "pca_var": pca.explained_variance_ratio_}
    if tsne:
        out["tsne"] = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto",
                           random_state=seed, n_jobs=-1).fit_transform(X)
    return out
