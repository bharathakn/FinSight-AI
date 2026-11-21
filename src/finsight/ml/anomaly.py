"""Module 3: anomaly detection (Plan §7.5, RQ6).

Each detector is fitted **only on LEGITIMATE rows of the training split** (no fraud and no
BORDERLINE rows), so it learns "normal" and flags whatever does not fit. The labels are used
afterwards for evaluation only.

  Isolation Forest   score = −score_samples   (shorter average path ⇒ more anomalous)
  One-class SVM      score = −decision_function (RBF, ν ∈ {0.01, 0.05})
  Autoencoder        score = mean squared reconstruction error
                     (PyTorch, d → 32 → 16 → 8 → 16 → 32 → d, early stopping on legit validation rows,
                     exported to ONNX for serving with onnxruntime)

Thresholds are percentiles (95th / 99th) of the scores of **legitimate validation rows**. Using
held-out rows instead of training rows means in-sample fit (the AE in particular) doesn't make
the threshold too low (D-025).
`anomaly_pct` maps any score to its percentile in that legit reference set; it becomes the
anomaly component of the risk score in Phase 8.

The PaySim scale track runs a separate Isolation Forest on 2.77 M PaySim TRANSFER/CASH_OUT rows,
wrangled lazily with Polars and evaluated against PaySim's own `isFraud`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.svm import OneClassSVM

from finsight import config
from finsight.ml.classifiers import preprocessor
from finsight.ml.feature_sets import feature_columns

PERCENTILES = (95, 99)
DETECTOR_LABELS = {"isolation_forest": "Isolation Forest", "ocsvm_nu0.01": "OC-SVM (ν=0.01)",
                   "ocsvm_nu0.05": "OC-SVM (ν=0.05)", "autoencoder": "Autoencoder"}


# ---------------------------------------------------------------- autoencoder (PyTorch → ONNX)

def _torch():
    import torch
    from torch import nn

    return torch, nn


def make_autoencoder(d: int, hidden=(32, 16, 8)):
    torch, nn = _torch()
    enc, dec, prev = [], [], d
    for h in hidden:
        enc += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    for h in list(hidden[-2::-1]) + [d]:
        dec += [nn.Linear(prev, h)] + ([nn.ReLU()] if h != d else [])
        prev = h
    return nn.Sequential(*enc, *dec)


def train_autoencoder(X_train: np.ndarray, X_val: np.ndarray, seed: int = config.SEED, max_epochs: int = 600,
                      patience: int = 20, batch_size: int = 128, lr: float = 1e-3):
    """Mini-batch Adam on MSE with early stopping on validation loss. Returns (model, history DataFrame)."""
    torch, nn = _torch()
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    model = make_autoencoder(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    best, best_state, wait, hist = np.inf, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        tot = 0.0
        for i in range(0, len(Xt), batch_size):
            xb = Xt[perm[i:i + batch_size]]
            opt.zero_grad()
            loss = loss_fn(model(xb), xb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xv), Xv).item()
        hist.append({"epoch": epoch, "train_loss": tot / len(Xt), "val_loss": vl})
        if vl < best - 1e-6:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(hist)


def _with_error(model):
    """Wrap the AE so the exported graph returns (reconstruction, per-row MSE)."""
    torch, nn = _torch()

    class AEWithError(nn.Module):
        def __init__(self, ae):
            super().__init__()
            self.ae = ae

        def forward(self, x):
            r = self.ae(x)
            return r, ((x - r) ** 2).mean(dim=1)

    return AEWithError(model).eval()


def torch_ae_score(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    torch, _ = _torch()
    with torch.no_grad():
        r, e = _with_error(model)(torch.tensor(X, dtype=torch.float32))
    return r.numpy(), e.numpy()


def export_onnx(model, d: int, path: Path) -> Path:
    """Legacy TorchScript exporter (no onnxscript dependency); batch dimension is dynamic."""
    torch, _ = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(_with_error(model), (torch.zeros(2, d),), str(path), input_names=["x"],
                      output_names=["reconstruction", "error"], opset_version=17, dynamo=False,
                      dynamic_axes={"x": {0: "n"}, "reconstruction": {0: "n"}, "error": {0: "n"}})
    return path


class OnnxAutoencoder:
    """Serving-side AE: onnxruntime only (no torch import)."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

    def run(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r, e = self.session.run(None, {"x": np.asarray(X, dtype=np.float32)})
        return r, e

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self.run(X)[1]


# ---------------------------------------------------------------- detector wrapper

@dataclass
class Detector:
    """Preprocessor + model + legit reference scores. `score` is always 'higher = more anomalous'."""
    name: str
    exp: str
    prep: object
    model: object
    reference: np.ndarray = field(default_factory=lambda: np.array([]))   # sorted legit-validation scores

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.prep.transform(df[feature_columns(self.exp)]).astype(np.float32)

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        if self.name == "isolation_forest":
            return -self.model.score_samples(X)
        if self.name.startswith("ocsvm"):
            return -self.model.decision_function(X)
        if isinstance(self.model, OnnxAutoencoder):
            return self.model.score_samples(X)
        return torch_ae_score(self.model, X)[1]

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.score_matrix(self.transform(df))

    def threshold(self, pct: float) -> float:
        return float(np.percentile(self.reference, pct))

    def anomaly_pct(self, scores: np.ndarray) -> np.ndarray:
        """Percentile (0–1) of each score in the legit reference distribution."""
        return np.searchsorted(self.reference, scores, side="right") / len(self.reference)


def fit_detectors(m: pd.DataFrame, split, exp: str = "C", seed: int = config.SEED,
                  ocsvm_nus=(0.01, 0.05)) -> tuple[dict[str, Detector], pd.DataFrame]:
    """Fit IF, OC-SVM(s) and the AE on legit training rows; set references on legit validation rows."""
    legit = m["risk_label"].astype(str).to_numpy() == "LEGITIMATE"
    tr, va = split.train[legit[split.train]], split.val[legit[split.val]]
    cols = feature_columns(exp)
    prep = preprocessor(exp, scaled=True).fit(m[cols].iloc[tr])
    Xtr = prep.transform(m[cols].iloc[tr]).astype(np.float32)
    Xva = prep.transform(m[cols].iloc[va]).astype(np.float32)

    dets = {"isolation_forest": Detector("isolation_forest", exp, prep,
                                         IsolationForest(n_estimators=300, random_state=seed, n_jobs=-1).fit(Xtr))}
    for nu in ocsvm_nus:
        dets[f"ocsvm_nu{nu}"] = Detector(f"ocsvm_nu{nu}", exp, prep,
                                         OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(Xtr))
    ae, hist = train_autoencoder(Xtr, Xva, seed=seed)
    dets["autoencoder"] = Detector("autoencoder", exp, prep, ae)
    for d in dets.values():
        d.reference = np.sort(d.score_matrix(Xva))
    return dets, hist


# ---------------------------------------------------------------- evaluation

def recall_at_fpr(y, s, fpr_max: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max())


def evaluate(y, s, thresholds: dict[str, float], borderline: np.ndarray | None = None) -> dict:
    y = np.asarray(y).astype(int)
    out = {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
           "recall_at_1pct_fpr": recall_at_fpr(y, s, 0.01)}
    for k, t in thresholds.items():
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        prec, rec = tp / max(int(pred.sum()), 1), tp / max(int(y.sum()), 1)
        out |= {f"{k}_threshold": float(t), f"{k}_precision": prec, f"{k}_recall": rec,
                f"{k}_f1": 2 * prec * rec / max(prec + rec, 1e-12), f"{k}_flag_rate": float(pred.mean())}
        if borderline is not None:
            out[f"{k}_borderline_flag_rate"] = float(pred[borderline].mean())
    return out


# ---------------------------------------------------------------- PaySim scale track

PAYSIM_RAW = ["is_transfer", "hour", "log_amount", "log_oldbalanceOrg", "log_newbalanceOrig",
              "log_oldbalanceDest", "log_newbalanceDest"]
PAYSIM_ERR = PAYSIM_RAW + ["slog_err_orig", "slog_err_dest"]


def paysim_features(lf=None):
    """Lazy Polars plan: TRANSFER/CASH_OUT only (the only types with fraud), log-scaled amounts and balances,
    and the two balance-inconsistency features (amount not reflected in the sender/receiver balances)."""
    import polars as pl

    from finsight.data.loader import scan_paysim

    lf = scan_paysim() if lf is None else lf
    slog = lambda e: e.sign() * e.abs().log1p()  # noqa: E731
    return (lf.filter(pl.col("type").is_in(["TRANSFER", "CASH_OUT"]))
            .with_columns(
                is_transfer=(pl.col("type") == "TRANSFER").cast(pl.Int8),
                hour=(pl.col("step") % 24).cast(pl.Int16),
                log_amount=pl.col("amount").log1p(),
                log_oldbalanceOrg=pl.col("oldbalanceOrg").log1p(),
                log_newbalanceOrig=pl.col("newbalanceOrig").log1p(),
                log_oldbalanceDest=pl.col("oldbalanceDest").log1p(),
                log_newbalanceDest=pl.col("newbalanceDest").log1p(),
                slog_err_orig=slog(pl.col("newbalanceOrig") + pl.col("amount") - pl.col("oldbalanceOrg")),
                slog_err_dest=slog(pl.col("oldbalanceDest") + pl.col("amount") - pl.col("newbalanceDest")),
            )
            .select(PAYSIM_ERR + ["isFraud"]))


def paysim_isolation_forest(fit_sample: int = 250_000, seed: int = config.SEED) -> dict:
    """Fully unsupervised IF (fit on a random sample of all rows, no labels) → score every row.
    Two feature sets: raw balances vs + balance-error features. Returns metrics + stage timings."""
    import resource

    t0 = time.perf_counter()
    df = paysim_features().collect(engine="streaming")
    t_wrangle = time.perf_counter() - t0
    y = df["isFraud"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(fit_sample, len(df)), replace=False)
    out = {"rows_scored": int(len(df)), "fraud_rows": int(y.sum()), "fraud_rate": float(y.mean()),
           "fit_sample": int(len(idx)), "wrangle_seconds": round(t_wrangle, 2), "feature_sets": {}}
    scores = {}
    for name, cols in (("raw", PAYSIM_RAW), ("raw+balance_error", PAYSIM_ERR)):
        X = df.select(cols).to_numpy().astype(np.float32)
        t1 = time.perf_counter()
        iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(X[idx])
        t_fit = time.perf_counter() - t1
        t2 = time.perf_counter()
        s = -iso.score_samples(X)
        t_score = time.perf_counter() - t2
        thr = {f"p{p}": float(np.percentile(s[idx], p)) for p in PERCENTILES}
        out["feature_sets"][name] = {"n_features": len(cols), "fit_seconds": round(t_fit, 2),
                                     "score_seconds": round(t_score, 2),
                                     "rows_per_second": round(len(X) / t_score), **evaluate(y, s, thr)}
        scores[name] = s
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return out | {"_scores": scores, "_y": y}
