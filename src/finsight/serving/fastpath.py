"""Low-latency serving paths (D-034). Each is checked for exact equality against the original code in tests.

CompiledPreprocessor   a fitted `finsight.ml.classifiers.preprocessor` ColumnTransformer turned into numpy ops on
                       row dicts: numeric → [signed-log1p → (x − mean) / scale] or passthrough; categorical → one-hot
                       with the fitted categories (unknown → all zeros, like handle_unknown="ignore").
                       Integer columns are cast to their training dtype before the log: numpy's log1p on int8
                       returns float16 (int16 → float32), and the models were fitted on those values, so exact
                       parity needs the same rounding.
                       sklearn's ColumnTransformer costs ~10 ms per single-row call (DataFrame column selection);
                       this costs ~0.05 ms, and it is applied four times per score.
items_for_record       the Phase 4 item rules compiled to Python bytecode once and evaluated on a dict
                       (pandas `df.eval` costs ~3 ms per rule per call).
"""
from __future__ import annotations

import math

import numpy as np

from finsight.mining.discretiser import INDIA_ITEM_RULES

_ITEM_CODE = {n: compile(expr, f"<item {n}>", "eval") for n, expr in INDIA_ITEM_RULES.items()}
_NO_BUILTINS = {"__builtins__": {}}


def _clean(v):
    if v is None:
        return math.nan
    if isinstance(v, (np.generic,)):
        return v.item()
    return v


def items_for_record(rec: dict) -> set[str]:
    """Names of the Phase 4 items that are true for one transaction (same semantics as `india_items`)."""
    env = {k: _clean(v) for k, v in rec.items()}
    for k in ("channel", "sender_kyc_status"):
        if k in env and not isinstance(env[k], str):
            env[k] = str(env[k])
    if "amount_inr" not in env and isinstance(env.get("log_amount"), (int, float)):
        env["amount_inr"] = math.expm1(env["log_amount"])
    out = set()
    for name, code in _ITEM_CODE.items():
        try:
            if eval(code, _NO_BUILTINS, env):          # noqa: S307 — fixed, trusted expressions compiled at import
                out.add(name)
        except (TypeError, NameError):
            pass                                          # missing / non-comparable value → item is false
    return out


class CompiledPreprocessor:
    def __init__(self, ct, dtypes: dict | None = None):
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler   # ml image only

        from finsight.ml.classifiers import signed_log1p

        self.dtypes = {k: np.dtype(v) for k, v in (dtypes or {}).items()
                       if isinstance(v, np.dtype) and v.kind in "iu"}     # only integer columns matter

        self.blocks = []
        for name, trans, cols in ct.transformers_:
            if name == "remainder" or trans == "drop":
                continue
            cols = list(cols)
            if isinstance(trans, str) and trans == "passthrough" or \
                    (isinstance(trans, FunctionTransformer) and trans.func is None):   # fitted 'passthrough' (sklearn ≥ 1.4)
                self.blocks.append(("num", cols, False, None, None))
            elif isinstance(trans, OneHotEncoder):
                cats = [{str(c): i for i, c in enumerate(cs)} for cs in trans.categories_]
                self.blocks.append(("cat", cols, cats))
            else:                                        # Pipeline(FunctionTransformer(signed_log1p), StandardScaler)
                steps = [s for _, s in trans.steps]
                if len(steps) != 2 or getattr(steps[0], "func", None) is not signed_log1p or not isinstance(steps[1], StandardScaler):
                    raise TypeError(f"unsupported numeric transformer for fast path: {trans}")
                sc = steps[1]
                self.blocks.append(("num", cols, True, sc.mean_, sc.scale_))
        self.feature_names = list(ct.get_feature_names_out())
        self.n_out = len(self.feature_names)

    def transform(self, rows: list[dict]) -> np.ndarray:
        n = len(rows)
        out = np.zeros((n, self.n_out), dtype=np.float64)
        j = 0
        for block in self.blocks:
            if block[0] == "num":
                _, cols, log, mean, scale = block
                A = np.array([[_clean(r.get(c)) for c in cols] for r in rows], dtype=np.float64)
                if log:
                    for k, c in enumerate(cols):
                        dt = self.dtypes.get(c)
                        if dt is not None and dt.kind in "iu" and not np.isnan(A[:, k]).any():
                            v = A[:, k].astype(dt)                       # replicate training-time dtype promotion
                            A[:, k] = (np.sign(v) * np.log1p(np.abs(v))).astype(np.float64)
                        else:
                            A[:, k] = np.sign(A[:, k]) * np.log1p(np.abs(A[:, k]))
                    A = (A - mean) / scale
                out[:, j:j + len(cols)] = A
                j += len(cols)
            else:
                _, cols, cats = block
                for col, mapping in zip(cols, cats):
                    for i, r in enumerate(rows):
                        k = mapping.get(str(r.get(col)))
                        if k is not None:
                            out[i, j + k] = 1.0
                    j += len(mapping)
        return out
