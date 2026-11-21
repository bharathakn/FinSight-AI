"""Train / validation / test splits (Plan §7.1).

* `stratified_split` — 60/20/20, stratified on `risk_label` (keeps BORDERLINE proportions)
* `temporal_split`   — train Aug 1–19, validation Aug 20–23, test Aug 24–30 (robustness check)
* `cv_folds`         — repeated stratified K-fold on the dev set (train + val) for paired comparisons
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from finsight import config


@dataclass(frozen=True)
class Split:
    name: str
    train: np.ndarray      # positional indices into the master frame
    val: np.ndarray
    test: np.ndarray

    @property
    def dev(self) -> np.ndarray:
        return np.sort(np.concatenate([self.train, self.val]))

    def sizes(self) -> dict:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


def stratified_split(m: pd.DataFrame, seed: int = config.SEED, test_size: float = 0.2, val_size: float = 0.2) -> Split:
    idx = np.arange(len(m))
    strat = m["risk_label"].astype(str).to_numpy()
    dev, test = train_test_split(idx, test_size=test_size, stratify=strat, random_state=seed)
    train, val = train_test_split(dev, test_size=val_size / (1 - test_size), stratify=strat[dev], random_state=seed)
    return Split("stratified_60_20_20", np.sort(train), np.sort(val), np.sort(test))


def temporal_split(m: pd.DataFrame, val_start: str = "2026-08-20", test_start: str = "2026-08-24") -> Split:
    day = m["timestamp"].dt.strftime("%Y-%m-%d").to_numpy()
    idx = np.arange(len(m))
    return Split("temporal_aug1-19_20-23_24-30", idx[day < val_start],
                 idx[(day >= val_start) & (day < test_start)], idx[day >= test_start])


def cv_folds(m: pd.DataFrame, dev: np.ndarray, n_splits: int = 5, n_repeats: int = 2,
             seed: int = config.SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds over `dev` (positional indices into m), stratified on risk_label. Same folds for every model ⇒ paired."""
    strat = m["risk_label"].astype(str).to_numpy()[dev]
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return [(dev[tr], dev[te]) for tr, te in rskf.split(dev, strat)]
