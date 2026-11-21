"""Fuzzy sanctions screening over the unified OFAC + UN entity list (RapidFuzz)."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz, process

from finsight.sanctions.text import normalise_name

DEFAULT_THRESHOLD = 90.0


@dataclass
class Match:
    entity_uid: str
    source: str
    matched_name: str
    primary_name: str
    entity_type: str
    program: str | None
    score: float


class SanctionsScreener:
    def __init__(self, entities: pd.DataFrame):
        self.entities = entities.reset_index(drop=True)
        self._choices = self.entities["name_norm"].tolist()

    def screen(self, name: str, threshold: float = DEFAULT_THRESHOLD, limit: int = 5) -> list[Match]:
        q = normalise_name(name)
        if len(q) < 3:
            return []
        # token_sort_ratio handles "LAST, First" vs "First Last"; WRatio guards partial matches
        hits = process.extract(q, self._choices, scorer=fuzz.token_sort_ratio, limit=limit * 4,
                               score_cutoff=threshold)
        best: dict[str, Match] = {}
        for _, score, i in hits:
            r = self.entities.iloc[i]
            if r.entity_uid not in best or score > best[r.entity_uid].score:
                best[r.entity_uid] = Match(r.entity_uid, r.source, r["name"], r.primary_name,
                                           r.entity_type, r.program, round(float(score), 1))
        return sorted(best.values(), key=lambda m: -m.score)[:limit]

    def screen_many(self, names: pd.Series, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
        rows = []
        for n in names:
            m = self.screen(n, threshold, limit=1)
            rows.append({"name": n, "hit": bool(m), "best_score": m[0].score if m else None,
                         "best_match": m[0].matched_name if m else None,
                         "entity_uid": m[0].entity_uid if m else None})
        return pd.DataFrame(rows)


def threshold_sweep(screener: SanctionsScreener, names: pd.Series,
                    thresholds=(70, 75, 80, 85, 90, 95)) -> pd.DataFrame:
    """DMW mini-experiment: alert rate vs fuzzy threshold on a population that is (by construction)
    not sanctioned — every hit is a false positive."""
    q = names.map(normalise_name).tolist()
    choices = screener._choices
    best = [process.extractOne(x, choices, scorer=fuzz.token_sort_ratio)[1] for x in q]
    s = pd.Series(best)
    return pd.DataFrame({"threshold": thresholds,
                         "alerts": [int((s >= t).sum()) for t in thresholds],
                         "false_positive_rate": [round(float((s >= t).mean()), 4) for t in thresholds]})
