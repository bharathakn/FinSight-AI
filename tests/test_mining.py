import pandas as pd
import pytest

from finsight import config
from finsight.mining import association as A
from finsight.mining.discretiser import INDIA_ITEM_RULES, LABEL_ITEMS


@pytest.fixture
def toy():
    rows = [["a", "b", "F"], ["a", "b", "F"], ["a", "b", "c", "F"], ["a", "c"], ["b", "c"], ["a"], ["c"], ["a", "b"]]
    items = sorted({i for r in rows for i in r})
    df = pd.DataFrame([[i in r for i in items] for r in rows], columns=items)
    return df.rename(columns={"F": "FRAUDULENT"})


def test_apriori_equals_fpgrowth(toy):
    a = A.mine(toy, "apriori", 0.2, track_memory=False).itemsets
    f = A.mine(toy, "fpgrowth", 0.2, track_memory=False).itemsets
    assert A._as_set(a) == A._as_set(f)


def test_closed_and_maximal(toy):
    fs = A.mine(toy, "fpgrowth", 0.2, track_memory=False).itemsets
    closed = {frozenset(s) for s in A.closed_itemsets(fs)["itemsets"]}
    # {a,b} (5/8) and {a,b,F} (3/8): {b,F} has support 3/8 = {a,b,F} → not closed
    assert frozenset({"a", "b", "FRAUDULENT"}) in closed
    assert frozenset({"b", "FRAUDULENT"}) not in closed
    maximal = A.mine(toy, "fpmax", 0.2, track_memory=False).itemsets
    assert len(maximal) <= len(closed) <= len(fs)


def test_target_rules_and_productive(toy):
    fs = A.mine(toy, "fpgrowth", 0.2, track_memory=False).itemsets
    r = A.target_rules(fs, len(toy), targets=["FRAUDULENT"], min_confidence=0.5)
    assert (r["consequent"] == "FRAUDULENT").all()
    assert not r["antecedents"].map(lambda a: bool(set(a) & set(LABEL_ITEMS))).any()
    p = A.productive_rules(r)
    # {a,b}→F (conf 3/5) must survive; {a,b,c}→F is a superset and must beat its subsets to survive
    assert "{a, b} → FRAUDULENT" in set(p["rule"])


def test_rule_detector(toy):
    rules = pd.DataFrame({"antecedents": [frozenset({"a", "b"})], "consequent": ["FRAUDULENT"]})
    d = A.rule_detector(toy.drop(columns="FRAUDULENT"), rules, toy["FRAUDULENT"].astype(int))
    assert (d["tp"], d["fp"], d["fn"]) == (3, 1, 0)


@pytest.mark.skipif(not (config.CURATED_DIR / "master_features_v1_1.parquet").exists(), reason="run make pipeline")
def test_india_items_match_rules():
    from finsight.eda import load_master
    from finsight.mining.discretiser import india_items

    m = load_master("v1_1")
    it = india_items(m)
    assert list(it.columns) == list(INDIA_ITEM_RULES) + LABEL_ITEMS
    assert it["PASS_THROUGH"].sum() == (m["is_pass_through"] == 1).sum()
    assert it.dtypes.eq(bool).all()
