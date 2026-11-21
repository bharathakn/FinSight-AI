import pandas as pd
import pytest

from finsight import config
from finsight.ml.feature_sets import EXPERIMENTS, assert_no_leakage, feature_columns, get_xy


def test_feature_sets_nested_and_clean():
    a, b, c = (set(feature_columns(e)) for e in "ABC")
    assert a < b < c
    for e in EXPERIMENTS:
        assert_no_leakage(feature_columns(e))


@pytest.mark.parametrize("bad", ["is_fraud", "rapid_forward_delta_sec", "is_flagged_alert", "sender_account_id"])
def test_blacklist_rejected(bad):
    with pytest.raises(ValueError):
        assert_no_leakage(["log_amount", bad])


master = config.CURATED_DIR / "master_features_v1_1.parquet"


@pytest.mark.skipif(not master.exists(), reason="run `make pipeline` first")
def test_get_xy_and_eda_tables():
    from finsight.eda import feature_ranking, flag_table, load_master

    m = load_master("v1_1")
    X, y = get_xy(m, "C")
    assert len(X) == len(y) == 10_000 and X.isna().sum().sum() == 0
    rank = feature_ranking(m)
    assert rank["auc"].between(0.5, 1).all()
    assert rank["auc"].max() < 0.95                           # no single-feature giveaway on v1.1
    flags = flag_table(m).set_index("flag")
    assert flags.loc["RECENT_BENEFICIARY (<24h)", "share_LEGITIMATE"] == 0   # documented generator artifact
