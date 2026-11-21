import filecmp

import numpy as np
import pandas as pd
import pytest

from finsight import config
from finsight.data.inject import inject
from finsight.data.loader import load_india

pytestmark = pytest.mark.skipif(not config.INDIA_V1_0_DIR.exists(), reason="dataset not present")


@pytest.fixture(scope="module")
def v10():
    return load_india(config.INDIA_V1_0_DIR)


@pytest.fixture(scope="module")
def v11(tmp_path_factory):
    out = tmp_path_factory.mktemp("v11")
    inject(out_dir=out)
    return out, load_india(out)


def test_labels_unchanged(v10, v11):
    _, d = v11
    a = v10.targets.set_index("transaction_id").sort_index()
    b = d.targets.set_index("transaction_id").sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_same_transactions_and_schema(v10, v11):
    _, d = v11
    assert list(v10.transactions.columns) == list(d.transactions.columns)
    assert set(v10.transactions.transaction_id) == set(d.transactions.transaction_id)


def test_invariants(v11):
    _, d = v11
    tx = d.transactions
    acc = d.accounts.set_index("account_id")
    assert (tx.sender_account_id != tx.receiver_account_id).all()
    assert np.allclose(tx.account_balance_before - tx.amount_inr, tx.account_balance_after, atol=0.02)
    ratio = tx.amount_inr / tx.sender_account_id.map(acc.historical_30d_avg_tx_amt_inr)
    assert np.isclose(ratio, tx.amount_ratio_vs_30d_avg, rtol=1e-2).mean() > 0.98
    assert (tx.sender_account_id.map(acc.customer_id) == tx.sender_customer_id).all()
    assert (tx.receiver_account_id.map(acc.customer_id) == tx.receiver_customer_id).all()
    assert not ((tx.channel == "RTGS") & (tx.amount_inr < 200_000)).any()
    assert tx.timestamp.dt.month.eq(8).all()
    assert (tx.hour_of_day == tx.timestamp.dt.hour).all()


def test_device_belongs_to_sender_except_shared(v11):
    out, d = v11
    log = pd.read_csv(out / "injection_log.csv")
    shared = set(log.loc[log.group_id.str.contains("shared_dev|family_dev"), "transaction_id"])
    owner = d.devices.set_index("device_id").customer_id
    tx = d.transactions[~d.transactions.transaction_id.isin(shared)]
    assert (tx.device_id.map(owner) == tx.sender_customer_id).all()


def test_deterministic(v11, tmp_path):
    out, _ = v11
    inject(out_dir=tmp_path)
    for f in ("transactions_features.csv", "graph_edges.csv", "injection_log.csv", "devices.csv"):
        assert filecmp.cmp(out / f, tmp_path / f, shallow=False), f


def test_v10_untouched(v10):
    assert len(v10.transactions) == 10_000
    assert (v10.edges.rapid_forward_delta_sec > 0).sum() == 350    # the original label proxy
