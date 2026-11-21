import pandas as pd
import pytest

from finsight import config
from finsight.graph.network_features import point_in_time_features
from finsight.sanctions.parser import normalise_name
from finsight.sanctions.screener import SanctionsScreener


def _tx(rows):
    df = pd.DataFrame(rows, columns=["transaction_id", "timestamp", "sender_account_id",
                                     "receiver_account_id", "amount_inr", "device_id", "ip_address"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(config.TIMEZONE)
    return df


def test_point_in_time_pass_through_chain():
    tx = _tx([
        ("t1", "2026-08-01 10:00", "A", "B", 1000.0, "d1", "ip1"),
        ("t2", "2026-08-01 10:10", "B", "C", 950.0, "d2", "ip2"),   # B forwards 95% in 10 min
        ("t3", "2026-08-01 10:20", "C", "D", 900.0, "d3", "ip3"),   # C forwards again
        ("t4", "2026-08-01 15:00", "D", "E", 100.0, "d3", "ip3"),   # late + small → not pass-through
    ])
    f = point_in_time_features(tx).set_index("transaction_id")
    assert f.loc["t1", "is_pass_through"] == 0
    assert f.loc["t2", "is_pass_through"] == 1 and f.loc["t2", "chain_depth"] == 1
    assert f.loc["t3", "chain_depth"] == 2
    assert f.loc["t4", "is_pass_through"] == 0
    assert f.loc["t4", "device_distinct_accounts"] == 2              # d3 used by C then D


def test_point_in_time_excludes_future():
    tx = _tx([(f"t{i}", f"2026-08-01 10:0{i}", f"S{i}", "HUB", 10.0, f"d{i}", f"ip{i}") for i in range(5)])
    f = point_in_time_features(tx).set_index("transaction_id")
    assert list(f["receiver_distinct_senders_24h"]) == [0, 1, 2, 3, 4]


def test_normalise_name():
    assert normalise_name("  Müller-Ödön, J. ") == "muller odon j"


@pytest.mark.skipif(not (config.RAW_DIR / "sanctions").exists(), reason="sanctions data not present")
def test_screener_finds_known_entities():
    from finsight.sanctions.parser import build_entities

    s = SanctionsScreener(build_entities())
    assert s.screen("Banco Nacional de Cuba")[0].source == "OFAC"
    assert s.screen("BNC", threshold=95)                         # OFAC a.k.a. parsed from remarks
    assert s.screen("Mousa Mohammed Abu Marzook")                # OFAC 'LAST, First' reordering
    assert s.screen("Ayaan Sharma") == []
