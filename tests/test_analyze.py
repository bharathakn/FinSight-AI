"""Phase 9b: live analysis of uploads, the demo dataset and simulated attacks."""
import io

import numpy as np
import pandas as pd
import pytest

from finsight.serving.settings import Settings

ROOT = Settings().artifacts_dir
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "reference").exists(), reason="run `make bundle`")


@pytest.fixture(scope="module")
def env():
    from finsight.scoring.risk_scorer import RiskScorer
    from finsight.serving import batch as B
    from finsight.serving.backends import InProcessBackend

    return {"B": B, "ref": B.Reference(ROOT), "be": InProcessBackend(ROOT),
            "sc": RiskScorer.load(ROOT / "models/final/risk_scorer.json")}


def _run(env, df, source=None):
    store = env["B"].AnalysisStore()
    s = env["B"].run(df, env["ref"], env["be"], env["sc"], store, source or {})
    return s, store.get(s["analysis_id"])


def test_numpy_metrics_equal_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score

    from finsight.serving.batch import average_precision, roc_auc
    rng = np.random.default_rng(3)
    y = (rng.random(4000) < 0.12).astype(int)
    s = np.round(rng.random(4000) * 50 * (1 + y))          # many ties on purpose
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_demo_dataset_reproduces_stored_features(env):
    from finsight.serving.store import CATEGORICAL, FEATURES
    s, a = _run(env, env["ref"].demo())
    stored = pd.read_parquet(ROOT / "data/transactions.parquet").set_index("transaction_id")
    got = a.features.set_index("transaction_id").loc[stored.index]
    num = [c for c in FEATURES if c not in CATEGORICAL]
    assert float((got[num].astype(float) - stored[num].astype(float)).abs().max().max()) == 0.0
    for c in CATEGORICAL:
        assert (got[c].astype(str) == stored[c].astype(str)).all(), c
    dq = s["data_quality"]
    assert dq["rows_valid"] == 10000 and not dq["defaulted_columns"] and not dq["imputed_features"]
    held = s["evaluation"]["held_out"]
    assert held["rows"] == 2000 and held["positives"] == 270 and held["pr_auc"] > 0.85


@pytest.mark.parametrize("camouflage", [False, True])
def test_simulated_attacks_are_detected(env, camouflage):
    from finsight.serving.simulate import ATTACKS, simulate
    for attack in ATTACKS:
        df, meta = simulate(env["ref"], [attack], size=10, camouflage=camouflage, background=800, seed=11)
        s, a = _run(env, df, {"type": "simulate", **meta})
        r = a.rows
        inj = r[r["scenario"].astype(str).str.startswith("SIM")]
        bg = r[~r["scenario"].astype(str).str.startswith("SIM")]
        alerted = inj["tier"].isin(["HIGH", "CRITICAL"]).mean()
        # camouflaged shared devices are the hardest case (only device sharing gives them away): 60–80% across seeds
        floor = 0.9 if not camouflage else (0.5 if attack == "shared_device" else 0.7)
        assert alerted >= floor, (attack, camouflage, alerted)
        assert bg["tier"].isin(["HIGH", "CRITICAL"]).mean() < 0.05, attack


def test_validation_and_defaults(env):
    B = env["B"]
    with pytest.raises(B.AnalysisError, match="Missing required column"):
        B.validate(pd.DataFrame({"a": [1]}), env["ref"])
    csv = ("transaction_id,timestamp,amount_inr,channel,sender_account_id,receiver_account_id\n"
           "T1,2026-08-31 10:00,48000,UPI,ACC_IND_000013,ACC_IND_000044\n"
           "T2,2026-08-31 10:05,-5,UPI,ACC_IND_000013,ACC_IND_000045\n"
           "T1,2026-08-31 10:06,100,UPI,ACC_IND_000013,ACC_IND_000045\n"
           "T3,not a date,100,UPI,ACC_IND_000013,ACC_IND_000045\n"
           "T4,2026-08-31 11:00,1200,BITCOIN,ACC_IND_000013,ACC_IND_000045\n"
           "T5,2026-08-31 12:00,1200,neft,ACC_NEW_1,ACC_IND_000013\n").encode()
    s, a = _run(env, B.read_csv(csv))
    dq = s["data_quality"]
    problems = {i["problem"]: i["rows"] for i in dq["issues"]}
    assert dq["rows_valid"] == 2 and problems["amount missing, non-numeric or ≤ 0"] == 1
    assert problems["timestamp could not be parsed"] == 1 and problems["duplicate transaction_id (first kept)"] == 1
    assert dq["unknown_accounts"] == 1 and "device_id" in dq["defaulted_columns"]
    assert dq["imputed_features"]["sender_account_age_days"] == 1          # the unknown sender
    assert a.rows["timestamp"].iloc[0].tzinfo is not None                    # naive → IST


def test_csv_export_neutralises_formulas(env):
    B = env["B"]
    s, a = _run(env, env["ref"].demo(200))
    a.rows.loc[a.rows.index[0], "sender_name"] = "=HYPERLINK(\"http://x\")"
    out = pd.read_csv(io.BytesIO(B.to_csv(a)))
    assert out["sender_name"].iloc[0].startswith("'=")


def test_analyze_api(settings_inproc=None):
    from fastapi.testclient import TestClient

    from finsight.serving.apps import make_gateway_app
    c = TestClient(make_gateway_app(Settings(artifacts_dir=ROOT)))
    sch = c.get("/analyze/schema").json()
    assert {r["column"] for r in sch["required"]} >= {"transaction_id", "amount_inr"} and "fraud_ring" in sch["attacks"]
    tpl = pd.read_csv(io.BytesIO(c.get("/analyze/template.csv").content))
    assert len(tpl) == 3 and "sender_account_id" in tpl.columns
    sample = c.get("/analyze/sample.csv", params={"rows": 400}).content
    up = c.post("/analyze/upload", files={"file": ("s.csv", sample, "text/csv")})
    assert up.status_code == 200 and up.json()["kpis"]["transactions"] == 400
    bad = c.post("/analyze/upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 422 and "Missing required" in bad.json()["detail"]
    sim = c.post("/analyze/simulate", json={"attacks": ["fraud_ring", "mule_chain"], "size": 8, "background": 500, "seed": 2}).json()
    aid = sim["analysis_id"]
    assert sim["groups"]["hubs"] and sim["evaluation"]["recall"] >= 0.9
    page = c.get(f"/analyses/{aid}/transactions", params={"tier": "CRITICAL,HIGH", "limit": 5}).json()
    assert page["total"] >= 10 and all(r["tier"] in ("HIGH", "CRITICAL") for r in page["items"])
    det = c.get(f"/analyses/{aid}/transactions/{page['items'][0]['transaction_id']}").json()
    assert det["tier"] in ("HIGH", "CRITICAL") and det["reasons"]
    dl = c.get(f"/analyses/{aid}/download.csv")
    assert dl.status_code == 200 and dl.headers["content-type"].startswith("text/csv")
    assert c.get("/analyses/nope").status_code == 404
    assert c.post("/analyze/simulate", json={"attacks": ["nope"]}).status_code == 422


def test_simulate_remote_mode_matches_inprocess():
    from fastapi.testclient import TestClient

    from finsight.serving import apps
    from finsight.serving.backends import InProcessBackend, RemoteBackend
    s = Settings(artifacts_dir=ROOT, internal_key="k")
    be = InProcessBackend(ROOT)
    clients = {"ml": TestClient(apps.make_ml_app(s, be.ml)), "mining": TestClient(apps.make_mining_app(s, be.mining)),
               "graph": TestClient(apps.make_graph_app(s, be.graph)), "sanctions": TestClient(apps.make_sanctions_app(s, be.sanctions))}
    remote = TestClient(apps.make_gateway_app(Settings(artifacts_dir=ROOT, internal_key="k", mode="remote"), backend=RemoteBackend(s, clients)))
    local = TestClient(apps.make_gateway_app(Settings(artifacts_dir=ROOT), backend=be))
    body = {"attacks": ["burst", "structuring"], "size": 8, "background": 600, "seed": 5}
    a, b = local.post("/analyze/simulate", json=body).json(), remote.post("/analyze/simulate", json=body).json()
    assert a["tiers"] == b["tiers"] and a["kpis"] == b["kpis"]
    assert [r["risk_score"] for r in a["top"]] == [r["risk_score"] for r in b["top"]]
