"""Phase 8 serving tests: fast paths ≡ training objects, every gateway endpoint, error mapping, the internal-key
guard, and inprocess ≡ remote mode (the gateway talks HTTP to the four services through TestClients)."""
import json

import numpy as np
import pytest

from finsight.serving.settings import Settings

ROOT = Settings().artifacts_dir
pytestmark = pytest.mark.skipif(not (ROOT / "manifest.json").exists(), reason="run `make bundle`")
TX = "TXN_IND_202608_001768"            # S8 mule pass-through fraud (Phase 7 example)


@pytest.fixture(scope="module")
def settings():
    return Settings(artifacts_dir=ROOT, internal_key="test-key")


@pytest.fixture(scope="module")
def inproc(settings):
    from fastapi.testclient import TestClient

    from finsight.serving.apps import make_gateway_app

    return TestClient(make_gateway_app(settings))


@pytest.fixture(scope="module")
def remote(settings, inproc):
    """Gateway in remote mode; each domain service is a real FastAPI app behind a TestClient."""
    from fastapi.testclient import TestClient

    from finsight.serving import apps
    from finsight.serving.backends import RemoteBackend

    be = inproc.app.state.gateway.backend               # reuse the loaded cores (memory), wrap them in apps
    clients = {"ml": TestClient(apps.make_ml_app(settings, be.ml)), "mining": TestClient(apps.make_mining_app(settings, be.mining)),
               "graph": TestClient(apps.make_graph_app(settings, be.graph)),
               "sanctions": TestClient(apps.make_sanctions_app(settings, be.sanctions))}
    gw = apps.make_gateway_app(Settings(artifacts_dir=ROOT, internal_key="test-key", mode="remote"),
                               backend=RemoteBackend(settings, clients))
    return TestClient(gw), clients


# ---------------------------------------------------------------- fast paths (D-034)
def test_fast_paths_equal_training_objects(inproc):
    from finsight.mining.discretiser import india_items
    from finsight.serving.store import FEATURES

    gw = inproc.app.state.gateway
    ml = gw.backend.ml
    df = gw.store.df.iloc[::7].reset_index(drop=True)
    rows = [gw.store.feature_row(t)[0] for t in df["transaction_id"]]
    for fp, ct in ((ml.fp_prod, ml.explainer.named_steps["prep"]), (ml.fp_an, ml.an_prep), (ml.fp_km, ml.km_prep)):
        ref = np.asarray(ct.transform(df[FEATURES]), float)
        assert np.allclose(fp.transform(rows), ref, atol=1e-12, equal_nan=True)
    p_fast, _ = ml.predict(rows)
    assert np.abs(p_fast - ml.model.predict_proba(df[FEATURES])[:, 1]).max() == 0.0
    ref_items = india_items(df, with_labels=False)
    from finsight.serving.fastpath import items_for_record
    for i in range(0, len(rows), 5):
        assert items_for_record(rows[i]) == set(ref_items.columns[ref_items.iloc[i].to_numpy()])


def test_served_probabilities_reproduce_phase7_test_metrics(inproc):
    from finsight.ml.evaluator import scores
    from finsight.ml.splits import stratified_split

    gw = inproc.app.state.gateway
    df = gw.store.df.reset_index(drop=True)
    sp = stratified_split(df)
    rows = [gw.store.feature_row(t)[0] for t in df["transaction_id"].iloc[sp.test]]
    p, _ = gw.backend.ml.predict(rows)
    metrics = json.loads((ROOT / "models/final/metrics.json").read_text())
    s = scores(df["is_fraud"].iloc[sp.test], p, metrics["decision_threshold"])
    assert s["pr_auc"] == pytest.approx(metrics["test"]["test_pr_auc"], abs=1e-9)
    assert s["f1"] == pytest.approx(metrics["test"]["test_f1"], abs=1e-9)


# ---------------------------------------------------------------- gateway endpoints
def test_health_and_metrics(inproc):
    h = inproc.get("/health").json()
    assert h["status"] == "ok" and h["mode"] == "inprocess" and h["transactions"] == 10000
    inproc.get("/health")
    m = inproc.get("/metrics").json()
    assert m["routes"]["GET /health"]["requests"] >= 1 and m["rss_mb"] > 0


def test_score_transaction_payload(inproc):
    r = inproc.post("/score-transaction", json={"transaction_id": TX})
    assert r.status_code == 200 and r.headers["X-Request-ID"]
    d = r.json()
    assert 0 <= d["risk_score"] <= 100 and d["tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert d["tier"] == "CRITICAL" and d["network"]["flags"]["PASS_THROUGH"] is True
    assert d["reasons"] and {"feature", "contribution", "direction"} <= set(d["reasons"][0])
    assert d["rules"]["n_fired_fraud"] >= 1 and d["cluster"]["cluster"] in range(4)
    assert set(d["components"]) == {"model_probability", "anomaly_pct", "graph_risk"}


def test_what_if_overrides_recompute_derived_features(inproc):
    base = inproc.post("/score-transaction", json={"transaction_id": TX}).json()
    small = inproc.post("/score-transaction", json={"transaction_id": TX, "overrides": {"amount_inr": 250}}).json()
    assert small["input"]["derived"] == ["log_amount", "amount_ratio_vs_30d_avg"]
    assert small["context"]["amount_inr"] == 250 and small["risk_score"] < base["risk_score"]


def test_error_mapping(inproc):
    assert inproc.post("/score-transaction", json={"transaction_id": "nope"}).status_code == 404
    assert inproc.post("/score-transaction", json={}).status_code == 422                    # neither source
    assert inproc.post("/score-transaction", json={"transaction_id": TX, "overrides": {"is_fraud": 0}}).status_code == 422
    assert inproc.post("/score-transaction", json={"features": {"log_amount": 9}}).status_code == 422
    assert inproc.get("/network/ACC_DOES_NOT_EXIST").status_code == 404


def test_features_mode_matches_store_mode(inproc):
    gw = inproc.app.state.gateway
    row = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in gw.store.get(TX).items()
           if k != "timestamp"}
    a = inproc.post("/score-transaction", json={"transaction_id": TX}).json()
    b = inproc.post("/score-transaction", json={"features": row}).json()
    assert a["risk_score"] == pytest.approx(b["risk_score"]) and a["components"] == b["components"]


def test_read_endpoints(inproc):
    assert inproc.post("/anomaly-score", json={"transaction_id": TX}).json()["detectors"]["autoencoder"]["above_p95"]
    assert inproc.get(f"/cluster/{TX}").json()["cluster"]["size"] > 0
    pats = inproc.get("/patterns", params={"consequent": "FRAUDULENT", "limit": 3}).json()
    assert len(pats["items"]) == 3 and all(p["consequent"] == "FRAUDULENT" for p in pats["items"])
    assert inproc.get("/patterns/benchmark").json()["apriori_equals_fpgrowth"]
    net = inproc.get("/network/ACC_IND_000013").json()
    assert net["ego"]["nodes"][0]["role"] == "focus" and net["activity"]["received"] > 0
    assert inproc.get("/models/compare").json()["production"]["production_model"] == "xgboost_calibrated"
    page = inproc.get("/transactions", params={"label": "BORDERLINE", "limit": 5}).json()
    assert page["total"] == 452 and len(page["items"]) == 5
    assert inproc.get(f"/transactions/{TX}").json()["transaction_id"] == TX
    assert inproc.get("/overview").json()["transactions"] == 10000


def test_sanctions_screening_finds_listed_name(inproc):
    import pandas as pd

    name = pd.read_parquet(ROOT / "sanctions/entities.parquet").query("source == 'UN' and name_kind == 'primary'").iloc[0]["name"]
    d = inproc.get("/sanctions-check", params={"name": name}).json()
    assert d["hit"] and d["best_score"] >= 90
    assert not inproc.get("/sanctions-check", params={"name": "Ayaan Sharma"}).json()["hit"]


def test_cors_allow_list(inproc):
    ok = inproc.options("/score-transaction", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    bad = inproc.options("/score-transaction", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-origin" not in bad.headers


# ---------------------------------------------------------------- remote mode
def test_internal_services_require_key(remote):
    _, clients = remote
    from fastapi.testclient import TestClient

    raw = TestClient(clients["sanctions"].app)                 # no X-Internal-Key header
    assert raw.get("/screen", params={"name": "x y z"}).status_code == 401
    assert raw.get("/health").status_code == 200                 # health stays open for platform probes


def test_remote_mode_equals_inprocess(remote, inproc):
    gw, _ = remote
    for tx in (TX, "TXN_IND_202608_006264", "TXN_IND_202608_000001"):
        a = inproc.post("/score-transaction", json={"transaction_id": tx}).json()
        b = gw.post("/score-transaction", json={"transaction_id": tx}).json()
        assert b["mode"] == "remote"
        for k in ("risk_score", "tier", "components", "reasons", "rules", "network", "cluster", "sanctions"):
            assert a[k] == b[k], k
    st = gw.get("/platform/status").json()
    assert st["mode"] == "remote" and all(s["status"] == "ok" for s in st["services"].values())


# ---------------------------------------------------------------- dashboard endpoints (Phase 9)
def test_dashboard_endpoints(inproc):
    cl = inproc.get("/clusters")
    assert cl.status_code == 200
    d = cl.json()
    pts = d["embedding"]["points"]
    assert len(pts) == 4000 and sum(p["risk_label"] != "LEGITIMATE" for p in pts) == 1800      # all fraud + borderline kept
    assert {c["cluster"] for c in d["profiles"]} == {0, 1, 2, 3} and len(d["sweep"]) == 14
    big = inproc.get("/clusters", headers={"Accept-Encoding": "gzip"})
    assert big.headers.get("content-encoding") == "gzip"
    acc = inproc.get("/accounts", params={"sort": "pagerank", "limit": 5}).json()
    assert len(acc["items"]) == 5 and acc["items"][0]["pagerank"] >= acc["items"][-1]["pagerank"]
    assert inproc.get("/accounts", params={"sort": "nope"}).status_code == 422
    dq = inproc.get("/data-quality").json()
    assert dq["quality"]["tables"] and dq["eda"]["rq1_top10"] and dq["sanctions"]["entities"] > 20000 and dq["items"]
    cmp = inproc.get("/models/compare").json()
    assert cmp["shap_importance"] and cmp["learning_curves"] and cmp["per_scenario"]
    assert isinstance(inproc.get("/platform/benchmarks").json(), dict)


def test_dashboard_endpoints_remote_parity(remote, inproc):
    gw, _ = remote
    assert gw.get("/accounts", params={"limit": 10}).json() == inproc.get("/accounts", params={"limit": 10}).json()
    assert gw.get("/clusters").json()["profiles"] == inproc.get("/clusters").json()["profiles"]
