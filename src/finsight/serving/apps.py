"""FastAPI app factories for the five services (Plan §8.1, §8.3). `services/<name>/main.py` call these.

Public:    gateway  :8000  /score-transaction · /anomaly-score · /patterns · /patterns/benchmark · /cluster/{tx_id}
                          /network/{account_id} · /sanctions-check · /models/compare · /transactions[/{tx_id}]
                          /overview · /platform/status · /health · /metrics · /docs
Internal (X-Internal-Key):
           ml       :8001  POST /score · POST /anomaly · GET /models/compare
           mining   :8002  POST /match · GET /patterns · GET /patterns/benchmark
           graph    :8003  POST /risk · GET /network/{account_id}
           sanctions:8004  GET /screen
"""
from __future__ import annotations

import json
import logging
from functools import wraps
from pathlib import Path

from fastapi import File, HTTPException, Query, Request, UploadFile   # module level: annotations are resolved by name
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from finsight.serving.common import create_app, jsonable, ok
from finsight.serving.schemas import (AnomalyRequest, DemoRequest, NamesRequest, RowsRequest, ScoreRequest, ScoreResponse,
                                     SimulateRequest)
from finsight.serving.settings import Settings, get_settings
from finsight.storage import ensure_bundle

log = logging.getLogger("finsight.serving")


def _root(settings: Settings) -> Path:
    return ensure_bundle(settings.artifacts_dir, settings.artifacts_uri)


def _version(root: Path) -> str:
    return json.loads((root / "manifest.json").read_text())["version"]


def _errors(fn):
    """Map domain errors to HTTP: unknown id → 404, bad input → 422, downstream failure → 502."""
    from finsight.serving.backends import BackendError

    @wraps(fn)
    def inner(*a, **kw):
        try:
            return fn(*a, **kw)
        except KeyError as e:
            raise HTTPException(404, f"not found: {e.args[0] if e.args else e}") from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        except BackendError as e:
            code = e.status if e.status in (404, 422) else 502
            raise HTTPException(code, f"{e.service}: {e.detail}") from None
    return inner


def _rid(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


# ---------------------------------------------------------------- internal services
def make_ml_app(settings: Settings | None = None, core=None):
    from finsight.serving.ml_core import MLCore

    s = settings or get_settings()
    root = _root(s)
    core = core or MLCore(root)
    app = create_app("ml-service", s, internal=True, version=_version(root), health_extra=core.health,
                     description="Classifier (calibrated XGBoost), native TreeSHAP reasons, anomaly detectors, clustering.")

    @app.post("/score", tags=["ml"])
    @_errors
    def score(body: RowsRequest):
        return ok({"results": core.score(body.rows, body.explain_top)})

    @app.post("/anomaly", tags=["ml"])
    @_errors
    def anomaly(body: RowsRequest):
        return ok({"results": core.anomaly(body.rows)})

    @app.get("/models/compare", tags=["ml"])
    def compare():
        return ok(core.compare())

    @app.get("/clusters", tags=["ml"])
    def clusters():
        return ok(core.clusters())

    return app


def make_mining_app(settings: Settings | None = None, core=None):
    from finsight.serving.cores import MiningCore

    s = settings or get_settings()
    root = _root(s)
    core = core or MiningCore(root)
    app = create_app("mining-service", s, internal=True, version=_version(root), health_extra=core.health,
                     description="Association rules (Apriori / FP-Growth results) and live rule matching.")

    @app.post("/match", tags=["mining"])
    @_errors
    def match(body: RowsRequest):
        return ok({"results": core.match(body.rows)})

    @app.get("/patterns", tags=["mining"])
    def patterns(consequent: str | None = None, min_confidence: float = 0.0, min_lift: float = 0.0,
                 network_only: bool = False, limit: int = Query(50, ge=1, le=500)):
        return ok(core.patterns(consequent, min_confidence, min_lift, network_only, limit))

    @app.get("/patterns/benchmark", tags=["mining"])
    def benchmark():
        return ok(core.benchmark())

    return app


def make_graph_app(settings: Settings | None = None, core=None):
    from finsight.serving.cores import GraphCore

    s = settings or get_settings()
    root = _root(s)
    core = core or GraphCore(root)
    app = create_app("graph-service", s, internal=True, version=_version(root), health_extra=core.health,
                     description="Network metrics, ego networks and the graph_risk component.")

    @app.post("/risk", tags=["graph"])
    @_errors
    def risk(body: RowsRequest):
        return ok({"results": core.risk(body.rows)})

    @app.get("/network/{account_id}", tags=["graph"])
    @_errors
    def network(account_id: str, limit: int = Query(40, ge=1, le=200)):
        return ok(core.network(account_id, limit))

    @app.get("/accounts", tags=["graph"])
    @_errors
    def accounts(sort: str = "pagerank", limit: int = Query(25, ge=1, le=500), query: str | None = None):
        return ok(core.accounts(sort, limit, query))

    return app


def make_sanctions_app(settings: Settings | None = None, core=None):
    from finsight.serving.cores import SanctionsCore

    s = settings or get_settings()
    root = _root(s)
    core = core or SanctionsCore(root)
    app = create_app("sanctions-service", s, internal=True, version=_version(root), health_extra=core.health,
                     description="Fuzzy OFAC + UN sanctions screening (RapidFuzz).")

    @app.get("/screen", tags=["sanctions"])
    def screen(name: str = Query(..., min_length=1, max_length=200), threshold: float = Query(90.0, ge=50, le=100)):
        return ok(core.screen(name, threshold))

    @app.post("/screen-batch", tags=["sanctions"])
    def screen_batch(body: NamesRequest):
        return ok({"results": core.screen_batch(body.names, body.threshold)})

    return app


# ---------------------------------------------------------------- public gateway
def make_gateway_app(settings: Settings | None = None, backend=None):
    from finsight.serving.backends import InProcessBackend, RemoteBackend
    from finsight.serving.gateway import Gateway

    s = settings or get_settings()
    root = _root(s)
    if backend is None:
        backend = InProcessBackend(root) if s.mode == "inprocess" else RemoteBackend(s)
    gw = Gateway(root, backend)
    app = create_app("api-gateway", s, internal=False, version=gw.version,
                     health_extra=lambda: {"backend": backend.mode, "transactions": len(gw.store),
                                           "model_version": gw.version},
                     description="FinSight AI public REST API: 0–100 transaction risk scoring with explanations, "
                                 "association rules, network analytics, sanctions screening and model results.")
    app.add_middleware(GZipMiddleware, minimum_size=2048)          # chart payloads (e.g. /clusters) shrink ~5×
    app.add_middleware(CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["GET", "POST", "OPTIONS"],
                       allow_headers=["*"], expose_headers=["X-Request-ID", "X-Served-By", "Server-Timing"])
    app.state.gateway = gw
    reports = root / "reports"

    def _report(name):
        f = reports / name
        return json.loads(f.read_text()) if f.exists() else None

    @app.post("/score-transaction", response_model=ScoreResponse, tags=["scoring"])
    @_errors
    def score_transaction(body: ScoreRequest, request: Request):
        """Risk score 0–100 + tier, component scores, top reasons, fired rules, network flags, cluster, sanctions."""
        return ok(gw.score(body, _rid(request)))      # already JSON-safe; skip re-validation (schema stays in OpenAPI)

    @app.post("/anomaly-score", tags=["scoring"])
    @_errors
    def anomaly_score(body: AnomalyRequest, request: Request):
        """Isolation Forest, One-class SVM and autoencoder scores with percentiles vs legitimate traffic."""
        return ok(gw.anomaly(body, _rid(request)))

    @app.get("/cluster/{tx_id}", tags=["ml"])
    @_errors
    def cluster(tx_id: str, request: Request):
        row, _ = gw.store.feature_row(tx_id)
        (r,) = backend.ml_score([row], 0, rid=_rid(request))
        return ok({"transaction_id": tx_id, "cluster": r["cluster"], "anomaly": r["anomaly"]})

    @app.get("/models/compare", tags=["ml"])
    @_errors
    def models_compare(request: Request):
        return ok(backend.ml_compare(rid=_rid(request)))

    @app.get("/patterns", tags=["mining"])
    @_errors
    def patterns(request: Request, consequent: str | None = None, min_confidence: float = 0.0, min_lift: float = 0.0,
                 network_only: bool = False, limit: int = Query(50, ge=1, le=500)):
        return ok(backend.mining_patterns({"consequent": consequent, "min_confidence": min_confidence,
                                           "min_lift": min_lift, "network_only": network_only, "limit": limit},
                                          rid=_rid(request)))

    @app.get("/patterns/benchmark", tags=["mining"])
    @_errors
    def patterns_benchmark(request: Request):
        return ok(backend.mining_benchmark(rid=_rid(request)))

    @app.get("/network/{account_id}", tags=["graph"])
    @_errors
    def network(account_id: str, request: Request, limit: int = Query(40, ge=1, le=200)):
        return ok(backend.graph_network(account_id, limit, rid=_rid(request)))

    @app.get("/sanctions-check", tags=["sanctions"])
    @_errors
    def sanctions_check(request: Request, name: str = Query(..., min_length=1, max_length=200),
                        threshold: float = Query(90.0, ge=50, le=100)):
        return ok(backend.sanctions_screen(name, threshold, rid=_rid(request)))

    @app.get("/clusters", tags=["ml"])
    @_errors
    def clusters(request: Request):
        """K-Means sweep + profiles, 2-D PCA embedding sample, anomaly detector comparison (Phase 6)."""
        return ok(backend.ml_clusters(rid=_rid(request)))

    @app.get("/accounts", tags=["graph"])
    @_errors
    def accounts(request: Request, sort: str = "pagerank", limit: int = Query(25, ge=1, le=500), query: str | None = None):
        return ok(backend.graph_accounts({"sort": sort, "limit": limit, "query": query}, rid=_rid(request)))

    @app.get("/data-quality", tags=["data"])
    def data_quality():
        """Phase 1 data-quality report, Phase 2 EDA summary (RQ1 ranking, flags) and the sanctions threshold sweep."""
        eda = _report("eda_summary.json") or {}
        eda.pop("figures", None)
        items = reports / "mining_items_v1_1.csv"
        return ok({"quality": _report("data_quality_india_v1_1.json"), "eda": eda, "sanctions": _report("sanctions_summary.json"),
                   "items": __import__("pandas").read_csv(items).to_dict(orient="records") if items.exists() else []})

    # ------------------------------------------------------------ live analysis (Phase 9b)
    import threading

    from fastapi.responses import Response

    from finsight.serving import batch as B
    from finsight.serving import simulate as SIM

    ref = B.Reference(root)
    analyses = B.AnalysisStore()
    slots = threading.BoundedSemaphore(2)            # ≤ 2 concurrent analyses: protects a 512 MB instance
    demo_cache: dict[int, str] = {}                  # sample size → pinned analysis id (the demo data never changes)

    def _run(df, source, request):
        if not slots.acquire(blocking=False):
            raise HTTPException(429, "Two analyses are already running; please retry in a few seconds.")
        try:
            return ok(B.run(df, ref, backend, gw.scorer, analyses, source, rid=_rid(request)))
        finally:
            slots.release()

    def _csv(data: bytes, name: str):
        return Response(data, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/analyze/schema", tags=["analyze"])
    def analyze_schema():
        """Columns an upload needs (required / optional with defaults / labels) and the limits."""
        return ok(B.schema() | {"attacks": SIM.ATTACKS})

    @app.get("/analyze/template.csv", tags=["analyze"])
    def analyze_template():
        return _csv(B.template_csv(ref), "finsight_template.csv")

    @app.get("/analyze/sample.csv", tags=["analyze"])
    def analyze_sample(rows: int = Query(1000, ge=100, le=10000)):
        """A labelled sample of the demo dataset in upload format (try uploading it)."""
        return _csv(B.sample_csv(ref, rows), f"finsight_sample_{rows}.csv")

    @app.post("/analyze/upload", tags=["analyze"])
    @_errors
    def analyze_upload(request: Request, file: UploadFile = File(...)):
        """Analyse an uploaded CSV of transactions (format: /analyze/schema; ≤ 10 MB, ≤ 25,000 rows)."""
        data = file.file.read(B.MAX_BYTES + 1)
        df = B.read_csv(data)
        return _run(df, {"type": "upload", "filename": file.filename, "bytes": len(data)}, request)

    @app.post("/analyze/demo", tags=["analyze"])
    @_errors
    def analyze_demo(body: DemoRequest, request: Request):
        """Analyse the built-in demo dataset (all 10,000 transactions, or a stratified sample).
        The result is deterministic, so it is computed once per sample size and then served from memory."""
        key = body.sample or 0
        if key in demo_cache:
            return ok(analyses.get(demo_cache[key]).summary | {"cached": True})
        resp = _run(ref.demo(body.sample), {"type": "demo", "rows": body.sample or len(ref.history)}, request)
        aid = json.loads(resp.body)["analysis_id"]
        analyses.pin(analyses.get(aid))
        demo_cache[key] = aid
        return resp

    @app.post("/analyze/simulate", tags=["analyze"])
    @_errors
    def analyze_simulate(body: SimulateRequest, request: Request):
        """Inject the chosen attacks into a normal day of payments and analyse the result."""
        df, meta = SIM.simulate(ref, body.attacks, body.size, body.camouflage, body.background, body.seed)
        return _run(df, {"type": "simulate", **meta}, request)

    @app.get("/analyses/{aid}", tags=["analyze"])
    @_errors
    def analysis(aid: str):
        return ok(analyses.get(aid).summary)

    @app.get("/analyses/{aid}/transactions", tags=["analyze"])
    @_errors
    def analysis_rows(aid: str, tier: str | None = None, q: str | None = None, sort: str = "risk_score", desc: bool = True,
                      limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        return ok(B.rows_page(analyses.get(aid), tier, q, sort, desc, limit, offset))

    @app.get("/analyses/{aid}/transactions/{tx_id}", tags=["analyze"])
    @_errors
    def analysis_row(aid: str, tx_id: str, request: Request):
        """Full explanation of one analysed transaction (same output as /score-transaction)."""
        row = B.feature_row(analyses.get(aid), tx_id)
        return ok(gw.score(ScoreRequest(features=row), _rid(request)))

    @app.get("/analyses/{aid}/download.csv", tags=["analyze"])
    @_errors
    def analysis_download(aid: str):
        return _csv(B.to_csv(analyses.get(aid)), f"finsight_analysis_{aid}.csv")

    @app.get("/platform/benchmarks", tags=["platform"])
    def platform_benchmarks():
        """Recorded platform measurements (Phase 8 local; Phase 10 cloud), if bundled."""
        return ok(_report("platform_summary.json") or {})

    @app.get("/transactions/{tx_id}", tags=["data"])
    @_errors
    def transaction(tx_id: str):
        return ok(gw.store.get(tx_id))

    @app.get("/transactions", tags=["data"])
    @_errors
    def transactions(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), label: str | None = None,
                     scenario: str | None = None, channel: str | None = None, account_id: str | None = None,
                     min_amount: float | None = None, sort: str = "timestamp", desc: bool = True):
        return ok(gw.store.query(limit, offset, label, scenario, channel, account_id, min_amount, sort, desc))

    @app.get("/overview", tags=["data"])
    def overview():
        return ok(gw.store.overview() | {"model": {"version": gw.version, "weights": gw.scorer.weights,
                                                    "cutoffs": gw.scorer.cutoffs}})

    @app.get("/platform/status", tags=["platform"])
    def platform_status(request: Request):
        """Gateway metrics + health of every downstream service (dashboard Platform page)."""
        return ok({"mode": backend.mode, "gateway": app.state.metrics.snapshot(), "services": backend.status(_rid(request)),
                   "bundle": {k: gw.manifest[k] for k in ("version", "created_utc", "total_mb", "libraries")}})

    return app


__all__ = ["make_gateway_app", "make_ml_app", "make_mining_app", "make_graph_app", "make_sanctions_app", "jsonable"]
