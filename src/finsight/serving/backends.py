"""How the gateway reaches the four domain services (D-023).

InProcessBackend   imports the cores and calls them directly: one process, one cold start (Render free default).
RemoteBackend      HTTP/JSON to ml / mining / graph / sanctions services (Docker Compose, Render "full" Blueprint),
                   sending X-Internal-Key and propagating X-Request-ID.
Both expose the same methods and return the same JSON-compatible dicts, so the gateway code is mode-agnostic.
Tests verify that the two modes give identical scores.
"""
from __future__ import annotations

import time
from pathlib import Path

from finsight.serving.common import jsonable
from finsight.serving.settings import Settings


class BackendError(RuntimeError):
    def __init__(self, service: str, status: int, detail):
        super().__init__(f"{service} returned {status}: {detail}")
        self.service, self.status, self.detail = service, status, detail


class InProcessBackend:
    mode = "inprocess"

    def __init__(self, root: Path):
        from finsight.serving.cores import GraphCore, MiningCore, SanctionsCore
        from finsight.serving.ml_core import MLCore

        t0 = time.perf_counter()
        self.ml, self.mining, self.graph, self.sanctions = MLCore(root), MiningCore(root), GraphCore(root), SanctionsCore(root)
        self.load_seconds = round(time.perf_counter() - t0, 2)

    # all methods mirror the remote HTTP endpoints 1:1 (and go through jsonable for identical output)
    def ml_score(self, rows, explain_top=6, rid=None):
        return jsonable(self.ml.score(rows, explain_top))

    def ml_anomaly(self, rows, rid=None):
        return jsonable(self.ml.anomaly(rows))

    def ml_compare(self, rid=None):
        return jsonable(self.ml.compare())

    def ml_clusters(self, rid=None):
        return jsonable(self.ml.clusters())

    def graph_accounts(self, params, rid=None):
        return jsonable(self.graph.accounts(**params))

    def mining_match(self, rows, rid=None):
        return jsonable(self.mining.match(rows))

    def mining_patterns(self, params, rid=None):
        return jsonable(self.mining.patterns(**params))

    def mining_benchmark(self, rid=None):
        return jsonable(self.mining.benchmark())

    def graph_risk(self, rows, rid=None):
        return jsonable(self.graph.risk(rows))

    def graph_network(self, account_id, limit=40, rid=None):
        return jsonable(self.graph.network(account_id, limit))

    def sanctions_screen(self, name, threshold=90.0, rid=None):
        return jsonable(self.sanctions.screen(name, threshold))

    def status(self, rid=None):
        return {n: {"status": "ok", "mode": "inprocess", **jsonable(getattr(self, n).health())}
                for n in ("ml", "mining", "graph", "sanctions")}


class RemoteBackend:
    mode = "remote"

    def __init__(self, settings: Settings, clients: dict | None = None):
        """`clients` (name → httpx.Client) lets tests inject FastAPI TestClients instead of real sockets."""
        import httpx

        headers = {"X-Internal-Key": settings.internal_key} if settings.internal_key else {}
        # short keep-alive expiry: pooled connections are re-opened (and DNS re-resolved) every few seconds, so
        # replicas added with `--scale` start receiving traffic; a platform load balancer does this on Render/K8s
        limits = httpx.Limits(max_connections=64, max_keepalive_connections=16, keepalive_expiry=2.0)
        self.clients = clients or {n: httpx.Client(base_url=u, timeout=settings.http_timeout, headers=headers, limits=limits)
                                   for n, u in settings.urls.items()}
        if clients:
            for c in clients.values():
                c.headers.update(headers)

    def _call(self, service, method, path, rid=None, **kw):
        h = {"X-Request-ID": rid} if rid else {}
        r = self.clients[service].request(method, path, headers=h, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except ValueError:
                detail = r.text[:200]
            raise BackendError(service, r.status_code, detail)
        return r.json()

    def ml_score(self, rows, explain_top=6, rid=None):
        return self._call("ml", "POST", "/score", rid, json={"rows": jsonable(rows), "explain_top": explain_top})["results"]

    def ml_anomaly(self, rows, rid=None):
        return self._call("ml", "POST", "/anomaly", rid, json={"rows": jsonable(rows)})["results"]

    def ml_compare(self, rid=None):
        return self._call("ml", "GET", "/models/compare", rid)

    def ml_clusters(self, rid=None):
        return self._call("ml", "GET", "/clusters", rid)

    def graph_accounts(self, params, rid=None):
        return self._call("graph", "GET", "/accounts", rid, params={k: v for k, v in params.items() if v is not None})

    def mining_match(self, rows, rid=None):
        return self._call("mining", "POST", "/match", rid, json={"rows": jsonable(rows)})["results"]

    def mining_patterns(self, params, rid=None):
        return self._call("mining", "GET", "/patterns", rid, params={k: v for k, v in params.items() if v is not None})

    def mining_benchmark(self, rid=None):
        return self._call("mining", "GET", "/patterns/benchmark", rid)

    def graph_risk(self, rows, rid=None):
        return self._call("graph", "POST", "/risk", rid, json={"rows": jsonable(rows)})["results"]

    def graph_network(self, account_id, limit=40, rid=None):
        return self._call("graph", "GET", f"/network/{account_id}", rid, params={"limit": limit})

    def sanctions_screen(self, name, threshold=90.0, rid=None):
        return self._call("sanctions", "GET", "/screen", rid, params={"name": name, "threshold": threshold})

    def status(self, rid=None):
        out = {}
        for n, c in self.clients.items():
            t0 = time.perf_counter()
            try:
                r = c.get("/health", headers={"X-Request-ID": rid} if rid else {})
                out[n] = {**r.json(), "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
            except Exception as e:                                   # noqa: BLE001 — report, don't fail
                out[n] = {"status": "unreachable", "error": type(e).__name__}
        return out
