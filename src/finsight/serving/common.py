"""Shared FastAPI plumbing for all five services (observability + security, Plan §8.2).

* Request IDs   X-Request-ID is accepted or generated, returned on every response and forwarded to
                downstream services, so one user request can be followed across services in the logs.
* JSON logs     one structured line per request (service, method, path, status, ms, request_id);
                Render's log stream and `docker compose logs` show them as-is.
* Metrics       per-route request count, error count and latency percentiles over a sliding window,
                plus process RSS and uptime, served at GET /metrics (JSON).
* Internal key  services other than the gateway reject calls without X-Internal-Key = FINSIGHT_INTERNAL_KEY
                (Render free services can't use private networking, so this replaces network isolation).
* /health       liveness + what is loaded; always open (used by Render health checks and the SPA's wake-up poll).
"""
from __future__ import annotations

import json
import logging
import math
import os
import socket
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import date, datetime, timezone

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from finsight.serving.settings import Settings

START_TIME = time.time()
HOST = socket.gethostname()


# ---------------------------------------------------------------- JSON helpers

def jsonable(obj):
    """numpy / pandas / NaN → plain JSON types (NaN and ±inf → None, since JSON has no NaN)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):                       # pandas Timestamp
        return obj.isoformat()
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    try:
        import pandas as pd

        if pd.isna(obj):
            return None
    except (TypeError, ValueError, ImportError):
        pass
    return str(obj)


# ---------------------------------------------------------------- logging

class JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
             "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        d.update(getattr(record, "extra_fields", {}))
        return json.dumps(d, default=str)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root.handlers[:] = [h]
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = True      # replaced by our structured access log
    logging.getLogger("httpx").setLevel(logging.WARNING)      # one line per service call is noise


# ---------------------------------------------------------------- metrics

def rss_mb() -> float:
    """Resident memory of this process (Linux /proc; falls back to peak RSS elsewhere)."""
    try:
        with open("/proc/self/statm") as f:
            return round(int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6, 1)
    except OSError:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


class Metrics:
    def __init__(self, window: int = 2000):
        self.count = defaultdict(int)
        self.errors = defaultdict(int)
        self.lat = defaultdict(lambda: deque(maxlen=window))

    def observe(self, route: str, ms: float, status: int) -> None:
        self.count[route] += 1
        if status >= 500:
            self.errors[route] += 1
        self.lat[route].append(ms)

    def snapshot(self) -> dict:
        routes = {}
        for r, n in self.count.items():
            v = np.array(self.lat[r]) if self.lat[r] else np.array([0.0])
            routes[r] = {"requests": n, "errors_5xx": self.errors[r],
                         "p50_ms": round(float(np.percentile(v, 50)), 2), "p95_ms": round(float(np.percentile(v, 95)), 2),
                         "p99_ms": round(float(np.percentile(v, 99)), 2), "mean_ms": round(float(v.mean()), 2),
                         "window": int(len(self.lat[r]))}
        return {"routes": routes, "rss_mb": rss_mb(), "uptime_s": round(time.time() - START_TIME, 1), "host": HOST}


# ---------------------------------------------------------------- app factory

OPEN_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/docs/oauth2-redirect"}


def create_app(service: str, settings: Settings, *, internal: bool, version: str = "dev",
               description: str = "", health_extra=None) -> FastAPI:
    """FastAPI app with request-ID, JSON access log, metrics, optional internal-key guard, /health, /metrics."""
    setup_logging(settings.log_level)
    log = logging.getLogger(f"finsight.{service}")
    app = FastAPI(title=f"FinSight AI · {service}", version=version, description=description)
    metrics = Metrics()
    app.state.metrics = metrics
    app.state.service = service

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        t0 = time.perf_counter()
        path = request.url.path
        if internal and settings.internal_key and path not in OPEN_PATHS \
                and request.headers.get("x-internal-key") != settings.internal_key:
            resp = JSONResponse({"detail": "missing or invalid X-Internal-Key"}, status_code=401)
        else:
            try:
                resp = await call_next(request)
            except Exception:                               # noqa: BLE001 — logged + 500 with the request id
                log.exception("unhandled error", extra={"extra_fields": {"request_id": rid, "path": path}})
                resp = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
        ms = (time.perf_counter() - t0) * 1000
        route = request.scope.get("route")
        key = f"{request.method} {getattr(route, 'path', path)}"
        metrics.observe(key, ms, resp.status_code)
        resp.headers["X-Request-ID"] = rid
        resp.headers["X-Served-By"] = f"{service}@{HOST}"
        resp.headers["Server-Timing"] = f"app;dur={ms:.1f}"
        log.info("request", extra={"extra_fields": {"service": service, "method": request.method, "path": path,
                                                     "status": resp.status_code, "ms": round(ms, 2), "request_id": rid}})
        return resp

    @app.get("/health", tags=["platform"])
    def health():
        body = {"status": "ok", "service": service, "version": version, "mode": settings.mode, "host": HOST,
                "uptime_s": round(time.time() - START_TIME, 1), "rss_mb": rss_mb()}
        if health_extra:
            body.update(jsonable(health_extra()))
        return body

    @app.get("/metrics", tags=["platform"])
    def get_metrics():
        return {"service": service, "mode": settings.mode, **metrics.snapshot()}

    return app


def ok(payload) -> JSONResponse:
    """JSONResponse that tolerates numpy types and NaN."""
    return JSONResponse(jsonable(payload))
