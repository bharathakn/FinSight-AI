"""12-factor configuration for every service: everything comes from environment variables (Plan §8.2).

FINSIGHT_SERVICE_MODE     inprocess (one process, modules imported directly; default) | remote (HTTP microservices)
FINSIGHT_ARTIFACTS_DIR    root of the extracted artifact bundle (default: artifacts/bundle/current)
FINSIGHT_ARTIFACTS_URI    optional: bundle .tar.gz to fetch at startup if the dir is missing (https://, s3://, file path)
FINSIGHT_INTERNAL_KEY     shared secret; internal services require it in X-Internal-Key (unset = open, local dev only)
FINSIGHT_CORS_ORIGINS     comma-separated allow-list for the public gateway (e.g. https://finsight.vercel.app)
FINSIGHT_ML_URL, FINSIGHT_MINING_URL, FINSIGHT_GRAPH_URL, FINSIGHT_SANCTIONS_URL   remote-mode service URLs
FINSIGHT_HTTP_TIMEOUT     seconds for service-to-service calls (default 30; covers a Render cold start of a peer)
FINSIGHT_LOG_LEVEL        INFO by default
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from finsight import config

DEFAULT_URLS = {"ml": "http://localhost:8001", "mining": "http://localhost:8002",
                "graph": "http://localhost:8003", "sanctions": "http://localhost:8004"}


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _with_scheme(url: str) -> str:
    """Render's `fromService … property: host` yields a bare hostname; public services are HTTPS."""
    return url if "://" in url else f"https://{url}"


@dataclass(frozen=True)
class Settings:
    mode: str = "inprocess"
    artifacts_dir: Path = config.ARTIFACTS_DIR / "bundle" / "current"
    artifacts_uri: str | None = None
    internal_key: str | None = None
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173",       # vite dev
                                     "http://localhost:4173", "http://127.0.0.1:4173")       # vite preview
    urls: dict = field(default_factory=lambda: dict(DEFAULT_URLS))
    http_timeout: float = 30.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        mode = _env("FINSIGHT_SERVICE_MODE", "inprocess")
        if mode not in ("inprocess", "remote"):
            raise ValueError(f"FINSIGHT_SERVICE_MODE must be inprocess|remote, got {mode!r}")
        cors = _env("FINSIGHT_CORS_ORIGINS")
        return cls(
            mode=mode,
            artifacts_dir=Path(_env("FINSIGHT_ARTIFACTS_DIR", str(cls.artifacts_dir))),
            artifacts_uri=_env("FINSIGHT_ARTIFACTS_URI"),
            internal_key=_env("FINSIGHT_INTERNAL_KEY"),
            cors_origins=tuple(o.strip() for o in cors.split(",") if o.strip()) if cors else cls.cors_origins,
            urls={k: _with_scheme(_env(f"FINSIGHT_{k.upper()}_URL", v)) for k, v in DEFAULT_URLS.items()},
            http_timeout=float(_env("FINSIGHT_HTTP_TIMEOUT", "30")),
            log_level=_env("FINSIGHT_LOG_LEVEL", "INFO"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
