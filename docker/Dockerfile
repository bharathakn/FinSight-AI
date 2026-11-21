# syntax=docker/dockerfile:1.7
# FinSight AI — one multi-stage Dockerfile for all five services (Plan §8.2, D-024, D-035).
#
#   SERVICE   gateway | ml | mining | graph | sanctions      (which services/<name>/main.py to run)
#   REQS      base | ml | full                                (dependency set; slim images for non-ML services)
#   ARTIFACTS_URL  optional https URL of finsight-artifacts.tar.gz (GitHub Release); if empty, the local
#                  docker/bundle/finsight-artifacts.tar.gz (from `make bundle`) is baked in instead.
#
#   docker build -f docker/Dockerfile --build-arg SERVICE=gateway --build-arg REQS=full -t finsight-gateway .
#
# Stages: deps (venv with pinned wheels) → bundle (download / copy + extract artifacts) → runtime
# (python:slim, non-root user, no compilers, no pip cache, artifacts read-only inside the image).

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------- 1. dependencies
FROM python:${PYTHON_VERSION}-slim AS deps
ARG REQS=full
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY docker/requirements/ /tmp/req/
RUN pip install -r /tmp/req/${REQS}.txt \
 && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
 && find /opt/venv -path "*/tests" -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------- 2. artifact bundle ("model registry")
FROM python:${PYTHON_VERSION}-slim AS bundle
ARG ARTIFACTS_URL=""
ARG GITHUB_TOKEN=""
COPY docker/bundle/ /tmp/bundle/
RUN python - <<'PY'
import os, pathlib, tarfile, urllib.request
url, token = os.environ.get("ARTIFACTS_URL", ""), os.environ.get("GITHUB_TOKEN", "")
tb = pathlib.Path("/tmp/bundle/finsight-artifacts.tar.gz")
if url:
    req = urllib.request.Request(url, headers={"Accept": "application/octet-stream"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=300) as r:
        tb.write_bytes(r.read())
    print("downloaded", url, tb.stat().st_size, "bytes")
elif not tb.exists():
    raise SystemExit("No artifact bundle: run `make bundle` or pass --build-arg ARTIFACTS_URL=<release asset URL>")
dest = pathlib.Path("/opt/finsight/artifacts")
dest.mkdir(parents=True)
with tarfile.open(tb, "r:gz") as t:
    t.extractall(dest, filter="data")
assert (dest / "manifest.json").exists(), "bundle has no manifest.json"
PY

# ---------------------------------------------------------------- 3. runtime
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG SERVICE=gateway
LABEL org.opencontainers.image.title="finsight-${SERVICE}" \
      org.opencontainers.image.description="FinSight AI ${SERVICE} service" \
      org.opencontainers.image.source="https://github.com/bharathakn/FinSight-AI"
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src:/app \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    FINSIGHT_SERVICE=${SERVICE} \
    FINSIGHT_ARTIFACTS_DIR=/opt/finsight/artifacts \
    ARROW_DEFAULT_MEMORY_POOL=system MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 \
    PORT=8000
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin finsight
COPY --from=deps /opt/venv /opt/venv
COPY --from=bundle /opt/finsight/artifacts /opt/finsight/artifacts
COPY src/finsight /app/src/finsight
COPY services /app/services
WORKDIR /app
USER finsight
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=30s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'), timeout=3)" || exit 1
# $PORT is injected by Render / Cloud Run. WEB_CONCURRENCY = uvicorn worker processes (default 1: scale out with
# replicas; the remote-mode gateway uses 2 because it is CPU-light per process but serves every request).
ENV WEB_CONCURRENCY=1
CMD ["sh", "-c", "exec uvicorn services.${FINSIGHT_SERVICE}.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --no-access-log --proxy-headers --forwarded-allow-ips='*'"]
