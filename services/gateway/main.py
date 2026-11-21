"""api-gateway :8000 — public REST API; inprocess or remote mode (FINSIGHT_SERVICE_MODE).

Run locally:  PYTHONPATH=src uvicorn services.gateway.main:app --port <port>
All logic lives in finsight.serving (apps.py builds the app); this module is only the process entry point.
"""
import services._bootstrap  # noqa: F401  (must run before numpy / pyarrow are imported)
from finsight.serving.apps import make_gateway_app

app = make_gateway_app()
