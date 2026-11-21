"""mining-service :8002 — association rules and live rule matching (internal).

Run locally:  PYTHONPATH=src uvicorn services.mining.main:app --port <port>
All logic lives in finsight.serving (apps.py builds the app); this module is only the process entry point.
"""
import services._bootstrap  # noqa: F401  (must run before numpy / pyarrow are imported)
from finsight.serving.apps import make_mining_app

app = make_mining_app()
