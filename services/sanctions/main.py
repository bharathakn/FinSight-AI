"""sanctions-service :8004 — fuzzy OFAC + UN screening (internal).

Run locally:  PYTHONPATH=src uvicorn services.sanctions.main:app --port <port>
All logic lives in finsight.serving (apps.py builds the app); this module is only the process entry point.
"""
import services._bootstrap  # noqa: F401  (must run before numpy / pyarrow are imported)
from finsight.serving.apps import make_sanctions_app

app = make_sanctions_app()
