"""ml-service :8001 — classifier, explanations, anomaly detectors, clustering (internal).

Run locally:  PYTHONPATH=src uvicorn services.ml.main:app --port <port>
All logic lives in finsight.serving (apps.py builds the app); this module is only the process entry point.
"""
import services._bootstrap  # noqa: F401  (must run before numpy / pyarrow are imported)
from finsight.serving.apps import make_ml_app

app = make_ml_app()
