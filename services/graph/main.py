"""graph-service :8003 — network metrics, ego networks, graph_risk (internal).

Run locally:  PYTHONPATH=src uvicorn services.graph.main:app --port <port>
All logic lives in finsight.serving (apps.py builds the app); this module is only the process entry point.
"""
import services._bootstrap  # noqa: F401  (must run before numpy / pyarrow are imported)
from finsight.serving.apps import make_graph_app

app = make_graph_app()
