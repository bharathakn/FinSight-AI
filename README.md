# FinSight AI

**Cloud-based financial intelligence, pattern mining and machine learning on a synthetic Indian payments ecosystem (UPI / NEFT / RTGS).**
One project submitted across three courses: **Data Mining & Wrangling**, **Machine Learning** and **Cloud Computing & Technology**.

| Layer | What it does |
|---|---|
| 📊 DMW | KDD pipeline, 7-file + XML wrangling, data-quality and leakage audit, graph-structure injection (v1.1), Apriori vs FP-Growth |
| 🤖 ML | 6 classifiers × 3 feature sets (transaction / behavioural / network), clustering, IF / OC-SVM / Autoencoder, tuning, calibrated stacking, SHAP |
| ☁️ CCT | FastAPI microservices + Streamlit dashboard, Docker Compose, GCP Cloud Run (Services + Job), GCS |

> ⚠️ All India data is **synthetic**. Results describe the generator, not real-world fraud.

## Quickstart
```bash
uv sync --all-extras      # see docs/SETUP.md for full install steps
make test
make pipeline             # data → features → mining → supervised → unsupervised → final model
make bundle               # package the trained artifacts for serving
make serve                # API on http://localhost:8000/docs (single process)
make up                   # or: 5 Docker microservices (gateway :8000 + ml / mining / graph / sanctions)
make dashboard            # React dashboard on http://localhost:5173 (9 pages)
```
Score a transaction: `curl -X POST localhost:8000/score-transaction -H 'content-type: application/json' -d '{"transaction_id": "TXN_IND_202608_001768"}'`

## Documentation
| Doc | Contents |
|---|---|
| [FinSight_AI_Project_Plan.md](FinSight_AI_Project_Plan.md) | Design spec: architecture, experiments, research questions, syllabus mapping |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Phase-by-phase tracker with acceptance criteria |
| [docs/DATA.md](docs/DATA.md) | Data sources, lineage, column roles, leakage list |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log with evidence |
| [docs/SETUP.md](docs/SETUP.md) | Installation + manual setup tasks |
| [CLAUDE.md](CLAUDE.md) | Working rules for AI coding agents |

## Results so far (phases 0–9 of 10)
- **Supervised (Phase 5):** XGBoost on all features, test PR-AUC **0.875**, F1 0.801. Network features significantly help 5 of 6 models on v1.1 and 0 of 6 on the v1.0 control.
- **Unsupervised (Phase 6):** K-Means finds 1 normal and 3 interpretable risk clusters (ARI 0.47). Detectors trained only on legitimate rows reach PR-AUC ≈ 0.55 (One-class SVM ≈ autoencoder, exported to ONNX) and flag most BORDERLINE rows: anomalous ≠ fraudulent.
- **Production model (Phase 7):** tuned + isotonic-calibrated XGBoost, test PR-AUC **0.885**, F1 **0.827** (precision 0.73, recall 0.94), 0.43 MB artifact; stacking didn't beat it. A 0–100 risk score with validation-chosen tiers puts 53% of fraud in CRITICAL and 79% of legitimate payments in LOW, with native SHAP reasons for every score.
- **Services (Phase 8):** FastAPI gateway + 4 microservices in Docker (or one process); `/score-transaction` p95 27 ms warm; 323 MB for the single-container deployment; 0 errors at 87 req/s in a local load test.
- **Dashboard (Phase 9):** React SPA with 9 pages (overview, data quality, explorer, pattern lab, ML lab, clustering, network, what-if risk scorer, platform), ready for Vercel.
- Full write-up: `reports/FinSight_AI_Project_Report.pdf` (`make report`).
