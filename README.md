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
make pipeline             # v1.1 injection → master tables → graph features → sanctions
```

## Documentation
| Doc | Contents |
|---|---|
| [FinSight_AI_Project_Plan.md](FinSight_AI_Project_Plan.md) | Design spec: architecture, experiments, research questions, syllabus mapping |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Phase-by-phase tracker with acceptance criteria |
| [docs/DATA.md](docs/DATA.md) | Data sources, lineage, column roles, leakage list |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log with evidence |
| [docs/SETUP.md](docs/SETUP.md) | Installation + manual setup tasks |
| [CLAUDE.md](CLAUDE.md) | Working rules for AI coding agents |

## Headline result so far (Phase 1b)
RF F1 on `is_fraud`: v1.0 control A/B/C = 0.724 / 0.874 / 0.872; v1.1 = 0.754 / 0.810 / **0.839**. Network features help only where real structure exists.
