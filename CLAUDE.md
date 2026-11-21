# CLAUDE.md — Agent Operating Manual for FinSight AI

Claude Code loads this file at the start of every session. Read it first, then check `docs/PROGRESS.md` to see where the project stands.

## What this project is
FinSight AI is one project submitted to three courses: **DMW** (Data Mining & Wrangling), **ML** (Machine Learning) and **CCT** (Cloud Computing & Technology). It covers the full KDD + association-rule mining pipeline, a complete ML model lifecycle, and FastAPI microservices on GCP Cloud Run, all on a synthetic Indian payments dataset (10K transactions, UPI/NEFT/RTGS).

## Document map (single source of truth for each topic)
| File | Purpose | Update when |
|---|---|---|
| `FinSight_AI_Project_Plan.md` | **Design spec**: architecture, experiments, research questions, syllabus mapping | Design changes (log the change in DECISIONS.md too) |
| `docs/PROGRESS.md` | **Phase tracker**: status, checklists, acceptance criteria, session log | **At the end of every work session / phase** |
| `docs/DECISIONS.md` | Decision log (what, why, evidence, alternatives) | Any non-obvious choice |
| `docs/DATA.md` | Dataset dictionary, lineage (v1.0 → v1.1), leakage list, audit facts | Data schema / pipeline changes |
| `docs/SETUP.md` | Install commands + manual (human-only) tasks | New dependency or infra step |
| `README.md` | Public overview + quickstart | Milestones |

## Working rules
1. **Phase order is mandatory:** data → injection → EDA → graph → mining → supervised → unsupervised/anomaly → tuning/ensemble → services → dashboard → deploy. Don't start a phase until the previous phase's acceptance criteria in PROGRESS.md are met.
2. **Logic lives in `src/finsight/`.** Notebooks only import and display results. Services only load artifacts produced by `finsight.pipeline`.
3. **Leakage blacklist (never use as model features):** `is_fraud`, `risk_label`, `scenario_id`, `alert_id`, `is_flagged_alert`, `rapid_forward_delta_sec` (original v1.0 column). ID columns (account/customer/device IDs, VPA, IFSC, IP) are not features either. Both lists live in `finsight.config` and must be enforced in code.
4. **Primary training data is v1.1** (`dataset/processed/custom_india_v1_1/`). v1.0 is the untouched control. Never edit `dataset/raw/` or `dataset/processed/custom_india/`.
5. **Graph features must be point-in-time**: they may only use transactions strictly before the current one.
6. **SMOTE only inside the training pipeline** (imblearn `Pipeline`). Thresholds are tuned on validation, never on test. Report PR-AUC + F1, never accuracy alone.
7. **Everything is seeded** (`config.SEED = 42`), so results must be reproducible.
8. **Figures:** always use `finsight.viz` (`apply_style()`, `LABEL_COLORS`, `save()`). Figure functions are pure (return a Figure) and live in `src/`; notebooks call them. Look at every rendered PNG before calling it done (D-015).
9. Every new module gets tests in `tests/`. Run `make test` before marking a task done.
10. At the end of a session: tick the boxes in `docs/PROGRESS.md`, add a session-log entry, record any decisions in `docs/DECISIONS.md`, and record numbers in the plan's result tables where relevant.
11. Commit only when the user asks. Never commit `dataset/`, `.venv/`, `artifacts/`, or secrets.

## Commands
```bash
make install          # uv sync --all-extras  (Python 3.12 venv in .venv)
make test             # pytest
make pipeline         # python -m finsight.pipeline (all implemented stages)
make pipeline STAGES=inject,wrangle,graph
make lab              # Jupyter Lab
```
Without make: `PYTHONPATH=src .venv/bin/python -m finsight.pipeline --stages ...`

## Code layout
```
src/finsight/   config · data/ · graph/ · mining/ · ml/ · sanctions/ · scoring/ · pipeline.py
services/       FastAPI apps (gateway, ml, mining, graph, sanctions)       [Phase 8]
frontend/       React + Vite dashboard → Vercel                            [Phase 9]
docker/ deploy/ Dockerfiles, compose, render.yaml, scripts                 [Phase 8, 10]
dataset/        raw + processed (git-ignored); curated/ = pipeline outputs
artifacts/      models, graph state, sanctions entities (git-ignored)
reports/        JSON/CSV/figures that back the report (tracked)
```

## Environment facts
- Ubuntu 26.04, x86_64, 8 cores, 14 GB RAM. NVIDIA MX450 (2 GB VRAM, driver 580 / CUDA 13) is present but **intentionally unused**: CPU-only wheels are configured in `pyproject.toml` (see D-013). Switch to GPU only for a measured bottleneck.
- Use the `.venv` interpreter (Python 3.12 via uv). The system Python is 3.14; don't use it.
- pandas 3 uses non-ns datetime units, so use `finsight.data.inject.epoch_seconds()` for Unix seconds.
- The network can be slow, so prefer small, targeted installs.
- Deployment: **Vercel** (React SPA in `frontend/`) + **Render** (FastAPI Docker services, `render.yaml`); artifacts via GitHub Releases (D-022–D-024). Render free tier: no private networking, 15-min spin-down, ~512 MB RAM.
