# FinSight AI — Build Progress Tracker

> **How to use:** each phase lists its goal, tasks, deliverables and acceptance criteria. A phase is **DONE** only when every acceptance criterion holds. Agents and humans update this file at the end of every session (tick boxes, update the status table, add a session-log entry).
> Status legend: ✅ done · 🟡 in progress · ⬜ not started · ⛔ blocked

## Current state
- **Current phase:** Phase 6 — Unsupervised + anomaly detection (next)
- **Blockers:** none. Deployment target is now **Vercel + Render** (D-022). Render/Vercel accounts and the `gh` CLI are needed before Phase 10 (see `docs/SETUP.md` §5).
- **Last updated:** 2026-11-21

## Status overview
| # | Phase | Subject | Status | Key output |
|---|---|---|---|---|
| 0 | Dataset audit | DMW | ✅ | Plan §0 findings |
| 1 | Wrangling + data quality | DMW | 🟡 core done | `dataset/curated/master_transactions_*.parquet`, `reports/data_quality_india_*.json` |
| 1b | Graph-structure injection (v1.1) | DMW | ✅ | `dataset/processed/custom_india_v1_1/` |
| 2 | EDA | DMW+ML | ✅ | `notebooks/02_eda.ipynb`, `reports/figures/eda_01…14`, `reports/eda_summary.json` |
| 3 | Graph features | ML+DMW | 🟡 features done | `master_features_*.parquet`, `artifacts/graph/` |
| 4 | Association rule mining | DMW | ✅ | `reports/association_rules_v1_{0,1}.csv`, `mining_benchmark.csv`, `figures/mining_01…08`, `artifacts/mining/rules_v1_1.parquet` |
| 5 | Supervised ML (A/B/C) | ML | ✅ | `reports/ml_experiment_results.csv`, `ml_paired_tests.csv`, `figures/ml_01…09`, `artifacts/models/phase5/` |
| 6 | Unsupervised + anomaly | ML | ⬜ | clustering + anomaly reports, `.onnx` AE |
| 7 | Tuning + stacking + SHAP | ML | ⬜ | `artifacts/models/`, `metrics.json` |
| 8 | Services + Docker | CCT | ⬜ | `services/`, `docker/docker-compose.yml`, `render.yaml` |
| 9 | Dashboard (React SPA) | All | ⬜ | `frontend/` (9 pages, Vercel) |
| 10 | Render + Vercel deploy + report | CCT | ⬜ | live URLs, final report |

---

## Phase 0 — Dataset Audit ✅
- [x] Load all 7 India files; dtypes, shapes, nulls, duplicates
- [x] Referential integrity (100% FK match; beneficiary pair-join only 8/10,000)
- [x] Label distribution (82% LEGIT / 13.5% FRAUD / 4.5% BORDERLINE)
- [x] Leakage scan (`rapid_forward_delta_sec`, `is_flagged_alert`, `alert_id`)
- [x] Separability check (Exp A ≈ 0.72 F1, Exp B ≈ 0.87 F1, network AUC ≈ 0.50, so no structure)
- [x] Plan revised to v2 (Plan §0)

## Phase 1 — Data Wrangling + Quality 🟡
**Goal:** KDD steps 1–3. Typed loading, cleaning, 7-file integration, quality reporting, sanctions parsing.
- [x] `finsight/data/loader.py` — typed loaders (India, PaySim lazy, IEEE-CIS join)
- [x] `finsight/data/preprocessor.py` — clean + merge → master (57 cols)
- [x] `finsight/data/quality.py` — completeness, duplicates, RI, redundancy, leakage scan
- [x] `finsight/sanctions/parser.py` + `screener.py` — 20,332 entities / 27,263 name variants
- [ ] IEEE-CIS profiling: transaction ⋈ identity, missingness heat map, imputation comparison → `reports/data_quality_ieee.json`
- [ ] PaySim profiling with Polars lazy scan → `reports/paysim_profile.json`
- [ ] `notebooks/01_data_wrangling.ipynb` (KDD steps labelled)

**Acceptance:** DQ reports exist for India v1.0, v1.1, IEEE-CIS and PaySim; the leakage scan flags the known proxies; `make test` passes.

## Phase 1b — Graph-Structure Injection ✅
- [x] `finsight/data/inject.py` — rings, mules, layering, bursts, structuring, shared devices, decoys, 50% camouflage
- [x] v1.1 written with identical schema + `injection_log.csv` + metadata
- [x] Tests: labels unchanged, invariants, determinism (`tests/test_data.py`)
- [x] Validation: v1.0 A/B/C = 0.724/0.874/0.872; v1.1 = 0.754/0.810/**0.839** (`reports/phase1b_validation.txt`)

## Phase 2 — EDA ✅
**Goal:** understand distributions and back RQ1 with figures.
- [x] Amount distributions by risk label × channel (log violins)
- [x] Fraud rate by channel / hour / day-of-week; city bubble map + state bars
- [x] Spearman correlation; single-feature AUC v1.0 vs v1.1; AUC + mutual-information ranking
- [x] Scenario fingerprints; BORDERLINE vs FRAUDULENT ECDFs; behaviour-flag table (lift)
- [x] Graph EDA: fan-in distribution v1.0 vs v1.1, ring-hub + layering-chain examples, network signal by scenario, camouflage figure
- [x] `src/finsight/eda.py` (pure figure functions) + `finsight/viz.py` (validated palette/style) + `finsight/ml/feature_sets.py` (Exp A/B/C + blacklist)
- [x] Pipeline stage `eda`; `notebooks/02_eda.ipynb` executed (0 errors, 14 figures); tests in `tests/test_features_eda.py`

**Acceptance:** ✅ notebook runs clean; 14 figures; RQ1 summary in the notebook (§7).

**Key findings:** RQ1 top features: log_amount (AUC 0.85), amount_ratio (0.81), beneficiary age (0.79), distance (0.74), new device (0.68), forward_ratio (0.64, network). Channel/weekday are flat; late-night hours reach 22–27% fraud vs 13.5% base. RECENT_BENEFICIARY and FAR_FROM_HOME never fire on legit rows (generator artifact). PASS_THROUGH is 99% precise at 1.6% support. BORDERLINE ≈ FRAUDULENT on all key features. Camouflaged organised fraud: 73% carry ≥1 network red flag vs 12% of legit. Max network single-feature AUC is 0.64, so there's no giveaway.

## Phase 3 — Graph Features 🟡
- [x] `finsight/graph/network_features.py` — 13 point-in-time + 6 global features
- [x] Pipeline stage `graph` → `master_features_*.parquet`, `artifacts/graph/account_metrics_*.parquet`
- [x] Tests: pass-through chain, no future leakage
- [ ] Ring/hub detection report (top fan-in accounts vs ring hubs; community purity)
- [ ] Graph-state artifact for graph-service (serialised edge list + account metrics)
- [ ] `notebooks/07_graph_mining.ipynb`

**Acceptance:** ≥ 8 of the 10 ring hubs rank in the top-20 receivers by 24h distinct senders; graph-state artifact loads in < 2 s.

## Phase 4 — Association Rule Mining ✅
- [x] `finsight/mining/discretiser.py` — 20 items (amount, behaviour, context, 5 network) + 2 label items; extended 49-item universe; PaySim items (Polars)
- [x] `finsight/mining/association.py` — Apriori / FP-Growth / FP-Max, closed itemsets, target rules, productive-rule pruning, minimal emerging patterns, rule detector
- [x] Benchmark: 3 algorithms × min_support {0.05 … 0.002} × {22, 49} items; untraced timing + traced memory (D-017)
- [x] Rules → FRAUDULENT / BORDERLINE on v1.0 and v1.1 (Lift > 1.5, Conf > 0.6); fraud-subset emerging patterns
- [x] PaySim scale run (100K → 6.36M rows) + PaySim fraud rules
- [x] `finsight/mining/report.py`, pipeline stage `mining` (~2.5 min), `notebooks/03_association_mining.ipynb` (0 errors, 8 figures), `tests/test_mining.py`

**Acceptance:** ✅ Apriori = FP-Growth at every support level (asserted); top-10 rules exported; benchmark charts saved.

**Key findings:** v1.1: 534 itemsets → 152 filtered → **49 productive rules (16 with network items)**; v1.0: 34 productive, **0 network**, so mining rediscovers exactly the injected structure. Top rules: {PASS_THROUGH}→F (99%, lift 7.4), {CH_UPI, HIGH_AMOUNT_SPIKE, HUB_RECEIVER}→F (100%), QUICK_INFLOW + behaviour anomaly (91–97%), {NEAR_THRESHOLD, RECENT_BENEFICIARY}→F (90%); one BORDERLINE rule {EXTREME_LOCATION, NEW_DEVICE} (lift 13.4). Benchmark: identical results; Apriori **1,317 MB vs FP-Growth 78.5 MB** at 49 items / s = 0.002; Apriori is faster on small searches (vectorised) while FP-Growth wins on large ones. Closed 39,713 / maximal 5,542 of 56,278. PaySim 6.36M: both linear in rows; Apriori 1.3 GB. Rule detector (in-sample): best F1 0.76 @ 39 rules; 426/886 false positives are BORDERLINE.

**Carry-over to Phase 5:** re-mine rules on the training split only and score the rule detector on the same held-out test set as the ML models (the in-sample numbers above are descriptive).

## Phase 5 — Supervised ML ✅
- [x] `finsight/ml/feature_sets.py` — Exp A/B/C (12/30/49 cols) + blacklist assertion (tested, incl. on fitted pipeline columns)
- [x] `finsight/ml/splits.py` (stratified 60/20/20, temporal, 5×2 CV), `classifiers.py` (6 imblearn pipelines), `evaluator.py` (PR-AUC, val-tuned threshold, per-scenario, paired tests)
- [x] `finsight/ml/supervised.py` + `supervised_figures.py`: 6 models × A/B/C × v1.0/v1.1 (360 CV fits + 36 hold-out); imbalance none/class_weight/SMOTE; temporal; rule baseline on test; 3-class extension; DT depth-4 plot
- [x] Pipeline stage `train_supervised` (~6 min); `notebooks/04_supervised_ml.ipynb` (0 errors, 9 figures); `tests/test_supervised.py` (31 tests total)

**Acceptance:** ✅ results matrix complete; blacklist tested; v1.1 C > B for RF (+0.054) and XGB (+0.047), p < 0.001.

**Key findings:** RQ3: B > A for 12/12 (model × version), p < 0.001. C > B on **v1.1 for 5/6 models** (+0.05 to +0.09, p < 0.001) and **0/6 on v1.0** (control). RQ4: **XGBoost Exp C test PR-AUC 0.875, ROC 0.980, F1 0.801 (P 0.70 / R 0.94)**; RF 0.838; MLP 0.798; KNN 0.709; DT 0.690; NB 0.642. SMOTE never improved PR-AUC (D-020). Rule baseline (train-mined): PR-AUC 0.779, F1 0.748. RQ7: network features lift S7 layering 65% → 85%, S8 mule 90% → 97.5%; hardest is S2 high-value spike (80%). BORDERLINE = 65% of XGB false positives (79% flagged vs 2% of legit); 3-class BORDERLINE F1 0.56. Temporal hold-out = stratified (0.876 vs 0.875).

**Carry-over to Phase 7:** imbalance = class_weight (DT/RF/XGB) / none (KNN/MLP/NB) per D-020; bar to beat = XGB-C test PR-AUC 0.875; consider a BORDERLINE "review" tier.

## Phase 6 — Unsupervised + Anomaly ⬜
- [ ] K-Means (K = 2..8; elbow, silhouette, DB index), DBSCAN (k-distance), PCA/t-SNE
- [ ] IF, OC-SVM, Autoencoder (PyTorch → ONNX) on legit training rows; per-scenario comparison
- [ ] PaySim IF scale track (Polars) vs PaySim `isFraud`
- [ ] `notebooks/05_unsupervised_ml.ipynb`

**Acceptance:** ONNX AE output matches PyTorch within 1e-5; detector comparison table exported.

## Phase 7 — Tuning + Ensemble + SHAP ⬜
- [ ] Grid (DT, KNN) / Randomized (RF, XGB; n_iter = 20) with 5-fold stratified CV
- [ ] Learning curves; stacking (RF + XGB + MLP → LR) + isotonic calibration
- [ ] Threshold + tier cut-offs on validation; scorer weights on validation
- [ ] SHAP global/local; artifacts + `metrics.json` saved
- [ ] `notebooks/06_tuning_ensemble.ipynb`

**Acceptance:** final test-set metrics reported once; artifacts reload and reproduce the test metrics.

## Phase 8 — Services + Docker ⬜
- [ ] `finsight/storage.py` (fsspec local / S3-compatible) + `make publish-artifacts` → GitHub Release
- [ ] FastAPI services: gateway :8000, ml :8001, mining :8002, graph :8003, sanctions :8004 (Pydantic schemas, OpenAPI)
- [ ] `FINSIGHT_SERVICE_MODE` = `inprocess` | `remote`; `X-Internal-Key` middleware; CORS allow-list; request-ID + JSON logs; `/metrics`
- [ ] `finsight/scoring/risk_scorer.py` (weights, overrides, tiers incl. BORDERLINE review tier)
- [ ] Serve-time explanations via XGBoost `pred_contribs` (no `shap` import); RAM < 450 MB in inprocess mode
- [ ] Dockerfiles (slim, multi-stage, non-root; artifacts downloaded at build) + `docker-compose.yml` (remote mode)
- [ ] API tests; locust load test; `--scale ml-service=3` demo

**Acceptance:** `docker compose up` brings up all services healthy; `/score-transaction` p95 < 200 ms warm; the inprocess container stays under 450 MB RSS.

## Phase 9 — Dashboard (React SPA) ⬜
- [ ] `frontend/` React + Vite + TypeScript, Plotly.js; `VITE_API_BASE_URL` env
- [ ] Pages: Overview · Data Quality · Transaction Explorer · Pattern Mining Lab · ML Lab · Clustering & Anomaly · Network Explorer · Risk Scorer · Platform
- [ ] "Waking the backend" state (polls `/health`) for Render cold starts

**Acceptance:** `npm run build` produces a static bundle; every page works against the local gateway and against Render.

## Phase 10 — Render + Vercel Deploy + Report ⬜  *(needs GitHub / Render / Vercel accounts, see SETUP §5)*
- [ ] `render.yaml` (Blueprint: single inprocess service + optional full 5-service variant); `frontend/vercel.json`; GitHub Actions CI (tests)
- [ ] Deploy: Render backend, Vercel frontend; set env vars (`FINSIGHT_INTERNAL_KEY`, `FINSIGHT_CORS_ORIGINS`, `VITE_API_BASE_URL`)
- [ ] Measure: cold start, warm p95, image size, RAM; microservices vs inprocess latency
- [ ] Final report covering all three syllabi; demo rehearsal

**Acceptance:** the public Vercel URL loads and scores a transaction via Render; internal endpoints reject calls without the key.

---

## Session log
| Date | Phase(s) | Summary | Next |
|---|---|---|---|
| 2026-11-21 | 0, 1, 1b, 3 | Audit; plan v2; loaders, merge, DQ report, sanctions, v1.1 injection (+ 50% camouflage), point-in-time graph features; 10 tests passing; docs set created | User installs deps (SETUP.md) → Phase 2 EDA |
| 2026-11-21 | Setup, 2 | User installed all deps (verified: torch CPU, xgboost 3.4, shap, Docker 29, gcloud 594, `finsight` kernel). Phase 2 EDA complete: eda.py, viz.py, feature_sets.py, 14 figures, notebook, 16 tests passing. Found `account_type=MULE_SUSPECT` is not a label proxy (D-014). | Phase 4 association mining (Phase 1 IEEE/PaySim profiling and Phase 3 extras still open) |
| 2026-11-21 | 4 | Association mining complete: discretiser, association, report; benchmark (22/49 items × 5 supports), v1.0/v1.1 rules, emerging patterns, PaySim scale + rules; 8 figures; notebook 03; 21 tests passing. Fixed a measurement bug (tracemalloc inflated FP-Growth time ~10×), see D-017. | Phase 5 supervised ML (add a rule-detector baseline on the test split) |
| 2026-11-21 | 5 | Supervised ML complete: splits, classifiers, evaluator, runner + figures; 5×2 CV paired tests; imbalance, temporal, rule-baseline, 3-class; notebook 04; 31 tests passing. RQ3 confirmed with significance; best = XGB-C 0.875 PR-AUC. | Phase 6 unsupervised + anomaly |
| 2026-11-21 | Planning | Deployment target changed to Vercel + Render (D-022–D-024); plan §8 rewritten; Streamlit → React SPA. Project report PDF generated (`reports/FinSight_AI_Project_Report.pdf`). | Phase 6 |
