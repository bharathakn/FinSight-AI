PY      := .venv/bin/python
STAGES  ?=

.PHONY: install install-core test pipeline lab report clean-curated bundle serve docker-build up down scale loadtest measure publish-artifacts dashboard dashboard-build

install:            ## full environment (all extras, CPU-only torch/xgboost)
	uv sync --all-extras

install-core:       ## minimal environment
	uv sync

test:
	$(PY) -m pytest -q

pipeline:           ## make pipeline  |  make pipeline STAGES=inject,wrangle,graph
	PYTHONPATH=src $(PY) -m finsight.pipeline $(if $(STAGES),--stages $(STAGES),)

lab:
	$(PY) -m jupyter lab --notebook-dir=notebooks

report:             ## rebuild reports/FinSight_AI_Project_Report.pdf from current results
	$(PY) docs/report/build_report.py

clean-curated:      ## remove regenerable outputs (curated data + artifacts)
	rm -rf dataset/curated artifacts/*

# ---------------------------------------------------------------- Phase 8: services + containers
COMPOSE := docker compose -f docker/docker-compose.yml

bundle:             ## package trained artifacts → artifacts/bundle/ (+ copy into the Docker build context)
	PYTHONPATH=src $(PY) -m finsight.serving.bundle build
	cp artifacts/bundle/finsight-artifacts.tar.gz docker/bundle/

serve:              ## run the gateway locally in inprocess mode (http://localhost:8000/docs)
	PYTHONPATH=src:. FINSIGHT_SERVICE_MODE=inprocess $(PY) -m uvicorn services.gateway.main:app --port 8000 --reload --reload-dir src

docker-build:       ## build all service images (incl. the single-container monolith)
	$(COMPOSE) --profile monolith build

up:                 ## microservices (remote mode): gateway :8000 + 4 internal services
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) --profile monolith down

scale:              ## horizontal scaling demo: make scale N=3
	$(COMPOSE) up -d --scale ml-service=$(or $(N),3)

loadtest:           ## locust against the gateway: make loadtest USERS=30 LABEL=remote
	$(PY) -m locust -f tests/load/locustfile.py --host $(or $(URL),http://localhost:8000) --headless \
		-u $(or $(USERS),30) -r 10 -t $(or $(DURATION),40s) --csv reports/load_$(or $(LABEL),run) --only-summary

measure:            ## warm latency (+ cold start with COLD=1): make measure URL=... LABEL=...
	$(PY) deploy/measure.py --url $(or $(URL),http://localhost:8000) --label $(or $(LABEL),local) $(if $(COLD),--cold,)

publish-artifacts:  ## upload the bundle as a GitHub Release (needs gh); prints ARTIFACTS_URL
	./deploy/publish_artifacts.sh

# ---------------------------------------------------------------- Phase 9: dashboard (React SPA → Vercel)
dashboard:          ## Vite dev server on :5173 (API at VITE_API_BASE_URL, default http://localhost:8000; run `make serve` too)
	cd frontend && npm install --no-audit --no-fund && npm run dev

dashboard-build:    ## type-check + static production build → frontend/dist (what Vercel serves)
	cd frontend && npm install --no-audit --no-fund && npm run build
