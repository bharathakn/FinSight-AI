PY      := .venv/bin/python
STAGES  ?=

.PHONY: install install-core test pipeline lab report clean-curated

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
