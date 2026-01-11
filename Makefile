SHELL := /bin/bash
PY := python
VENV := env
CONFIG := config/config.yaml

.PHONY: help setup run clean clean-reports clean-cache

help:
	@echo "Targets:"
	@echo "  setup         Create venv and install deps"
	@echo "  run           Run full analysis pipeline"
	@echo "  clean         Remove processed data and reports"
	@echo "  clean-reports Remove only reports outputs"
	@echo "  clean-cache   Remove report cache directories"

setup:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

run:
	$(VENV)/bin/python run.py --config $(CONFIG)

clean:
	rm -rf data/processed/* reports/results/* reports/figures/* reports/report.html reports/report.md

clean-reports:
	rm -rf reports/results/* reports/figures/* reports/report.html reports/report.md

clean-cache:
	rm -rf reports/.cache reports/.mpl_cache
