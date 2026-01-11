SHELL := /bin/bash
PY := python
VENV := env
CONFIG := config/config.yaml

.PHONY: help setup run serve clean clean-reports clean-cache
.PHONY: demo

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

serve:
	$(VENV)/bin/python serve.py --config $(CONFIG)

clean:
	rm -rf data/processed/* reports/results/* reports/figures/* reports/report.html reports/report.md

clean-reports:
	rm -rf reports/results/* reports/figures/* reports/report.html reports/report.md

clean-cache:
	rm -rf reports/.cache reports/.mpl_cache

demo:
	@if command -v open >/dev/null 2>&1; then open demo/breakup_simulator.html; \
	elif command -v xdg-open >/dev/null 2>&1; then xdg-open demo/breakup_simulator.html; \
	else echo "Open demo/breakup_simulator.html in your browser."; fi
