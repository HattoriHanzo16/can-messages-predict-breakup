# Can Messages Predict Breakup?

## Overview
This project answers one question: **Can message content predict breakups?**
We analyze relationship-related Reddit posts, engineer text signals, and compare three models
(Logistic Regression, Decision Tree, Linear SVM). The results are packaged into a
conference-ready HTML report plus supporting artifacts.

## Prerequisites
- Python 3.10+ installed
- macOS / Linux / Windows

## Setup (Recommended)
```bash
make setup
```

## Run the Full Pipeline
```bash
make run
```
This generates the cleaned dataset, figures, model metrics, and reports.

## Open the Report
The HTML report is here:
- `reports/report.html`

To auto-open after every run, set in `config/config.yaml`:
```
reporting:
  open_report: true
```

## Live Demo (ML-Backed)
Run the model server and open the interactive demo:
```bash
make run
make serve
```
Then open:
- `http://127.0.0.1:5000`

If the server is not running, the demo still works using a lightweight fallback heuristic.

## Outputs (What You Get)
- Cleaned dataset: `data/processed/combined_cleaned.csv`
- Data quality report: `reports/data_quality_report.md`
- Metrics: `reports/results/metrics.md`, `reports/results/metrics.json`
- Misclassifications: `reports/results/misclassifications.csv`
- Excel report (if `openpyxl` installed): `reports/results/metrics.xlsx`
- Main report: `reports/report.html` + `reports/report.md`
- Figures: `reports/figures/*.png`
- Interactive chart (if Plotly installed): `reports/figures/interactive_breakup_signals.html`
- Saved model: `models/breakup_model.joblib`

## Configuration
Edit `config/config.yaml` to change:
- dataset paths
- test split and TF-IDF settings
- output locations
- report behavior

## Useful Make Targets
```bash
make setup         # create env + install deps
make run           # run full pipeline
make serve         # serve the ML-backed demo
make clean         # remove processed data + reports
make clean-reports # remove reports only
make clean-cache   # remove matplotlib caches
```

## Troubleshooting
- **Excel export missing:** install `openpyxl` then re-run.
- **Interactive chart missing:** install `plotly` then re-run.
- **Demo server missing:** install `flask` then re-run.

## Notes
- Labels are proxy-based on subreddit topic, not verified outcomes.
- Models use message content only; engagement metadata is excluded.
