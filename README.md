# Can Messages Predict Breakup?

## Overview
This is a production-grade analysis pipeline that answers one question: **Can message content predict breakups?**
It uses Reddit posts as conversation-level samples and trains three models (Logistic Regression, Decision Tree,
Linear SVM) on TF-IDF text plus engineered linguistic signals.

## Quick Start
```bash
make setup
make run
```

## What the Pipeline Produces
- Cleaned dataset: `data/processed/combined_cleaned.csv`
- Data quality report: `reports/data_quality_report.md`
- Full metrics: `reports/results/metrics.md` + `reports/results/metrics.json`
- Excel report (metrics + feature summary + misclassifications): `reports/results/metrics.xlsx`
- HTML report: `reports/report.html`
- Figures: `reports/figures/*.png`
- Optional interactive plot: `reports/figures/interactive_breakup_signals.html`

## Configuration
Edit `config/config.yaml` to adjust:
- dataset paths
- TF-IDF settings
- test split
- output locations

## Dependencies
Install with:
```bash
pip install -r requirements.txt
```

## Notes
- Labels are proxy-based on subreddit topic, not verified outcomes.
- Models use message content only; engagement metadata is excluded.
- If Plotly is missing, the interactive chart is skipped gracefully.
- If openpyxl is missing, Excel export will be skipped with a log message.
