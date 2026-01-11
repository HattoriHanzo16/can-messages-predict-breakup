# Can Messages Predict Breakup?

## Project Overview
This project tests whether **message content alone** can predict breakup-related posts.
Each Reddit post is treated as a conversation-level sample, and we train two baseline
text classifiers (Logistic Regression and Decision Tree) using TF-IDF features.

## Problem Statement
Can the language in relationship messages indicate a breakup outcome?

## Dataset
Two Reddit datasets stored in `data/raw/`:
- `reddit_breakup_dataset_cleaned.csv` (label: breakup = 1)
- `relationship_advice.csv` (label: breakup = 0)

**Labeling strategy (proxy):**
The label is derived from subreddit source, not verified ground-truth outcomes.

## Repository Structure (Minimal)
```
project/
|-- data/
|   |-- raw/
|   |-- processed/
|-- reports/
|   |-- figures/
|   |-- results/
|-- scripts/
|   |-- run_full_analysis.py
|-- src/
|   |-- __init__.py
|   |-- data_processing.py
|   |-- visualization.py
|-- requirements.txt
|-- README.md
```

## Setup (if needed)
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Run (Generates All Results)
```bash
env/bin/python scripts/run_full_analysis.py
```

## Outputs
- `data/processed/combined_cleaned.csv`
- `reports/data_quality_report.md`
- `reports/results/metrics.md`
- `reports/results/metrics.json`
- `reports/figures/*.png`

## Latest Results (from current run)
- Logistic Regression: Accuracy 0.9029, Precision 0.8469, Recall 0.8783, F1 0.8623
- Decision Tree: Accuracy 0.8132, Precision 0.7231, Recall 0.7460, F1 0.7344

See `reports/results/metrics.md` for full details and `reports/figures/` for charts.

## Notes
- Models use **text only** (no engagement or metadata signals).
- Labels are proxy and may reflect subreddit-specific language.
