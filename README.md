# Can Messages Predict Breakup?

## Project Overview
This project investigates whether relationship-related messages can predict a breakup outcome.
We treat each Reddit post as a conversation-level sample and build a binary classifier using
text and lightweight metadata features. The core goal is to compare two baseline models and
interpret which performs better.

## Problem Statement
Can the language and context in relationship messages indicate a breakup-related outcome?

## Dataset
We use two Reddit-based datasets stored in `data/raw/`:
- `reddit_breakup_dataset_cleaned.csv` (positive class: breakup-related posts)
- `relationship_advice.csv` (negative class: general advice posts)

**Labeling strategy (proxy):**
Posts from r/BreakUps are labeled `breakup = 1`, and posts from r/relationship_advice are
labeled `breakup = 0`. This is a proxy label based on subreddit topic; results should be
interpreted with that limitation in mind.

## Repository Structure
```
project/
|-- data/
|   |-- raw/
|   |-- processed/
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_data_preprocessing.ipynb
|   |-- 03_eda_visualization.ipynb
|   |-- 04_machine_learning.ipynb
|-- src/
|   |-- __init__.py
|   |-- data_processing.py
|   |-- visualization.py
|   |-- models.py
|-- reports/
|   |-- figures/
|   |-- results/
|-- README.md
|-- CONTRIBUTIONS.md
|-- requirements.txt
|-- .gitignore
```

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run notebooks in order:
1. `notebooks/01_data_exploration.ipynb`
2. `notebooks/02_data_preprocessing.ipynb`
3. `notebooks/03_eda_visualization.ipynb`
4. `notebooks/04_machine_learning.ipynb`

## Results (to be updated)
- Baseline metrics for Logistic Regression and Decision Tree
- Model comparison and key insights

## Notes
- This is a conversation-level classification project.
- The dataset labeling is based on subreddit topic, not verified ground truth.
