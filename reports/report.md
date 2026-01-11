# Can Messages Predict Breakup?
Text-based breakup prediction using Reddit messages.

## Answer to the Research Question
Yes: Can messages predict breakup?
Yes. Messages can predict breakup-related outcomes in this dataset. The best model (Linear Svm) reaches F1 0.860 and ROC-AUC 0.958 on held-out data.

Evidence:
- Best model performance: F1 0.860, ROC-AUC 0.958
- Top predictive terms are breakup-centric (e.g., breakup, broke, miss, pain).
- Breakup-labeled posts tend to be longer and more detailed.

## Executive Summary
Message content shows strong predictive power for breakup-related posts. The top-performing model (Linear Svm) achieved an F1 score of 0.860 and ROC-AUC of 0.958, indicating reliable detection of breakup language patterns using text alone.

## Key Insights
- Linear Svm delivered the strongest F1 score and overall balance of precision/recall.
- Breakup-related language (e.g., breakup, broke, miss, pain) dominates predictive terms.
- Longer, more detailed posts are more common in breakup-labeled samples.

## Dataset
- Total rows: 2728
- Breakup posts: 945
- Non-breakup posts: 1783
- Breakup share: 34.6%

## Model Performance
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.868 | 0.806 | 0.815 | 0.811 | 0.939 |
| Decision Tree | 0.800 | 0.725 | 0.683 | 0.703 | 0.775 |
| Linear Svm | 0.905 | 0.874 | 0.847 | 0.860 | 0.958 |

## Top Predictive Terms (Breakup)
- breakup: 1.919
- broke: 1.798
- struggling: 1.560
- miss: 1.505
- breakups: 1.423
- pain: 1.331
- did: 1.159
- days: 1.153
- breaking: 1.133
- wasn: 1.125
- days ago: 1.122
- friends family: 1.122
- healing: 1.121
- love: 1.120
- new: 1.100
- message: 1.086
- dogs: 1.058
- anymore: 1.050
- end: 1.014
- song: 1.013

## Top Predictive Terms (Non-breakup)
- comment: -4.665
- christmas: -1.683
- boyfriend: -1.494
- husband: -1.489
- bf: -1.422
- really: -1.298
- thinks: -1.214
- school: -1.199
- gf: -1.199
- tldr: -1.169
- tell: -1.096
- hurtful: -1.073
- explanation: -1.058
- began: -1.037
- asked: -1.021
- partner: -1.002
- messaged: -0.990
- sex: -0.989
- situation: -0.979
- does: -0.976

## Error Analysis
The best model (Linear Svm) misclassified 52 posts. False positives often involve general relationship conflict language, while false negatives tend to be subtle breakup narratives without explicit breakup terms.

## Limitations
- Labels are proxy-based on subreddit topic, not verified outcomes.
- Language patterns may reflect subreddit style rather than real-world breakup signals.
- This is a single train/test split; performance may vary across samples.