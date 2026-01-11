# Data Quality Report
- Raw rows: 2833
- Clean rows: 2728
- Empty text rows (raw): 0
- Duplicate text rows (raw): 98

## Class Balance (cleaned)
- breakup=1: 945
- breakup=0: 1783

## Numeric Summary (cleaned)
| stat | text_length | word_count | breakup_term_count | breakup_term_ratio | sentiment_score | first_person_ratio | question_marks | exclamation_marks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count | 2728.00 | 2728.00 | 2728.00 | 2728.00 | 2728.00 | 2728.00 | 2728.00 | 2728.00 |
| mean | 1125.44 | 218.12 | 2.05 | 0.01 | 0.00 | 0.08 | 1.13 | 0.00 |
| std | 1043.81 | 197.35 | 2.37 | 0.02 | 0.01 | 0.05 | 1.41 | 0.00 |
| min | 8.00 | 1.00 | 0.00 | 0.00 | -0.07 | 0.00 | 0.00 | 0.00 |
| 25% | 333.75 | 65.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.00 |
| 50% | 786.50 | 155.50 | 1.00 | 0.01 | 0.00 | 0.09 | 1.00 | 0.00 |
| 75% | 1635.00 | 325.25 | 3.00 | 0.02 | 0.01 | 0.11 | 2.00 | 0.00 |
| max | 3798.50 | 687.00 | 7.50 | 0.50 | 0.14 | 0.33 | 5.00 | 0.00 |

## Notes
- Labels are proxy based on subreddit source (topic-level, not ground truth).
- Models use message content only; no engagement metadata is used.
