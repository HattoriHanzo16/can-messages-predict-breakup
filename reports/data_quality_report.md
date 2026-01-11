# Data Quality Report
- Raw rows: 2833
- Clean rows: 2728
- Empty text rows (raw): 0
- Duplicate text rows (raw): 98

## Class Balance (cleaned)
- breakup=1: 945
- breakup=0: 1783

## Numeric Summary (cleaned)
| stat | text_length | word_count |
| --- | --- | --- |
| count | 2728.00 | 2728.00 |
| mean | 1125.44 | 214.90 |
| std | 1043.81 | 195.05 |
| min | 8.00 | 2.00 |
| 25% | 333.75 | 64.00 |
| 50% | 786.50 | 153.00 |
| 75% | 1635.00 | 321.00 |
| max | 3798.50 | 678.50 |

## Notes
- Labels are proxy based on subreddit source (topic-level, not ground truth).
- This analysis uses message content only; engagement signals are not used for modeling.
