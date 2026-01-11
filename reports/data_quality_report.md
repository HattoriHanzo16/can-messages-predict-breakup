# Data Quality Report

## Initial Assessment
- Two sources: r/BreakUps and r/relationship_advice
- Total rows (raw): 948 + 1885
- Text fields include missing values and inconsistent formatting
- Numeric fields stored as strings

## Cleaning Actions
- Combined title + body into a unified `text` field
- Removed URLs and normalized whitespace
- Dropped rows with empty text
- Converted `upvotes/score` and `comments` to numeric
- Filled numeric missing values with median
- Capped numeric outliers using IQR
- Removed duplicate text entries

## Known Limitations
- Labels are proxy (subreddit-based), not verified outcomes
- Potential topic/style bias between subreddits
