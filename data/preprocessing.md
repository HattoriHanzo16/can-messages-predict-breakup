# Data Preprocessing Notes

## Cleaning Steps
1. Standardize columns across both datasets.
2. Convert numeric-like columns (upvotes/score, comments) to numeric.
3. Remove duplicate posts by `title + body`.
4. Fill missing text fields with empty string and drop rows with no text at all.
5. Create derived text features (text length, word count).

## Missing Values Handling
- Text fields: fill with empty strings; drop if combined text is empty.
- Numeric fields: fill missing with median.
- Metadata fields (e.g., author_age, relationship_length): retained but not used in baseline models.

## Outlier Handling
- Use IQR method on numeric features (upvotes/comments) to detect outliers.
- Cap extreme values at upper/lower bounds for modeling to reduce skew.

## Feature Engineering
- `text = title + " " + body`
- `text_length`, `word_count`

## Labeling Rationale
The `breakup` label is a proxy derived from subreddit source:
- r/BreakUps posts -> breakup = 1
- r/relationship_advice posts -> breakup = 0

This is not ground-truth breakup status; it is topic-based.
