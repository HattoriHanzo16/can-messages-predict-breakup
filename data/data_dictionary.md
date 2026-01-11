# Data Dictionary

## Source 1: reddit_breakup_dataset_cleaned.csv (r/BreakUps)
- title: Post title
- body: Post body text
- upvotes: Upvote count (string in raw data)
- comments_count: Number of comments (string in raw data)
- post_date: Original post date/time (string)
- flair: Post flair (may be empty)
- url: Reddit post URL
- author_age: Author age if provided (often missing)
- relationship_length: Relationship length if provided (often missing)
- top_comments: Concatenated top comments (delimiter: " ||| ")

## Source 2: relationship_advice.csv (r/relationship_advice)
- title: Post title
- score: Reddit score (string)
- id: Post id
- url: Reddit post URL
- comms_num: Number of comments (string)
- created: Unix timestamp (string)
- body: Post body text
- timestamp: Human-readable timestamp

## Derived Fields (processed)
- text: Concatenation of title + body
- text_length: Character length of text
- word_count: Word count of text
- upvotes_num: Numeric upvotes/score
- comments_num: Numeric comment count
- breakup: Label (1 = breakup subreddit, 0 = relationship advice subreddit)
