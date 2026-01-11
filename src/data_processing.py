"""Data loading and preprocessing utilities."""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd


_URL_RE = re.compile(r"https?://\S+")
_WORD_RE = re.compile(r"[a-z']+")

_BREAKUP_TERMS = {
    "breakup",
    "broke",
    "broken",
    "break",
    "dumped",
    "ex",
    "exes",
    "no",
    "contact",
    "heartbreak",
    "heartbroken",
    "split",
    "separated",
    "divorce",
    "left",
    "leave",
    "leaving",
}

_POSITIVE_WORDS = {
    "love",
    "loved",
    "happy",
    "hope",
    "support",
    "trust",
    "care",
    "safe",
    "kind",
    "good",
    "better",
}

_NEGATIVE_WORDS = {
    "sad",
    "hurt",
    "pain",
    "angry",
    "upset",
    "cry",
    "cried",
    "lonely",
    "anxious",
    "stress",
    "cheat",
    "cheated",
    "lying",
    "toxic",
}

_FIRST_PERSON = {"i", "me", "my", "mine", "we", "our", "us"}


def load_raw_data(
    breakup_path: str,
    advice_path: str,
) -> pd.DataFrame:
    """Load raw CSVs and return a unified DataFrame with labels.

    Parameters
    ----------
    breakup_path : str
        Path to breakup subreddit dataset.
    advice_path : str
        Path to relationship advice dataset.

    Returns
    -------
    pd.DataFrame
        Combined dataset with standardized columns and labels.
    """
    try:
        breakup_df = pd.read_csv(breakup_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Breakup dataset not found: {breakup_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read breakup dataset: {breakup_path}") from exc

    try:
        advice_df = pd.read_csv(advice_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Advice dataset not found: {advice_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read advice dataset: {advice_path}") from exc

    breakup_df = _standardize_breakup(breakup_df)
    advice_df = _standardize_advice(advice_df)

    combined = pd.concat([breakup_df, advice_df], ignore_index=True)
    combined["source"] = combined["source"].astype("category")
    combined["breakup"] = combined["breakup"].astype(int)
    return combined


def _standardize_breakup(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename fields from the breakup dataset."""
    df = df.copy()
    df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
    df["source"] = "breakup"
    df["breakup"] = 1
    keep = ["text", "source", "breakup"]
    return df[keep]


def _standardize_advice(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename fields from the advice dataset."""
    df = df.copy()
    df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
    df["source"] = "advice"
    df["breakup"] = 0
    keep = ["text", "source", "breakup"]
    return df[keep]


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and clean text fields."""
    df = df.copy()
    df["text"] = (
        df["text"]
        .fillna("")
        .str.replace(_URL_RE, "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )
    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived text features for modeling."""
    df = df.copy()
    df["text_length"] = df["text"].str.len()

    def extract_features(text: str) -> pd.Series:
        tokens = _WORD_RE.findall(text)
        word_count = len(tokens)
        denom = max(word_count, 1)
        breakup_count = sum(t in _BREAKUP_TERMS for t in tokens)
        pos_count = sum(t in _POSITIVE_WORDS for t in tokens)
        neg_count = sum(t in _NEGATIVE_WORDS for t in tokens)
        first_person_count = sum(t in _FIRST_PERSON for t in tokens)

        return pd.Series(
            {
                "word_count": word_count,
                "breakup_term_count": breakup_count,
                "breakup_term_ratio": breakup_count / denom,
                "sentiment_score": (pos_count - neg_count) / denom,
                "first_person_ratio": first_person_count / denom,
                "question_marks": text.count("?"),
                "exclamation_marks": text.count("!"),
            }
        )

    features = df["text"].apply(extract_features)
    df = pd.concat([df, features], axis=1)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and drop empty texts."""
    df = df.copy()
    df = df[df["text"].astype(str).str.len() > 0]
    for col in [
        "text_length",
        "word_count",
        "breakup_term_count",
        "breakup_term_ratio",
        "sentiment_score",
        "first_person_ratio",
        "question_marks",
        "exclamation_marks",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns: Tuple[str, ...]) -> pd.DataFrame:
    """Cap outliers using IQR for the specified numeric columns."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def preprocess_pipeline(breakup_path: str, advice_path: str) -> pd.DataFrame:
    """Full preprocessing pipeline returning a cleaned dataset."""
    try:
        df = load_raw_data(breakup_path, advice_path)
        df = clean_text(df)
        df = add_text_features(df)
        df = handle_missing(df)
        df = cap_outliers_iqr(
            df,
            (
                "text_length",
                "word_count",
                "breakup_term_count",
                "question_marks",
                "exclamation_marks",
            ),
        )
        df = df.drop_duplicates(subset=["text"])
        return df
    except Exception as exc:
        raise RuntimeError("Preprocessing pipeline failed") from exc
