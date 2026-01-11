"""Data loading and preprocessing utilities."""

from __future__ import annotations

import re
from typing import Tuple

import pandas as pd


_URL_RE = re.compile(r"https?://\S+")


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
    df["upvotes_num"] = pd.to_numeric(df.get("upvotes"), errors="coerce")
    df["comments_num"] = pd.to_numeric(df.get("comments_count"), errors="coerce")
    df["source"] = "breakup"
    df["breakup"] = 1
    keep = ["text", "upvotes_num", "comments_num", "source", "breakup"]
    return df[keep]


def _standardize_advice(df: pd.DataFrame) -> pd.DataFrame:
    """Select and rename fields from the advice dataset."""
    df = df.copy()
    df["text"] = (df["title"].fillna("") + " " + df["body"].fillna("")).str.strip()
    df["upvotes_num"] = pd.to_numeric(df.get("score"), errors="coerce")
    df["comments_num"] = pd.to_numeric(df.get("comms_num"), errors="coerce")
    df["source"] = "advice"
    df["breakup"] = 0
    keep = ["text", "upvotes_num", "comments_num", "source", "breakup"]
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
    """Add simple derived text features."""
    df = df.copy()
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().apply(len)
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and drop empty texts."""
    df = df.copy()
    df = df[df["text"].astype(str).str.len() > 0]
    for col in ["upvotes_num", "comments_num", "text_length", "word_count"]:
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
        df = cap_outliers_iqr(df, ("upvotes_num", "comments_num"))
        df = df.drop_duplicates(subset=["text"])
        return df
    except Exception as exc:
        raise RuntimeError("Preprocessing pipeline failed") from exc
