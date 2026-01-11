"""Visualization helpers for EDA."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set_theme(style="whitegrid")


def plot_class_balance(df: pd.DataFrame, label_col: str = "breakup") -> None:
    """Plot class distribution for the label."""
    ax = sns.countplot(data=df, x=label_col)
    ax.set_title("Class Balance")
    ax.set_xlabel("Breakup Label")
    ax.set_ylabel("Count")
    plt.tight_layout()


def plot_text_length_distribution(df: pd.DataFrame) -> None:
    """Histogram of text lengths."""
    ax = sns.histplot(df["text_length"], bins=50, kde=True)
    ax.set_title("Text Length Distribution")
    ax.set_xlabel("Text Length (chars)")
    ax.set_ylabel("Count")
    plt.tight_layout()


def plot_word_count_distribution(df: pd.DataFrame) -> None:
    """Histogram of word counts."""
    ax = sns.histplot(df["word_count"], bins=50, kde=True)
    ax.set_title("Word Count Distribution")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Count")
    plt.tight_layout()


def plot_numeric_boxplots(df: pd.DataFrame) -> None:
    """Box plots for numeric columns."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(y=df["upvotes_num"], ax=axes[0])
    axes[0].set_title("Upvotes")
    sns.boxplot(y=df["comments_num"], ax=axes[1])
    axes[1].set_title("Comments")
    plt.tight_layout()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap for numeric features."""
    corr = df[["upvotes_num", "comments_num", "text_length", "word_count"]].corr()
    ax = sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()


def plot_scatter_text_vs_comments(df: pd.DataFrame) -> None:
    """Scatter plot with trend line."""
    ax = sns.regplot(x="text_length", y="comments_num", data=df, scatter_kws={"alpha": 0.5})
    ax.set_title("Text Length vs Comments")
    ax.set_xlabel("Text Length (chars)")
    ax.set_ylabel("Comments")
    plt.tight_layout()
