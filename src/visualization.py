"""Visualization helpers for EDA."""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set_theme(style="whitegrid")


def _finalize_plot(save_path: str | None) -> None:
    """Save or show the current plot."""
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_class_balance(
    df: pd.DataFrame,
    label_col: str = "breakup",
    save_path: str | None = None,
) -> None:
    """Plot class distribution for the label."""
    ax = sns.countplot(data=df, x=label_col)
    ax.set_title("Class Balance")
    ax.set_xlabel("Breakup Label")
    ax.set_ylabel("Count")
    plt.tight_layout()
    _finalize_plot(save_path)


def plot_text_length_distribution(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Histogram of text lengths."""
    ax = sns.histplot(df["text_length"], bins=50, kde=True)
    ax.set_title("Text Length Distribution")
    ax.set_xlabel("Text Length (chars)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    _finalize_plot(save_path)


def plot_word_count_distribution(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Histogram of word counts."""
    ax = sns.histplot(df["word_count"], bins=50, kde=True)
    ax.set_title("Word Count Distribution")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Count")
    plt.tight_layout()
    _finalize_plot(save_path)


def plot_text_length_by_class(
    df: pd.DataFrame,
    label_col: str = "breakup",
    save_path: str | None = None,
) -> None:
    """Box plot of text length by class label."""
    ax = sns.boxplot(data=df, x=label_col, y="text_length")
    ax.set_title("Text Length by Breakup Label")
    ax.set_xlabel("Breakup Label")
    ax.set_ylabel("Text Length (chars)")
    plt.tight_layout()
    _finalize_plot(save_path)


def plot_numeric_boxplots(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Box plots for numeric columns."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(y=df["text_length"], ax=axes[0])
    axes[0].set_title("Text Length")
    sns.boxplot(y=df["word_count"], ax=axes[1])
    axes[1].set_title("Word Count")
    plt.tight_layout()
    _finalize_plot(save_path)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Correlation heatmap for numeric features."""
    corr = df[["text_length", "word_count"]].corr()
    ax = sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    _finalize_plot(save_path)


def plot_text_length_vs_word_count(df: pd.DataFrame, save_path: str | None = None) -> None:
    """Scatter plot with trend line."""
    ax = sns.regplot(x="text_length", y="word_count", data=df, scatter_kws={"alpha": 0.5})
    ax.set_title("Text Length vs Word Count")
    ax.set_xlabel("Text Length (chars)")
    ax.set_ylabel("Word Count")
    plt.tight_layout()
    _finalize_plot(save_path)
