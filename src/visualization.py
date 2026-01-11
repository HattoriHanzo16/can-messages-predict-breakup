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
    numeric_cols = [
        "text_length",
        "word_count",
        "breakup_term_count",
        "breakup_term_ratio",
        "sentiment_score",
        "first_person_ratio",
        "question_marks",
        "exclamation_marks",
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    corr = df[cols].corr()
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


def plot_confusion_matrix(cm: list, title: str, save_path: str | None = None) -> None:
    """Plot a confusion matrix heatmap."""
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    _finalize_plot(save_path)


def plot_roc_curves(curves: list, save_path: str | None = None) -> None:
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(6, 5))
    for curve in curves:
        plt.plot(curve["fpr"], curve["tpr"], label=f"{curve['label']} (AUC={curve['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#666666")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    _finalize_plot(save_path)


def plot_top_terms(top_terms: dict, save_path: str | None = None) -> None:
    """Plot top positive/negative terms with coefficients."""
    pos_terms = top_terms["positive"][::-1]
    neg_terms = top_terms["negative"][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].barh([t[0] for t in neg_terms], [t[1] for t in neg_terms], color="#4C72B0")
    axes[0].set_title("Top Terms for Non-Breakup (0)")
    axes[0].set_xlabel("Coefficient")

    axes[1].barh([t[0] for t in pos_terms], [t[1] for t in pos_terms], color="#DD8452")
    axes[1].set_title("Top Terms for Breakup (1)")
    axes[1].set_xlabel("Coefficient")

    plt.tight_layout()
    _finalize_plot(save_path)
