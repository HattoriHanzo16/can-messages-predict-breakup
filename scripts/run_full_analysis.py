import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "reports" / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_processing import load_raw_data, preprocess_pipeline
from src.visualization import (
    plot_class_balance,
    plot_correlation_heatmap,
    plot_numeric_boxplots,
    plot_text_length_distribution,
    plot_text_length_by_class,
    plot_text_length_vs_word_count,
    plot_word_count_distribution,
)


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def write_data_quality_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame, path: str) -> None:
    raw_rows = len(raw_df)
    clean_rows = len(clean_df)
    empty_text = raw_df["text"].fillna("").str.strip().eq("").sum()
    dup_text = raw_df.duplicated(subset=["text"]).sum()

    class_counts = clean_df["breakup"].value_counts().to_dict()

    numeric_summary = clean_df[["text_length", "word_count"]].describe().round(2)
    numeric_md = dataframe_to_markdown(numeric_summary)

    lines = [
        "# Data Quality Report\n",
        f"- Raw rows: {raw_rows}\n",
        f"- Clean rows: {clean_rows}\n",
        f"- Empty text rows (raw): {empty_text}\n",
        f"- Duplicate text rows (raw): {dup_text}\n",
        "\n## Class Balance (cleaned)\n",
        f"- breakup=1: {class_counts.get(1, 0)}\n",
        f"- breakup=0: {class_counts.get(0, 0)}\n",
        "\n## Numeric Summary (cleaned)\n",
        numeric_md,
        "\n\n## Notes\n",
        "- Labels are proxy based on subreddit source (topic-level, not ground truth).\n",
        "- This analysis uses message content only; engagement signals are not used for modeling.\n",
    ]

    Path(path).write_text("".join(lines))


def save_confusion_matrix(cm: np.ndarray, title: str, path: str) -> None:
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def extract_top_terms(pipeline: Pipeline, top_n: int = 15) -> dict:
    """Extract top positive/negative terms from a TF-IDF + logistic regression pipeline."""
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]
    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_n]
    return {
        "positive": [(feature_names[i], float(coefs[i])) for i in top_pos_idx],
        "negative": [(feature_names[i], float(coefs[i])) for i in top_neg_idx],
    }


def save_top_terms_plot(top_terms: dict, path: str) -> None:
    """Save a bar chart of top positive/negative terms."""
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
    plt.savefig(path, dpi=150)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without external dependencies."""
    headers = ["stat"] + [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in df.iterrows():
        values = [str(idx)] + [f"{v:.2f}" for v in row.values]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def evaluate_models(df: pd.DataFrame) -> dict:
    X = df["text"]
    y = df["breakup"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "decision_tree": DecisionTreeClassifier(max_depth=20, random_state=42, class_weight="balanced"),
    }

    results = {}
    for name, model in models.items():
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        pipeline = Pipeline(steps=[("tfidf", vectorizer), ("model", model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        payload = {
            "metrics": metrics,
            "classification_report": classification_report(y_test, y_pred, digits=4),
        }

        if name == "logistic_regression":
            top_terms = extract_top_terms(pipeline, top_n=15)
            payload["top_terms"] = top_terms

        results[name] = payload

        cm_path = f"reports/figures/confusion_matrix_{name}.png"
        save_confusion_matrix(confusion_matrix(y_test, y_pred), f"Confusion Matrix: {name}", cm_path)

    return results


def write_metrics_report(results: dict, path_md: str, path_json: str) -> None:
    Path(path_json).write_text(json.dumps(results, indent=2))

    lines = ["# Model Results\n\n"]
    for name, payload in results.items():
        metrics = payload["metrics"]
        lines.append(f"## {name.replace('_', ' ').title()}\n")
        lines.append(f"- Accuracy: {metrics['accuracy']:.4f}\n")
        lines.append(f"- Precision: {metrics['precision']:.4f}\n")
        lines.append(f"- Recall: {metrics['recall']:.4f}\n")
        lines.append(f"- F1: {metrics['f1']:.4f}\n")
        lines.append(f"- ROC-AUC: {metrics['roc_auc']:.4f}\n")
        lines.append(f"- Confusion Matrix: {metrics['confusion_matrix']}\n\n")
        lines.append("Classification Report:\n\n```")
        lines.append(payload["classification_report"].strip())
        lines.append("```\n\n")

        if name == "logistic_regression" and "top_terms" in payload:
            lines.append("Top indicative terms (message content):\n\n")
            lines.append("Breakup-indicative terms:\n\n")
            for term, weight in payload["top_terms"]["positive"]:
                lines.append(f"- {term}: {weight:.4f}\n")
            lines.append("\nNon-breakup indicative terms:\n\n")
            for term, weight in payload["top_terms"]["negative"]:
                lines.append(f"- {term}: {weight:.4f}\n")
            lines.append("\n")

    Path(path_md).write_text("\n".join(lines))


def main() -> None:
    ensure_dirs("data/processed", "reports/figures", "reports/results")

    raw_df = load_raw_data(
        "data/raw/reddit_breakup_dataset_cleaned.csv",
        "data/raw/relationship_advice.csv",
    )
    clean_df = preprocess_pipeline(
        "data/raw/reddit_breakup_dataset_cleaned.csv",
        "data/raw/relationship_advice.csv",
    )

    clean_df.to_csv("data/processed/combined_cleaned.csv", index=False)

    write_data_quality_report(raw_df, clean_df, "reports/data_quality_report.md")

    plot_class_balance(clean_df, save_path="reports/figures/class_balance.png")
    plot_text_length_distribution(clean_df, save_path="reports/figures/text_length_distribution.png")
    plot_word_count_distribution(clean_df, save_path="reports/figures/word_count_distribution.png")
    plot_text_length_by_class(clean_df, save_path="reports/figures/text_length_by_class.png")
    plot_numeric_boxplots(clean_df, save_path="reports/figures/numeric_boxplots.png")
    plot_correlation_heatmap(clean_df, save_path="reports/figures/correlation_heatmap.png")
    plot_text_length_vs_word_count(clean_df, save_path="reports/figures/text_length_vs_word_count.png")

    results = evaluate_models(clean_df)
    write_metrics_report(results, "reports/results/metrics.md", "reports/results/metrics.json")

    if "logistic_regression" in results and "top_terms" in results["logistic_regression"]:
        save_top_terms_plot(
            results["logistic_regression"]["top_terms"],
            "reports/figures/top_terms_logistic_regression.png",
        )

    print("Reports generated:")
    print("- data/processed/combined_cleaned.csv")
    print("- reports/data_quality_report.md")
    print("- reports/figures/*.png")
    print("- reports/results/metrics.md")
    print("- reports/results/metrics.json")


if __name__ == "__main__":
    main()
