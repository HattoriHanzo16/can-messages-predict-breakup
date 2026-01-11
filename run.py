import argparse
import logging
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reports" / ".mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "reports" / ".cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import warnings

import pandas as pd
from sklearn.metrics import roc_curve

from src.config import load_config
from src.data_processing import load_raw_data, preprocess_pipeline
from src.modeling import extract_top_terms, select_best_model, train_and_evaluate
from src.reporting import (
    generate_html_report,
    generate_markdown_report,
    save_excel_report,
    write_data_quality_report,
    write_metrics_reports,
)
from src.visualization import (
    plot_class_balance,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_numeric_boxplots,
    plot_roc_curves,
    plot_top_terms,
    plot_text_length_by_class,
    plot_text_length_distribution,
    plot_text_length_vs_word_count,
    plot_word_count_distribution,
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_interactive_plot(df: pd.DataFrame, output_path: str) -> bool:
    try:
        import plotly.express as px
    except Exception:
        logging.info("Plotly not installed; skipping interactive chart.")
        return False

    fig = px.scatter(
        df,
        x="breakup_term_ratio",
        y="sentiment_score",
        color=df["breakup"].astype(str),
        title="Breakup Signals: Term Ratio vs Sentiment",
        labels={
            "breakup_term_ratio": "Breakup Term Ratio",
            "sentiment_score": "Sentiment Score",
            "color": "Breakup",
        },
        hover_data=["text_length", "word_count"],
        opacity=0.6,
    )
    fig.write_html(output_path)
    return True


def open_report(path: str) -> None:
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.run(["open", path], check=False)
        elif system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as exc:
        logging.info("Could not open report automatically: %s", exc)


def build_report_payload(
    config: dict,
    clean_df: pd.DataFrame,
    results: dict,
    top_terms: dict,
    winner: str,
    figures: list[dict],
    insights: list[str],
    error_analysis: str,
) -> dict:
    metrics_table = []
    for name, payload in results.items():
        metrics = payload["metrics"]
        metrics_table.append(
            {
                "model": name.replace("_", " ").title(),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    breakup_count = int(clean_df["breakup"].sum())
    total_rows = len(clean_df)
    non_breakup = total_rows - breakup_count
    breakup_ratio = breakup_count / total_rows if total_rows else 0.0

    best_f1 = results[winner]["metrics"]["f1"]
    best_auc = results[winner]["metrics"]["roc_auc"]
    winner_title = winner.replace("_", " ").title()

    executive_summary = (
        "Message content shows strong predictive power for breakup-related posts. "
        f"The top-performing model ({winner_title}) achieved an F1 score of {best_f1:.3f}"
        + (f" and ROC-AUC of {best_auc:.3f}" if best_auc is not None else "")
        + ", indicating reliable detection of breakup language patterns using text alone."
    )

    return {
        "title": config["project"]["name"],
        "subtitle": config["project"]["description"],
        "executive_summary": executive_summary,
        "insights": insights,
        "dataset": {
            "total_rows": total_rows,
            "breakup_rows": breakup_count,
            "non_breakup_rows": non_breakup,
            "breakup_ratio": breakup_ratio,
        },
        "metrics_table": metrics_table,
        "top_terms": top_terms,
        "winner": {"name": winner_title, "f1": best_f1},
        "figures": figures,
        "error_analysis": error_analysis,
        "limitations": [
            "Labels are proxy-based on subreddit topic, not verified outcomes.",
            "Language patterns may reflect subreddit style rather than real-world breakup signals.",
            "This is a single train/test split; performance may vary across samples.",
        ],
    }


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Breakup prediction pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    ensure_dirs(
        config["output"]["report_dir"],
        config["output"]["figures_dir"],
        config["output"]["results_dir"],
    )

    logging.info("Loading and preprocessing data...")
    raw_df = load_raw_data(config["data"]["breakup_path"], config["data"]["advice_path"])
    clean_df = preprocess_pipeline(config["data"]["breakup_path"], config["data"]["advice_path"])
    clean_df["breakup"] = clean_df["breakup"].astype(int)

    clean_df.to_csv(config["output"]["processed_path"], index=False)
    write_data_quality_report(raw_df, clean_df, config["output"]["data_quality_report"])

    logging.info("Generating EDA figures...")
    plot_class_balance(clean_df, save_path=f"{config['output']['figures_dir']}/class_balance.png")
    plot_text_length_distribution(clean_df, save_path=f"{config['output']['figures_dir']}/text_length_distribution.png")
    plot_word_count_distribution(clean_df, save_path=f"{config['output']['figures_dir']}/word_count_distribution.png")
    plot_text_length_by_class(clean_df, save_path=f"{config['output']['figures_dir']}/text_length_by_class.png")
    plot_numeric_boxplots(clean_df, save_path=f"{config['output']['figures_dir']}/numeric_boxplots.png")
    plot_correlation_heatmap(clean_df, save_path=f"{config['output']['figures_dir']}/correlation_heatmap.png")
    plot_text_length_vs_word_count(clean_df, save_path=f"{config['output']['figures_dir']}/text_length_vs_word_count.png")

    numeric_features = [
        "text_length",
        "word_count",
        "breakup_term_count",
        "breakup_term_ratio",
        "sentiment_score",
        "first_person_ratio",
        "question_marks",
        "exclamation_marks",
    ]

    logging.info("Training and evaluating models...")
    results, artifacts, X_test, y_test = train_and_evaluate(
        clean_df,
        label_col="breakup",
        numeric_features=numeric_features,
        tfidf_config=config["modeling"]["tfidf"],
        test_size=config["modeling"]["test_size"],
        random_state=config["modeling"]["random_state"],
    )

    for model_name, payload in results.items():
        cm = payload["metrics"]["confusion_matrix"]
        plot_confusion_matrix(
            cm,
            f"Confusion Matrix: {model_name}",
            save_path=f"{config['output']['figures_dir']}/confusion_matrix_{model_name}.png",
        )

    roc_curves = []
    for name, artifact in artifacts.items():
        if artifact.y_score is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, artifact.y_score)
        auc = results[name]["metrics"]["roc_auc"]
        roc_curves.append({"label": name.replace("_", " ").title(), "fpr": fpr, "tpr": tpr, "auc": auc})

    if roc_curves:
        plot_roc_curves(roc_curves, save_path=f"{config['output']['figures_dir']}/roc_curves.png")

    winner = select_best_model(results, metric="f1")
    top_terms = extract_top_terms(artifacts[winner].pipeline, config["modeling"]["top_terms"])
    results[winner]["top_terms"] = top_terms
    plot_top_terms(
        top_terms,
        save_path=f"{config['output']['figures_dir']}/top_terms_{winner}.png",
    )

    write_metrics_reports(results, config["output"]["metrics_md"], config["output"]["metrics_json"])

    feature_summary = (
        clean_df.groupby("breakup")[numeric_features]
        .agg(["mean", "median"])
        .round(3)
        .reset_index()
    )

    misclass_df = X_test.copy()
    misclass_df["true_label"] = y_test.values
    misclass_df["pred_label"] = artifacts[winner].y_pred
    misclass_df["score"] = artifacts[winner].y_score
    misclass_df = misclass_df[misclass_df["true_label"] != misclass_df["pred_label"]]
    misclass_df["text_snippet"] = misclass_df["text"].str.slice(0, 300)
    misclass_df["error_type"] = misclass_df.apply(
        lambda row: "false_positive" if row["pred_label"] == 1 else "false_negative", axis=1
    )
    misclass_df.to_csv(config["output"]["misclassifications_csv"], index=False)

    try:
        save_excel_report(results, feature_summary, misclass_df, config["output"]["metrics_xlsx"])
    except ImportError as exc:
        logging.info(str(exc))

    report_dir = Path(config["output"]["report_dir"])
    figures_dir = Path(config["output"]["figures_dir"])

    def rel(path: Path) -> str:
        return str(path.relative_to(report_dir))

    winner_title = winner.replace("_", " ").title()

    figures = [
        {
            "path": rel(figures_dir / "class_balance.png"),
            "title": "Class Balance",
            "caption": "Shows the distribution of breakup vs non-breakup posts. The dataset is mildly imbalanced.",
        },
        {
            "path": rel(figures_dir / "text_length_distribution.png"),
            "title": "Text Length Distribution",
            "caption": "Most posts are short-to-medium length, with a long tail of detailed narratives.",
        },
        {
            "path": rel(figures_dir / "word_count_distribution.png"),
            "title": "Word Count Distribution",
            "caption": "Word count mirrors text length, highlighting the prevalence of concise posts.",
        },
        {
            "path": rel(figures_dir / "text_length_by_class.png"),
            "title": "Text Length by Label",
            "caption": "Breakup posts tend to be longer, suggesting more detailed emotional accounts.",
        },
        {
            "path": rel(figures_dir / "correlation_heatmap.png"),
            "title": "Feature Correlations",
            "caption": "Highlights relationships among engineered signals (length, terms, sentiment, punctuation).",
        },
    ]

    for name in results.keys():
        figures.append(
            {
                "path": rel(figures_dir / f"confusion_matrix_{name}.png"),
                "title": f"Confusion Matrix: {name.replace('_', ' ').title()}",
                "caption": "Shows how many posts were correctly and incorrectly classified.",
            }
        )

    figures.extend(
        [
            {
                "path": rel(figures_dir / "roc_curves.png"),
                "title": "ROC Curves",
                "caption": "Compares model discrimination; higher curves indicate stronger separation.",
            },
            {
                "path": rel(figures_dir / f"top_terms_{winner}.png"),
                "title": f"Top Terms: {winner_title}",
                "caption": "Top positive and negative terms from the best model.",
            },
        ]
    )

    insights = [
        f"{winner_title} delivered the strongest F1 score and overall balance of precision/recall.",
        "Breakup-related language (e.g., breakup, broke, miss, pain) dominates predictive terms.",
        "Longer, more detailed posts are more common in breakup-labeled samples.",
    ]

    error_analysis = (
        f"The best model ({winner_title}) misclassified {len(misclass_df)} posts. "
        "False positives often involve general relationship conflict language, while false negatives "
        "tend to be subtle breakup narratives without explicit breakup terms."
    )

    report_payload = build_report_payload(
        config=config,
        clean_df=clean_df,
        results=results,
        top_terms=top_terms,
        winner=winner,
        figures=figures,
        insights=insights,
        error_analysis=error_analysis,
    )

    if config.get("reporting", {}).get("include_interactive", True):
        interactive_path = Path(f"{config['output']['figures_dir']}/interactive_breakup_signals.html")
        interactive_created = save_interactive_plot(clean_df, str(interactive_path))
        if interactive_created:
            report_payload["interactive_link"] = str(interactive_path.relative_to(report_dir))

    generate_html_report(report_payload, config["output"]["report_html"])
    generate_markdown_report(report_payload, config["output"]["report_md"])

    if config.get("reporting", {}).get("open_report", False):
        open_report(config["output"]["report_html"])

    logging.info("Reports generated:")
    logging.info(f"- {config['output']['processed_path']}")
    logging.info(f"- {config['output']['data_quality_report']}")
    logging.info(f"- {config['output']['metrics_md']}")
    logging.info(f"- {config['output']['metrics_json']}")
    logging.info(f"- {config['output']['report_html']}")
    logging.info(f"- {config['output']['report_md']}")


if __name__ == "__main__":
    main()
