"""Reporting utilities for markdown, HTML, and Excel outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = ["stat"] + [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for idx, row in df.iterrows():
        values = [str(idx)] + [f"{v:.2f}" for v in row.values]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_data_quality_report(raw_df: pd.DataFrame, clean_df: pd.DataFrame, path: str) -> None:
    raw_rows = len(raw_df)
    clean_rows = len(clean_df)
    empty_text = raw_df["text"].fillna("").str.strip().eq("").sum()
    dup_text = raw_df.duplicated(subset=["text"]).sum()
    class_counts = clean_df["breakup"].value_counts().to_dict()

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
    numeric_summary = clean_df[numeric_cols].describe().round(2)
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
        "- Models use message content only; no engagement metadata is used.\n",
    ]

    Path(path).write_text("".join(lines))


def write_metrics_reports(results: Dict[str, dict], path_md: str, path_json: str) -> None:
    Path(path_json).write_text(json.dumps(results, indent=2))

    lines = ["# Model Results\n\n"]
    for name, payload in results.items():
        metrics = payload["metrics"]
        lines.append(f"## {name.replace('_', ' ').title()}\n")
        lines.append(f"- Accuracy: {metrics['accuracy']:.4f}\n")
        lines.append(f"- Precision: {metrics['precision']:.4f}\n")
        lines.append(f"- Recall: {metrics['recall']:.4f}\n")
        lines.append(f"- F1: {metrics['f1']:.4f}\n")
        if metrics["roc_auc"] is None:
            lines.append("- ROC-AUC: N/A\n")
        else:
            lines.append(f"- ROC-AUC: {metrics['roc_auc']:.4f}\n")
        lines.append(f"- Confusion Matrix: {metrics['confusion_matrix']}\n\n")
        lines.append("Classification Report:\n\n```")
        lines.append(payload["classification_report"].strip())
        lines.append("```\n\n")

        if "top_terms" in payload:
            lines.append("Top indicative terms (message content):\n\n")
            lines.append("Breakup-indicative terms:\n\n")
            for term, weight in payload["top_terms"]["positive"]:
                lines.append(f"- {term}: {weight:.4f}\n")
            lines.append("\nNon-breakup indicative terms:\n\n")
            for term, weight in payload["top_terms"]["negative"]:
                lines.append(f"- {term}: {weight:.4f}\n")
            lines.append("\n")

    Path(path_md).write_text("\n".join(lines))


def save_excel_report(
    results: Dict[str, dict],
    summary_df: pd.DataFrame,
    misclass_df: pd.DataFrame,
    path: str,
) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError as exc:
        raise ImportError("openpyxl is required for Excel export. Install with: pip install openpyxl") from exc

    metrics_rows = []
    for name, payload in results.items():
        metrics = payload["metrics"]
        metrics_rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )

    metrics_df = pd.DataFrame(metrics_rows)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        summary_df.to_excel(writer, sheet_name="feature_summary")
        misclass_df.to_excel(writer, sheet_name="misclassifications", index=False)


def generate_html_report(report: dict, output_path: str) -> None:
    metrics_rows = "".join(
        f"<tr><td>{row['model']}</td><td>{row['accuracy']:.3f}</td>"
        f"<td>{row['precision']:.3f}</td><td>{row['recall']:.3f}</td>"
        f"<td>{row['f1']:.3f}</td><td>{row['roc_auc'] if row['roc_auc'] is not None else 'N/A'}</td></tr>"
        for row in report["metrics_table"]
    )

    top_terms_html = "".join(
        f"<li><strong>{term}</strong>: {weight:.3f}</li>" for term, weight in report["top_terms"]["positive"]
    )

    top_terms_neg_html = "".join(
        f"<li><strong>{term}</strong>: {weight:.3f}</li>" for term, weight in report["top_terms"]["negative"]
    )

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{report['title']}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1a1a1a; }}
    h1, h2 {{ color: #102a43; }}
    .section {{ margin-bottom: 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #e0e0e0; padding: 12px; border-radius: 8px; background: #fafafa; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e0e0e0; padding: 6px 8px; font-size: 14px; }}
    th {{ background: #f0f4f8; text-align: left; }}
    .muted {{ color: #52606d; }}
    .figure {{ width: 100%; max-width: 700px; }}
  </style>
</head>
<body>
  <h1>{report['title']}</h1>
  <p class="muted">{report['subtitle']}</p>

  <div class="section">
    <h2>Executive Summary</h2>
    <p>{report['executive_summary']}</p>
  </div>

  <div class="section grid">
    <div class="card">
      <h3>Dataset</h3>
      <ul>
        <li>Total rows: {report['dataset']['total_rows']}</li>
        <li>Breakup posts: {report['dataset']['breakup_rows']}</li>
        <li>Non-breakup posts: {report['dataset']['non_breakup_rows']}</li>
      </ul>
    </div>
    <div class="card">
      <h3>Model Winner</h3>
      <p>{report['winner']['name']} (F1: {report['winner']['f1']:.3f})</p>
      <p class="muted">Chosen by best F1 score.</p>
    </div>
  </div>

  <div class="section">
    <h2>Model Performance</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
          <th>ROC-AUC</th>
        </tr>
      </thead>
      <tbody>
        {metrics_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Top Predictive Terms (Breakup vs Non-breakup)</h2>
    <div class="grid">
      <div class="card">
        <h4>Breakup-indicative</h4>
        <ul>{top_terms_html}</ul>
      </div>
      <div class="card">
        <h4>Non-breakup indicative</h4>
        <ul>{top_terms_neg_html}</ul>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Key Figures</h2>
    <div class="grid">
      {''.join(f'<img class="figure" src="{fig}" alt="{fig}" />' for fig in report['figures'])}
    </div>
  </div>

  <div class="section">
    <h2>Error Analysis</h2>
    <p>{report['error_analysis']}</p>
  </div>

  <div class="section">
    <h2>Limitations</h2>
    <ul>
      {''.join(f'<li>{item}</li>' for item in report['limitations'])}
    </ul>
  </div>

</body>
</html>
"""

    Path(output_path).write_text(html)


def generate_markdown_report(report: dict, output_path: str) -> None:
    metrics_rows = []
    for row in report["metrics_table"]:
        roc_auc = "N/A" if row["roc_auc"] is None else f"{row['roc_auc']:.3f}"
        metrics_rows.append(
            f"| {row['model']} | {row['accuracy']:.3f} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {roc_auc} |"
        )
    metrics_table = "\n".join(
        [
            "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        + metrics_rows
    )

    lines = [
        f"# {report['title']}",
        report["subtitle"],
        "",
        "## Executive Summary",
        report["executive_summary"],
        "",
        "## Dataset",
        f"- Total rows: {report['dataset']['total_rows']}",
        f"- Breakup posts: {report['dataset']['breakup_rows']}",
        f"- Non-breakup posts: {report['dataset']['non_breakup_rows']}",
        "",
        "## Model Performance",
        metrics_table,
        "",
        "## Top Predictive Terms (Breakup)",
        "\n".join(f"- {term}: {weight:.3f}" for term, weight in report["top_terms"]["positive"]),
        "",
        "## Top Predictive Terms (Non-breakup)",
        "\n".join(f"- {term}: {weight:.3f}" for term, weight in report["top_terms"]["negative"]),
        "",
        "## Error Analysis",
        report["error_analysis"],
        "",
        "## Limitations",
        "\n".join(f"- {item}" for item in report["limitations"]),
    ]

    Path(output_path).write_text("\n".join(lines))
