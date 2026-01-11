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

    insights_html = "".join(f"<li>{item}</li>" for item in report["insights"])
    limitations_html = "".join(f"<li>{item}</li>" for item in report["limitations"])

    figures_html = "".join(
        f"""
        <div class=\"figure-card\">
          <button class=\"figure-button\" type=\"button\" data-full=\"{fig['path']}\">
            <img class=\"figure\" src=\"{fig['path']}\" alt=\"{fig['title']}\" />
          </button>
          <div class=\"figure-caption\">
            <h4>{fig['title']}</h4>
            <p>{fig['caption']}</p>
          </div>
        </div>
        """
        for fig in report["figures"]
    )

    interactive_html = ""
    if report.get("interactive_link"):
        interactive_html = f"""
        <div class=\"card\">
          <h3>Interactive Exploration</h3>
          <p class=\"muted\">Open the interactive scatter plot to explore breakup term density vs sentiment.</p>
          <a class=\"button\" href=\"{report['interactive_link']}\">Open Interactive Chart</a>
        </div>
        """

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{report['title']}</title>
  <style>
    :root {{
      --ink: #0f172a;
      --muted: #475569;
      --accent: #0b7285;
      --accent-soft: #e6f6f8;
      --bg: #f8fafc;
      --card: #ffffff;
      --border: #e2e8f0;
    }}
    body {{
      font-family: \"IBM Plex Serif\", \"Source Serif 4\", \"Merriweather\", \"Georgia\", serif;
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      line-height: 1.6;
    }}
    header {{
      background: linear-gradient(120deg, #0b7285, #1971c2);
      color: #ffffff;
      padding: 36px 40px;
    }}
    header h1 {{
      margin: 0 0 8px 0;
      font-size: 2.2rem;
      letter-spacing: 0.3px;
    }}
    header p {{
      margin: 0;
      opacity: 0.9;
    }}
    main {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px 32px 64px;
    }}
    .section {{
      margin: 28px 0;
    }}
    .section h2 {{
      margin-bottom: 12px;
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }}
    .card {{
      border: 1px solid var(--border);
      padding: 16px;
      border-radius: 12px;
      background: var(--card);
      box-shadow: 0 8px 16px rgba(15, 23, 42, 0.05);
    }}
    .answer-card {{
      background: linear-gradient(135deg, #e6fcf5, #e3fafc);
      border: 1px solid #99e9f2;
    }}
    .button {{
      display: inline-block;
      background: var(--accent);
      color: #ffffff;
      padding: 8px 14px;
      border-radius: 999px;
      text-decoration: none;
      font-weight: 600;
      font-size: 14px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 600;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 10px 12px;
      font-size: 14px;
      text-align: left;
    }}
    th {{
      background: #eef2ff;
      color: #1e293b;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
    }}
    .figure-card {{
      background: var(--card);
      border-radius: 16px;
      border: 1px solid var(--border);
      overflow: hidden;
      box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
    }}
    .figure-button {{
      border: none;
      padding: 0;
      background: transparent;
      cursor: zoom-in;
      width: 100%;
    }}
    .figure {{
      width: 100%;
      display: block;
    }}
    .figure-caption {{
      padding: 12px 14px;
    }}
    .figure-caption h4 {{
      margin: 0 0 6px 0;
      color: var(--accent);
    }}
    .muted {{
      color: var(--muted);
    }}
    ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .lightbox {{
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.7);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 24px;
      z-index: 999;
      opacity: 0;
      pointer-events: none;
      transition: opacity 0.2s ease;
    }}
    .lightbox.open {{
      opacity: 1;
      pointer-events: auto;
    }}
    .lightbox-content {{
      background: #ffffff;
      border-radius: 16px;
      padding: 16px;
      max-width: 960px;
      width: 100%;
      box-shadow: 0 18px 40px rgba(15, 23, 42, 0.3);
    }}
    .lightbox-content img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 12px;
    }}
    .lightbox-close {{
      margin-top: 12px;
      background: var(--accent);
      color: #ffffff;
      border: none;
      padding: 8px 16px;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <header>
    <h1>{report['title']}</h1>
    <p>{report['subtitle']}</p>
  </header>

  <main>
    <section class=\"section\">
      <div class=\"card answer-card\">
        <span class=\"badge\">Answer</span>
        <h2>{report['answer_label']}: Can messages predict breakup?</h2>
        <p>{report['answer_statement']}</p>
        <ul>
          {''.join(f'<li>{item}</li>' for item in report['evidence'])}
        </ul>
      </div>
    </section>
    <section class=\"section\">
      <h2>Executive Summary</h2>
      <p>{report['executive_summary']}</p>
    </section>

    <section class=\"section grid\">
      <div class=\"card\">
        <span class=\"badge\">Dataset</span>
        <h3>Overview</h3>
        <ul>
          <li>Total rows: {report['dataset']['total_rows']}</li>
          <li>Breakup posts: {report['dataset']['breakup_rows']}</li>
          <li>Non-breakup posts: {report['dataset']['non_breakup_rows']}</li>
          <li>Breakup share: {report['dataset']['breakup_ratio']:.1%}</li>
        </ul>
      </div>
      <div class=\"card\">
        <span class=\"badge\">Top Model</span>
        <h3>{report['winner']['name']}</h3>
        <p class=\"muted\">Best F1 score: {report['winner']['f1']:.3f}</p>
        <p class=\"muted\">Selected by balanced precision and recall.</p>
      </div>
      <div class=\"card\">
        <span class=\"badge\">Key Insights</span>
        <ul>{insights_html}</ul>
      </div>
      {interactive_html}
    </section>

    <section class=\"section\">
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
    </section>

    <section class=\"section\">
      <h2>Top Predictive Terms</h2>
      <div class=\"grid\">
        <div class=\"card\">
          <h3>Breakup-indicative</h3>
          <ul>{top_terms_html}</ul>
        </div>
        <div class=\"card\">
          <h3>Non-breakup indicative</h3>
          <ul>{top_terms_neg_html}</ul>
        </div>
      </div>
    </section>

    <section class=\"section\">
      <h2>Figures & Explanations</h2>
      <div class=\"figure-grid\">
        {figures_html}
      </div>
    </section>

    <section class=\"section\">
      <h2>Error Analysis</h2>
      <p>{report['error_analysis']}</p>
    </section>

    <section class=\"section\">
      <h2>Limitations</h2>
      <ul>
        {limitations_html}
      </ul>
    </section>
  </main>
  <div class=\"lightbox\" id=\"lightbox\">
    <div class=\"lightbox-content\">
      <img id=\"lightbox-img\" src=\"\" alt=\"Expanded figure\" />
      <button class=\"lightbox-close\" id=\"lightbox-close\" type=\"button\">Close</button>
    </div>
  </div>
  <script>
    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    const closeBtn = document.getElementById('lightbox-close');
    document.querySelectorAll('.figure-button').forEach(btn => {{
      btn.addEventListener('click', () => {{
        lightboxImg.src = btn.dataset.full;
        lightbox.classList.add('open');
      }});
    }});
    const closeLightbox = () => lightbox.classList.remove('open');
    closeBtn.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', (event) => {{
      if (event.target === lightbox) {{
        closeLightbox();
      }}
    }});
  </script>
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
        "## Answer to the Research Question",
        f"{report['answer_label']}: Can messages predict breakup?",
        report["answer_statement"],
        "",
        "Evidence:",
        "\n".join(f"- {item}" for item in report["evidence"]),
        "",
        "## Executive Summary",
        report["executive_summary"],
        "",
        "## Key Insights",
        "\n".join(f"- {item}" for item in report["insights"]),
        "",
        "## Dataset",
        f"- Total rows: {report['dataset']['total_rows']}",
        f"- Breakup posts: {report['dataset']['breakup_rows']}",
        f"- Non-breakup posts: {report['dataset']['non_breakup_rows']}",
        f"- Breakup share: {report['dataset']['breakup_ratio']:.1%}",
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
