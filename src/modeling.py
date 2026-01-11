"""Model training, evaluation, and interpretability helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class ModelArtifacts:
    pipeline: Pipeline
    y_pred: np.ndarray
    y_score: np.ndarray | None


def build_preprocessor(tfidf_config: dict, numeric_features: List[str]) -> ColumnTransformer:
    tfidf = TfidfVectorizer(
        max_features=tfidf_config.get("max_features", 5000),
        ngram_range=tuple(tfidf_config.get("ngram_range", [1, 2])),
        stop_words=tfidf_config.get("stop_words", "english"),
    )
    scaler = StandardScaler(with_mean=False)
    return ColumnTransformer(
        transformers=[
            ("text", tfidf, "text"),
            ("num", scaler, numeric_features),
        ]
    )


def build_models() -> Dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        "decision_tree": DecisionTreeClassifier(max_depth=20, random_state=42, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced", random_state=42, max_iter=5000, dual="auto"),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    label_col: str,
    numeric_features: List[str],
    tfidf_config: dict,
    test_size: float,
    random_state: int,
) -> Tuple[Dict[str, dict], Dict[str, ModelArtifacts], pd.Series, pd.Series]:
    X = df[["text"] + numeric_features]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = build_preprocessor(tfidf_config, numeric_features)
    models = build_models()

    results: Dict[str, dict] = {}
    artifacts: Dict[str, ModelArtifacts] = {}

    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline, "decision_function"):
            y_score = pipeline.decision_function(X_test)
        else:
            y_score = None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_score)) if y_score is not None else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        results[name] = {
            "metrics": metrics,
            "classification_report": classification_report(y_test, y_pred, digits=4),
        }
        artifacts[name] = ModelArtifacts(pipeline=pipeline, y_pred=y_pred, y_score=y_score)

    return results, artifacts, X_test, y_test


def select_best_model(results: Dict[str, dict], metric: str = "f1") -> str:
    best_name = ""
    best_value = -1.0
    for name, payload in results.items():
        value = payload["metrics"].get(metric)
        if value is not None and value > best_value:
            best_name = name
            best_value = value
    return best_name


def extract_top_terms(pipeline: Pipeline, top_n: int) -> dict:
    preprocessor = pipeline.named_steps["preprocess"]
    vectorizer = preprocessor.named_transformers_["text"]
    model = pipeline.named_steps["model"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0][: len(feature_names)]

    top_pos_idx = np.argsort(coefs)[-top_n:][::-1]
    top_neg_idx = np.argsort(coefs)[:top_n]

    return {
        "positive": [(feature_names[i], float(coefs[i])) for i in top_pos_idx],
        "negative": [(feature_names[i], float(coefs[i])) for i in top_neg_idx],
    }
