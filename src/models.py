"""Model training and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


@dataclass
class ModelResults:
    accuracy: float
    precision: float
    recall: float
    confusion: list


def build_preprocessor(text_col: str = "text", numeric_cols: Tuple[str, ...] = ("upvotes_num", "comments_num", "text_length", "word_count")) -> ColumnTransformer:
    """Create a preprocessor with TF-IDF for text and passthrough for numeric features."""
    return ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), text_col),
            ("num", "passthrough", list(numeric_cols)),
        ]
    )


def train_models(df: pd.DataFrame, label_col: str = "breakup") -> Dict[str, Pipeline]:
    """Train Logistic Regression and Decision Tree classifiers."""
    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    lr_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ]
    )

    dt_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", DecisionTreeClassifier(max_depth=20, random_state=42))
        ]
    )

    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)

    return {"logistic_regression": lr_model, "decision_tree": dt_model}


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResults:
    """Evaluate a trained model and return metrics."""
    y_pred = model.predict(X_test)
    return ModelResults(
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred),
        recall=recall_score(y_test, y_pred),
        confusion=confusion_matrix(y_test, y_pred).tolist(),
    )


def train_test_evaluate(df: pd.DataFrame, label_col: str = "breakup") -> Dict[str, ModelResults]:
    """Train models and evaluate on a held-out test set."""
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(max_depth=20, random_state=42),
    }

    results: Dict[str, ModelResults] = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        results[name] = evaluate_model(pipeline, X_test, y_test)

    return results
