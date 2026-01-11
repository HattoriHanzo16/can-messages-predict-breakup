import argparse
import logging
import math
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from joblib import load

from src.config import load_config
from src.data_processing import add_text_features, clean_text


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def build_app(config: dict) -> Flask:
    app = Flask(__name__, static_folder="demo", static_url_path="")

    model_path = config["output"]["model_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python run.py --config config/config.yaml"
        )

    model_bundle = load(model_path)
    pipeline = model_bundle["pipeline"]
    metadata = model_bundle["metadata"]
    numeric_features = metadata["numeric_features"]
    model_name = metadata["model_name"].replace("_", " ").title()

    @app.route("/")
    def index():
        return send_from_directory("demo", "breakup_simulator.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "model": model_name})

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(silent=True) or {}
        text_a = payload.get("messages_a", "")
        text_b = payload.get("messages_b", "")
        combined = f"{text_a}\n{text_b}".strip()

        df = pd.DataFrame({"text": [combined]})
        df = clean_text(df)
        df = add_text_features(df)
        df = df[["text"] + numeric_features]

        y_pred = int(pipeline.predict(df)[0])

        if hasattr(pipeline, "predict_proba"):
            prob = float(pipeline.predict_proba(df)[:, 1][0])
        elif hasattr(pipeline, "decision_function"):
            score = float(pipeline.decision_function(df)[0])
            prob = sigmoid(score)
        else:
            prob = None

        signals = {
            "word_count": float(df["word_count"].iloc[0]),
            "breakup_term_ratio": float(df["breakup_term_ratio"].iloc[0]),
            "sentiment_score": float(df["sentiment_score"].iloc[0]),
            "first_person_ratio": float(df["first_person_ratio"].iloc[0]),
        }

        return jsonify(
            {
                "prediction": y_pred,
                "probability": prob,
                "model": model_name,
                "signals": signals,
            }
        )

    return app


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run breakup prediction demo server")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    config = load_config(args.config)
    app = build_app(config)
    logging.info("Starting server at http://%s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
