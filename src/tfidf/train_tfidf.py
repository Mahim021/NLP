"""Model A -- TF-IDF features + Logistic Regression classifier.

Pipeline:
    text -> TF-IDF -> Logistic Regression -> Yes/No

TF-IDF is the feature extractor; Logistic Regression is the classifier.

The TF-IDF vectorizer is fitted only on the training split.
Validation and test data are used only for evaluation.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.data import load_splits
from src.common.seeding import set_seed
from src.evaluation.metrics import (
    compute_metrics,
    print_metrics,
    save_results,
)


DEFAULTS = {
    "model_name": "tfidf_logreg",
    "seed": 42,
    "splits_dir": str(ROOT / "data" / "splits"),
    "results_dir": str(ROOT / "results"),
    "models_dir": str(ROOT / "models"),
    "lowercase": True,
    "ngram_range": [1, 2],
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
    "max_features": None,
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 2000,
}


def build_pipeline(cfg):
    """Create the TF-IDF + Logistic Regression pipeline."""

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=cfg["lowercase"],
                ngram_range=tuple(cfg["ngram_range"]),
                min_df=cfg["min_df"],
                max_df=cfg["max_df"],
                sublinear_tf=cfg["sublinear_tf"],
                max_features=cfg["max_features"],
                stop_words=None,
                strip_accents="unicode",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=cfg["C"],
                class_weight=cfg["class_weight"],
                max_iter=cfg["max_iter"],
                solver="liblinear",
                random_state=cfg["seed"],
            ),
        ),
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Train Model A: TF-IDF + Logistic Regression."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "tfidf.json",
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
    )

    parser.add_argument(
        "--models-dir",
        type=Path,
    )

    parser.add_argument(
        "--seed",
        type=int,
    )

    args = parser.parse_args()

 
    cfg = dict(DEFAULTS)

 
    if args.config.exists():
        with open(args.config, "r", encoding="utf-8") as file:
            cfg.update(json.load(file))

  
    if args.splits_dir is not None:
        cfg["splits_dir"] = str(args.splits_dir)

    if args.results_dir is not None:
        cfg["results_dir"] = str(args.results_dir)

    if args.models_dir is not None:
        cfg["models_dir"] = str(args.models_dir)

    if args.seed is not None:
        cfg["seed"] = args.seed
 
    set_seed(cfg["seed"])
 
    train, val, test = load_splits(cfg["splits_dir"])

    print(
        f"Train: {len(train)} | "
        f"Validation: {len(val)} | "
        f"Test: {len(test)}"
    )

 
    pipeline = build_pipeline(cfg)

 
    pipeline.fit(
        train["text"],
        train["label_id"],
    )

    vocabulary_size = len(
        pipeline.named_steps["tfidf"].vocabulary_
    )

    print(f"Vocabulary size: {vocabulary_size}")

 
    metrics = {}
    test_predictions = None

    for split_name, split_data in (
        ("val", val),
        ("test", test),
    ):
        probabilities = pipeline.predict_proba(
            split_data["text"]
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        metrics[split_name] = compute_metrics(
            split_data["label_id"],
            predictions,
            probabilities,
        )

        print_metrics(
            "Model A: TF-IDF + Logistic Regression",
            split_name,
            metrics[split_name],
        )

 
        if split_name == "test":
            test_predictions = pd.DataFrame({
                "id": split_data["id"],
                "text": split_data["text"],
                "true_label": split_data["label"],
                "pred_label": [
                    "Yes" if prediction == 1 else "No"
                    for prediction in predictions
                ],
                "prob_actionable": probabilities,
            })

 
    results_dir = Path(cfg["results_dir"])
    save_results(
        cfg["model_name"],
        results_dir,
        metrics,
        cfg,
        predictions=test_predictions,
    )

 
    models_dir = (
        Path(cfg["models_dir"])
        / cfg["model_name"]
    )

    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pipeline,
        models_dir / "pipeline.joblib",
    )

  
    with open(
        models_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cfg,
            file,
            indent=2,
        )

    print(
        f"Saved model to "
        f"{models_dir / 'pipeline.joblib'}"
    )


if __name__ == "__main__":
    main()