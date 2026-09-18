"""Model A -- TF-IDF features + Logistic Regression classifier.

    text -> preprocessing -> TF-IDF -> Logistic Regression -> Yes/No

TF-IDF is the feature extractor; Logistic Regression is the model.
The vectoriser is fitted on the TRAINING split only -- never on val or test --
so no test information leaks into the vocabulary or the idf weights.

    python -m src.tfidf.train_tfidf --config configs/tfidf.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.common.data import load_splits  # noqa: E402
from src.common.seeding import set_seed  # noqa: E402
from src.evaluation.metrics import compute_metrics, print_metrics, save_results  # noqa: E402

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
    "tune": True,
    "param_grid": {
        "tfidf__ngram_range": [[1, 1], [1, 2]],
        "tfidf__min_df": [1, 2],
        "clf__C": [0.25, 1.0, 4.0, 16.0],
    },
}


def build_pipeline(cfg: dict) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=cfg["lowercase"],
            ngram_range=tuple(cfg["ngram_range"]),
            min_df=cfg["min_df"],
            max_df=cfg["max_df"],
            sublinear_tf=cfg["sublinear_tf"],
            max_features=cfg["max_features"],
            # No stop-word list: "please", "can you", "by tomorrow" are signal.
            stop_words=None,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            class_weight=cfg["class_weight"],
            max_iter=cfg["max_iter"],
            solver="liblinear",
            random_state=cfg["seed"],
        )),
    ])


def tune(cfg: dict, train: pd.DataFrame) -> dict:
    """Select hyper-parameters by cross-validation on the TRAINING split only.

    The test split is not touched here, so model selection cannot see it.  The
    validation split is left out of tuning as well, which keeps it an honest
    held-out set for the same purpose the LSTM and BERT runs use it for.
    """
    grid = {k: [tuple(v) if isinstance(v, list) else v for v in vals]
            for k, vals in cfg["param_grid"].items()}
    dev = train
    search = GridSearchCV(
        build_pipeline(cfg), grid, scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"]),
        n_jobs=-1, refit=False)
    search.fit(dev["text"], dev["label_id"])
    best = dict(search.best_params_)
    print(f"Best CV F1 = {search.best_score_:.4f} with {best}")
    out = dict(cfg)
    if "tfidf__ngram_range" in best:
        out["ngram_range"] = list(best["tfidf__ngram_range"])
    if "tfidf__min_df" in best:
        out["min_df"] = best["tfidf__min_df"]
    if "clf__C" in best:
        out["C"] = best["clf__C"]
    out["cv_best_f1"] = float(search.best_score_)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Model A (TF-IDF + Logistic Regression).")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "tfidf.json")
    ap.add_argument("--splits-dir", type=Path)
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--models-dir", type=Path)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no-tune", action="store_true", help="skip cross-validated tuning")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    if args.config and Path(args.config).exists():
        cfg.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key in ("splits_dir", "results_dir", "models_dir", "seed"):
        if getattr(args, key) is not None:
            cfg[key] = str(getattr(args, key)) if key.endswith("_dir") else getattr(args, key)
    if args.no_tune:
        cfg["tune"] = False

    set_seed(cfg["seed"])
    train, val, test = load_splits(cfg["splits_dir"])
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    if cfg.get("tune"):
        cfg = tune(cfg, train)

    # Final fit on the training split only: identical training data to the LSTM
    # and BERT runs, and it keeps val a genuine held-out set for all three.
    pipe = build_pipeline(cfg)
    pipe.fit(train["text"], train["label_id"])
    print(f"Vocabulary size: {len(pipe.named_steps['tfidf'].vocabulary_)}")

    metrics = {}
    preds_df = None
    for split, df in (("val", val), ("test", test)):
        prob = pipe.predict_proba(df["text"])[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics[split] = compute_metrics(df["label_id"], pred, prob)
        print_metrics("Model A: TF-IDF + Logistic Regression", split, metrics[split])
        if split == "test":
            preds_df = pd.DataFrame({
                "id": df["id"], "text": df["text"], "true_label": df["label"],
                "pred_label": ["Yes" if p else "No" for p in pred],
                "prob_actionable": prob,
            })

    save_results(cfg["model_name"], Path(cfg["results_dir"]), metrics, cfg,
                 predictions=preds_df)

    models_dir = Path(cfg["models_dir"]) / cfg["model_name"]
    models_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(pipe, models_dir / "pipeline.joblib")
    (models_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Saved model to {models_dir / 'pipeline.joblib'}")


if __name__ == "__main__":
    main()
