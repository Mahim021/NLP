"""Collect every model's saved test metrics into one comparison table.

    python -m src.evaluation.compare

Reads results/<model>/metrics.json for each model that has been trained and
writes results/comparison.csv, results/comparison.md and the confusion-matrix
figures used in the report.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.plots import plot_confusion_matrix  # noqa: E402

DISPLAY = {
    "tfidf_logreg": "Model A: TF-IDF + Logistic Regression",
    "lstm": "Model B: LSTM",
    "bert": "Model C: BERT",
}
ORDER = ["tfidf_logreg", "lstm", "bert"]


def collect(results_dir: Path, split: str = "test") -> pd.DataFrame:
    rows = []
    for name in ORDER + sorted(p.name for p in results_dir.iterdir()
                               if p.is_dir() and p.name not in ORDER):
        path = results_dir / name / "metrics.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        m = payload["metrics"].get(split)
        if m is None:
            continue
        rows.append({
            "model": DISPLAY.get(name, name),
            "key": name,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "roc_auc": m.get("roc_auc"),
            "TP": m["true_positives"],
            "TN": m["true_negatives"],
            "FP": m["false_positives"],
            "FN": m["false_negatives"],
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, split: str) -> str:
    head = (f"# MailSense -- model comparison ({split} set)\n\n"
            "Positive class = **Actionable (Yes)**. "
            "FN = an actionable e-mail predicted non-actionable "
            "(the costly error for this application).\n\n")
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc",
            "TP", "TN", "FP", "FN"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append("-" if pd.isna(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return head + "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare all trained MailSense models.")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--split", type=str, default="test", choices=["val", "test"])
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    df = collect(args.results_dir, args.split)
    if df.empty:
        print("No results found. Train at least one model first.")
        return

    out = df.drop(columns=["key"])
    out.to_csv(args.results_dir / f"comparison_{args.split}.csv", index=False)
    (args.results_dir / f"comparison_{args.split}.md").write_text(
        to_markdown(df, args.split), encoding="utf-8")
    print(to_markdown(df, args.split))

    if not args.no_figures:
        for _, r in df.iterrows():
            cm_path = args.results_dir / r["key"] / f"confusion_matrix_{args.split}.csv"
            if cm_path.exists():
                cm = pd.read_csv(cm_path, index_col=0).to_numpy()
                plot_confusion_matrix(
                    cm, f"{r['model']} ({args.split})",
                    args.results_dir / "figures" / f"cm_{r['key']}_{args.split}.png")
        print(f"Figures written to {args.results_dir / 'figures'}")
    print(f"Wrote {args.results_dir / f'comparison_{args.split}.csv'}")


if __name__ == "__main__":
    main()
