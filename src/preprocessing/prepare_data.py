"""Load the raw MailSense dataset, inspect it, clean it and build the splits.

Run once; every model then reads the saved splits so that all three models
are trained and evaluated on exactly the same partition.

    python -m src.preprocessing.prepare_data \
        --dataset data/Ask0729-fixed.txt --output-dir data

Outputs
    data/processed/mailsense_clean.csv   cleaned corpus (original label kept)
    data/splits/{train,val,test}.csv     stratified split, reused by all models
    results/data_quality_report.json     counts, distributions, removals
    results/removed_examples.csv         every dropped row + the reason
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.labels import LABEL_TO_ID, to_id  # noqa: E402
from src.preprocessing.clean import clean_text, is_meaningless  # noqa: E402


def read_raw(path: Path) -> pd.DataFrame:
    """Read the tab-separated ``label<TAB>text`` file, reporting malformed rows."""
    rows, malformed = [], []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                malformed.append((lineno, line, "empty line"))
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                malformed.append((lineno, line, "no tab separator"))
                continue
            label, text = parts[0].strip(), "\t".join(parts[1:])
            if label not in LABEL_TO_ID:
                malformed.append((lineno, line, f"unknown label {label!r}"))
                continue
            rows.append({"line_no": lineno, "label": label, "raw_text": text})
    df = pd.DataFrame(rows)
    mal = pd.DataFrame(malformed, columns=["line_no", "raw_text", "reason"])
    return df, mal


def build(dataset: Path, output_dir: Path, results_dir: Path, seed: int,
          test_size: float, val_size: float) -> dict:
    df, malformed = read_raw(dataset)
    report = {
        "dataset_file": str(dataset),
        "raw_rows_parsed": int(len(df)),
        "malformed_rows": int(len(malformed)),
        "raw_class_distribution": df["label"].value_counts().to_dict(),
        "missing_values": int(df["raw_text"].isna().sum()),
        "seed": seed,
    }

    removed = []
    if len(malformed):
        for _, r in malformed.iterrows():
            removed.append({"line_no": r["line_no"], "label": "", "text": r["raw_text"],
                            "reason": r["reason"]})

    df["text"] = df["raw_text"].map(clean_text)

    # 1. fragments with essentially no linguistic content
    empty_mask = df["text"].map(is_meaningless)
    for _, r in df[empty_mask].iterrows():
        removed.append({"line_no": int(r["line_no"]), "label": r["label"],
                        "text": r["raw_text"], "reason": "empty/meaningless after cleaning"})
    report["removed_empty_or_meaningless"] = int(empty_mask.sum())
    df = df[~empty_mask].copy()

    # 2. identical text carrying BOTH labels -> genuinely ambiguous, drop all copies.
    #    We never relabel an example by hand (PROJECT_SPEC.md Sec. 3).
    label_counts = df.groupby("text")["label"].nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    conflict_mask = df["text"].isin(conflicting)
    for _, r in df[conflict_mask].iterrows():
        removed.append({"line_no": int(r["line_no"]), "label": r["label"],
                        "text": r["raw_text"], "reason": "duplicate text with conflicting labels"})
    report["removed_conflicting_duplicates"] = int(conflict_mask.sum())
    report["conflicting_duplicate_texts"] = int(len(conflicting))
    df = df[~conflict_mask].copy()

    # 3. exact duplicates (same text, same label): keep the first occurrence.
    #    Duplicates spanning the split would leak training text into the test set.
    dup_mask = df.duplicated(subset=["text"], keep="first")
    for _, r in df[dup_mask].iterrows():
        removed.append({"line_no": int(r["line_no"]), "label": r["label"],
                        "text": r["raw_text"], "reason": "exact duplicate (same text and label)"})
    report["removed_exact_duplicates"] = int(dup_mask.sum())
    df = df[~dup_mask].copy()

    df = df.reset_index(drop=True)
    df.insert(0, "id", range(len(df)))
    df["label_id"] = df["label"].map(to_id)
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().map(len)

    report["final_rows"] = int(len(df))
    report["final_class_distribution"] = df["label"].value_counts().to_dict()
    report["text_length_words"] = {
        k: float(v) for k, v in df["n_words"].describe().to_dict().items()
    }
    report["text_length_chars"] = {
        k: float(v) for k, v in df["n_chars"].describe().to_dict().items()
    }
    report["word_length_percentiles"] = {
        f"p{p}": float(df["n_words"].quantile(p / 100)) for p in (50, 75, 90, 95, 99)
    }

    # Stratified split, defined once and reused by all three models.
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label_id"])
    val_fraction = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_fraction, random_state=seed, stratify=train_val["label_id"])

    cols = ["id", "label", "label_id", "text", "raw_text", "n_words", "n_chars"]
    processed_dir = output_dir / "processed"
    splits_dir = output_dir / "splits"
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    df[cols].to_csv(processed_dir / "mailsense_clean.csv", index=False, encoding="utf-8")
    for name, part in (("train", train), ("val", val), ("test", test)):
        part = part.sort_values("id")
        part[cols].to_csv(splits_dir / f"{name}.csv", index=False, encoding="utf-8")
        report[f"{name}_size"] = int(len(part))
        report[f"{name}_class_distribution"] = part["label"].value_counts().to_dict()

    pd.DataFrame(removed, columns=["line_no", "label", "text", "reason"]).to_csv(
        results_dir / "removed_examples.csv", index=False, encoding="utf-8")
    report["removed_total"] = len(removed)
    report["split_sizes"] = {"test_size": test_size, "val_size": val_size}

    with open(results_dir / "data_quality_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Prepare the MailSense dataset and splits.")
    ap.add_argument("--dataset", type=Path, default=root / "data" / "Ask0729-fixed.txt")
    ap.add_argument("--output-dir", type=Path, default=root / "data")
    ap.add_argument("--results-dir", type=Path, default=root / "results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--val-size", type=float, default=0.15)
    args = ap.parse_args()

    report = build(args.dataset, args.output_dir, args.results_dir,
                   args.seed, args.test_size, args.val_size)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
