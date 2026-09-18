"""
Prepare the MailSense dataset.

Steps:
1. Read the raw dataset.
2. Clean the email text.
3. Remove invalid and duplicate examples.
4. Create a stratified train/validation/test split.
5. Save the processed data and quality report.

Run:
    python -m src.preprocessing.prepare_data
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.common.labels import LABEL_TO_ID, to_id
from src.preprocessing.clean import clean_text, is_meaningless


def read_raw_dataset(path):
    """Read the raw label<TAB>text dataset."""
    rows = []
    removed = []

    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\r\n")

            if not line.strip():
                removed.append({
                    "line_no": line_number,
                    "label": "",
                    "text": line,
                    "reason": "empty line"
                })
                continue

            parts = line.split("\t", 1)

            if len(parts) != 2:
                removed.append({
                    "line_no": line_number,
                    "label": "",
                    "text": line,
                    "reason": "invalid format"
                })
                continue

            label = parts[0].strip()
            text = parts[1].strip()

            if label not in LABEL_TO_ID:
                removed.append({
                    "line_no": line_number,
                    "label": label,
                    "text": text,
                    "reason": "unknown label"
                })
                continue

            rows.append({
                "line_no": line_number,
                "label": label,
                "raw_text": text
            })

    return pd.DataFrame(rows), removed


def class_distribution(data):
    """Return class counts as normal Python integers."""
    return {
        str(label): int(count)
        for label, count in data["label"].value_counts().items()
    }


def prepare_data(
    dataset_path,
    output_dir,
    results_dir,
    seed=42,
    test_size=0.15,
    val_size=0.15
):
    """Clean the dataset and create the common data split."""

    data, removed = read_raw_dataset(dataset_path)

    report = {
        "dataset_file": str(dataset_path),
        "raw_rows": len(data),
        "malformed_or_invalid_rows": len(removed),
        "raw_class_distribution": class_distribution(data),
        "seed": seed
    }

    # Clean text
    data["text"] = data["raw_text"].apply(clean_text)

    # Remove empty or meaningless text
    meaningless = data["text"].apply(is_meaningless)

    for _, row in data[meaningless].iterrows():
        removed.append({
            "line_no": int(row["line_no"]),
            "label": row["label"],
            "text": row["raw_text"],
            "reason": "empty or meaningless after cleaning"
        })

    data = data[~meaningless].copy()

    # Remove identical text appearing with different labels.
    # We cannot decide which label is correct, so remove those examples.
    label_counts = data.groupby("text")["label"].nunique()
    conflicting_texts = set(label_counts[label_counts > 1].index)

    conflicts = data["text"].isin(conflicting_texts)

    for _, row in data[conflicts].iterrows():
        removed.append({
            "line_no": int(row["line_no"]),
            "label": row["label"],
            "text": row["raw_text"],
            "reason": "same text has conflicting labels"
        })

    data = data[~conflicts].copy()

    # Remove exact duplicate text with the same label.
    duplicates = data.duplicated(subset="text", keep="first")

    for _, row in data[duplicates].iterrows():
        removed.append({
            "line_no": int(row["line_no"]),
            "label": row["label"],
            "text": row["raw_text"],
            "reason": "duplicate text"
        })

    data = data[~duplicates].copy()

    # Add useful information
    data = data.reset_index(drop=True)
    data.insert(0, "id", range(len(data)))

    data["label_id"] = data["label"].apply(to_id)
    data["n_chars"] = data["text"].str.len()
    data["n_words"] = data["text"].str.split().str.len()

    report["final_rows"] = len(data)
    report["final_class_distribution"] = class_distribution(data)

    # Create the same stratified split for every model.
    train_val, test = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        stratify=data["label_id"]
    )

    validation_fraction = val_size / (1 - test_size)

    train, validation = train_test_split(
        train_val,
        test_size=validation_fraction,
        random_state=seed,
        stratify=train_val["label_id"]
    )

    # Save files
    processed_dir = output_dir / "processed"
    splits_dir = output_dir / "splits"

    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "id",
        "label",
        "label_id",
        "text",
        "raw_text",
        "n_words",
        "n_chars"
    ]

    data[columns].to_csv(
        processed_dir / "mailsense_clean.csv",
        index=False
    )

    splits = {
        "train": train,
        "val": validation,
        "test": test
    }

    for name, split in splits.items():
        split = split.sort_values("id")

        split[columns].to_csv(
            splits_dir / f"{name}.csv",
            index=False
        )

        report[f"{name}_size"] = len(split)
        report[f"{name}_class_distribution"] = class_distribution(split)

    # Save information about removed examples
    removed_path = results_dir / "removed_examples.csv"

    pd.DataFrame(
        removed,
        columns=["line_no", "label", "text", "reason"]
    ).to_csv(
        removed_path,
        index=False
    )

    report["removed_total"] = len(removed)

    with open(
        results_dir / "data_quality_report.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    return report


def main():
    root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Prepare the MailSense dataset."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "data" / "Ask0729-fixed.txt"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data"
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=root / "results"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15
    )

    args = parser.parse_args()

    report = prepare_data(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()