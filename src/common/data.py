"""Loading the fixed train/val/test splits.

All models read the same three CSV files, so the comparison is on identical
data and no model ever sees the test set during training or model selection.
"""
from pathlib import Path

import pandas as pd

SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


def load_split(splits_dir: Path, split: str) -> pd.DataFrame:
    path = Path(splits_dir) / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:  python -m src.preprocessing.prepare_data")
    df = pd.read_csv(path, encoding="utf-8")
    df["text"] = df["text"].fillna("").astype(str)
    return df


def load_splits(splits_dir: Path):
    return (load_split(splits_dir, "train"),
            load_split(splits_dir, "val"),
            load_split(splits_dir, "test"))
