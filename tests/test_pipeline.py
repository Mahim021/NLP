"""Sanity checks for the MailSense data and evaluation pipeline."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.data import load_splits
from src.common.labels import LABEL_TO_ID, to_id
from src.evaluation.metrics import compute_metrics
from src.preprocessing.clean import clean_text


SPLITS = ROOT / "data" / "splits"
RAW_DATASET = ROOT / "data" / "Ask0729-fixed.txt"


def test_label_mapping_is_deterministic():
    assert LABEL_TO_ID == {"No": 0, "Yes": 1}

    assert to_id("Yes") == 1
    assert to_id("No") == 0
    assert to_id(" Yes ") == 1

    try:
        to_id("Maybe")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Unknown labels must be rejected, never guessed"
        )


def test_cleaning_keeps_actionability_cues():
    text = (
        "Please confirm&nbsp;<b>ASAP</b> — "
        "submit the report by tomorrow at http://x.com/a"
    )

    output = clean_text(text)

    for cue in ("Please", "confirm", "ASAP", "submit", "by tomorrow"):
        assert cue in output, (
            f"Cleaning destroyed the cue {cue!r}: {output!r}"
        )

    assert "<b>" not in output
    assert "&nbsp;" not in output
    assert "$LINK" in output
    assert "http" not in output
    assert "  " not in output


def test_cleaning_is_idempotent():
    text = (
        "Can you review this?  See  "
        "http://example.com/x  and mail me@x.org"
    )

    once = clean_text(text)

    assert clean_text(once) == once


def test_raw_dataset_matches_expected_counts():
    assert RAW_DATASET.exists(), (
        f"Raw dataset not found: {RAW_DATASET}"
    )

    rows = []

    with RAW_DATASET.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\r\n")

            if not line.strip():
                continue

            parts = line.split("\t", 1)

            assert len(parts) == 2, (
                f"Malformed row at line {line_number}: {line!r}"
            )

            label, text = parts

            assert label in {"Yes", "No"}, (
                f"Invalid label at line {line_number}: {label!r}"
            )

            rows.append((label, text))

    assert len(rows) == 3657

    counts = pd.Series(
        [label for label, _ in rows]
    ).value_counts().to_dict()

    assert counts == {
        "No": 1938,
        "Yes": 1719,
    }


def test_splits_are_disjoint_stratified_and_label_faithful():
    train, val, test = load_splits(SPLITS)

    # Example IDs must not occur in more than one split.
    ids = [
        set(train["id"]),
        set(val["id"]),
        set(test["id"]),
    ]

    assert not (ids[0] & ids[1]), (
        "training IDs leaked into validation"
    )
    assert not (ids[0] & ids[2]), (
        "training IDs leaked into test"
    )
    assert not (ids[1] & ids[2]), (
        "validation IDs leaked into test"
    )

    # Cleaned text must not overlap across splits.
    texts = [
        set(train["text"]),
        set(val["text"]),
        set(test["text"]),
    ]

    assert not (texts[0] & texts[1]), (
        "training text leaked into validation"
    )
    assert not (texts[0] & texts[2]), (
        "training text leaked into test"
    )
    assert not (texts[1] & texts[2]), (
        "validation text leaked into test"
    )

    # Labels must remain valid and consistent with label_id.
    for dataframe in (train, val, test):
        assert set(dataframe["label"]) <= {"Yes", "No"}

        assert (
            dataframe["label"].map(to_id)
            == dataframe["label_id"]
        ).all(), "label_id drifted from label"

    # Verify approximately equal class proportions.
    full = pd.concat(
        [train, val, test],
        ignore_index=True,
    )

    base_ratio = (full["label"] == "Yes").mean()

    for dataframe in (train, val, test):
        split_ratio = (dataframe["label"] == "Yes").mean()

        assert abs(split_ratio - base_ratio) < 0.02, (
            "split is not sufficiently stratified"
        )


def test_metrics_positive_class_is_actionable():
    # True: No, Yes, Yes, No
    # Pred: No, No,  Yes, Yes
    metrics = compute_metrics(
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    )

    assert metrics["true_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1

    assert metrics["accuracy"] == 0.5

    assert metrics["confusion_matrix"] == [
        [1, 1],
        [1, 1],
    ]


def test_all_trained_models_use_same_test_set_size():
    """Check that saved model results have the same test-set size."""

    results_dir = ROOT / "results"

    seen = {}

    for path in results_dir.glob("*/metrics.json"):
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )

        test_metrics = payload["metrics"].get("test")

        if test_metrics is not None:
            seen[path.parent.name] = test_metrics["support"]

    # This test is only meaningful once multiple models have been trained.
    if len(seen) > 1:
        supports = list(seen.values())

        assert all(
            support == supports[0]
            for support in supports
        ), f"Test-set sizes differ: {seen}"


if __name__ == "__main__":
    failures = 0

    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print(f"PASS  {name}")
            except Exception as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")

    print(f"\n{failures} failure(s)")

    sys.exit(1 if failures else 0)