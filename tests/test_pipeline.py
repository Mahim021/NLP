"""Sanity checks for the properties PROJECT_SPEC.md actually cares about.

Run with pytest, or directly:  python tests/test_pipeline.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.data import load_splits  # noqa: E402
from src.common.labels import LABEL_TO_ID, to_id  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.preprocessing.clean import clean_text  # noqa: E402
from src.preprocessing.prepare_data import read_raw  # noqa: E402

SPLITS = ROOT / "data" / "splits"


def test_label_mapping_is_deterministic():
    assert LABEL_TO_ID == {"No": 0, "Yes": 1}
    assert to_id("Yes") == 1 and to_id("No") == 0
    assert to_id(" Yes ") == 1
    try:
        to_id("Maybe")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown labels must be rejected, never guessed")


def test_cleaning_keeps_actionability_cues():
    text = "Please  confirm&nbsp;<b>ASAP</b> — submit the report by tomorrow at http://x.com/a"
    out = clean_text(text)
    for cue in ("Please", "confirm", "ASAP", "submit", "by tomorrow"):
        assert cue in out, f"cleaning destroyed the cue {cue!r}: {out!r}"
    assert "<b>" not in out and "&nbsp;" not in out
    assert "$LINK" in out and "http" not in out
    assert "  " not in out


def test_cleaning_is_idempotent():
    text = "Can you review this?  See  http://example.com/x  and mail me@x.org"
    once = clean_text(text)
    assert clean_text(once) == once


def test_raw_dataset_matches_the_spec():
    raw, malformed = read_raw(ROOT / "data" / "Ask0729-fixed.txt")
    counts = raw["label"].value_counts().to_dict()
    assert len(malformed) == 0
    assert len(raw) == 3657, len(raw)
    assert counts == {"No": 1938, "Yes": 1719}, counts


def test_splits_are_disjoint_stratified_and_label_faithful():
    train, val, test = load_splits(SPLITS)
    ids = [set(df["id"]) for df in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])

    # No text overlap either: a duplicate across the boundary would leak.
    texts = [set(df["text"]) for df in (train, val, test)]
    assert not (texts[0] & texts[2]), "training text leaked into the test set"
    assert not (texts[1] & texts[2]), "validation text leaked into the test set"

    for df in (train, val, test):
        assert set(df["label"]) <= {"Yes", "No"}
        assert (df["label"].map(to_id) == df["label_id"]).all(), "label_id drifted from label"

    full = pd.concat([train, val, test])
    base = (full["label"] == "Yes").mean()
    for df in (train, val, test):
        assert abs((df["label"] == "Yes").mean() - base) < 0.02, "split is not stratified"


def test_metrics_positive_class_is_actionable():
    # true: No, Yes, Yes, No ; pred: No, No, Yes, Yes
    m = compute_metrics([0, 1, 1, 0], [0, 0, 1, 1])
    assert m["true_positives"] == 1
    assert m["false_negatives"] == 1, "a missed actionable e-mail must count as an FN"
    assert m["false_positives"] == 1
    assert m["true_negatives"] == 1
    assert m["accuracy"] == 0.5
    assert m["confusion_matrix"] == [[1, 1], [1, 1]]


def test_all_models_were_scored_on_the_same_test_set():
    """Whatever has been trained so far must agree on the test examples."""
    import json
    results = ROOT / "results"
    seen = {}
    for path in results.glob("*/metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        support = payload["metrics"]["test"]["support"]
        seen[path.parent.name] = support
    if len(seen) > 1:
        values = list(seen.values())
        assert all(v == values[0] for v in values), f"test sets differ: {seen}"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
