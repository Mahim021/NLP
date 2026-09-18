"""Authoritative label handling.

The dataset labels are ``Yes`` (actionable) and ``No`` (non-actionable).
They are preserved verbatim in every processed file; the numeric label is a
deterministic mapping of the original string and nothing else.
"""

LABEL_TO_ID = {"No": 0, "Yes": 1}
ID_TO_LABEL = {0: "No", 1: "Yes"}

# Human-readable names used in reports / confusion matrices.
CLASS_NAMES = ["Non-Actionable (No)", "Actionable (Yes)"]

POSITIVE_LABEL = "Yes"   # actionable == positive class
POSITIVE_ID = 1


def to_id(label: str) -> int:
    """Map an original dataset label to its numeric id (deterministic)."""
    key = label.strip()
    if key not in LABEL_TO_ID:
        raise ValueError(f"Unknown label {label!r}; expected one of {sorted(LABEL_TO_ID)}")
    return LABEL_TO_ID[key]
