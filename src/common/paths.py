"""Path helpers.

Nothing in this project hard-codes a Kaggle path.  Every script takes the
dataset / output locations as arguments; these values are only defaults that
make the local repository work out of the box.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATASET = DATA_DIR / "Ask0729-fixed.txt"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIGS_DIR = PROJECT_ROOT / "configs"
