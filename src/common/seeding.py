"""Reproducible seeding shared by every model."""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and (when installed) torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
