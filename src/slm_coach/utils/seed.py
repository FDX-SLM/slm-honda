"""Reproducible seeding for every train/eval entrypoint."""

from __future__ import annotations

import os
import random

from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int) -> int:
    """Seed all relevant RNGs for reproducibility.

    Seeds Python's :mod:`random`, ``PYTHONHASHSEED``, NumPy, and — when available —
    PyTorch (CPU and CUDA). Torch is imported lazily so this works on a machine with
    no GPU/torch installed.

    Args:
        seed: The integer seed to apply everywhere.

    Returns:
        The seed that was applied (echoed for convenience).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a core dep
        logger.debug("numpy not available; skipping numpy seeding")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding (no-GPU environment)")

    logger.info("Global seed set", extra={"seed": seed})
    return seed
