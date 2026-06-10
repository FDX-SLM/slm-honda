"""Optional offline generation-latency measurement.

Measures p50/p95 of ``model.generate`` calls during offline generation. This is raw generation
timing for reference, NOT a serving benchmark.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyStats:
    """Summary of generation latencies (seconds)."""

    n: int
    p50: float
    p95: float
    mean: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serializable summary."""
        return {
            "n": self.n,
            "p50": round(self.p50, 4),
            "p95": round(self.p95, 4),
            "mean": round(self.mean, 4),
        }


def summarize_latencies(durations: Sequence[float]) -> LatencyStats:
    """Summarize per-call generation durations into p50/p95/mean.

    Args:
        durations: Per-``generate``-call wall-clock durations in seconds.

    Returns:
        A :class:`LatencyStats` summary (all-zero when ``durations`` is empty).
    """
    if not durations:
        return LatencyStats(n=0, p50=0.0, p95=0.0, mean=0.0)
    array = np.asarray(durations, dtype=float)
    return LatencyStats(
        n=len(array),
        p50=float(np.percentile(array, 50)),
        p95=float(np.percentile(array, 95)),
        mean=float(array.mean()),
    )


def measure_generation_latency(
    generate_one: Callable[[list[dict[str, str]]], str],
    prompts: Sequence[list[dict[str, str]]],
    n_samples: int,
) -> LatencyStats:
    """Time ``generate_one`` over up to ``n_samples`` prompts and summarize.

    Args:
        generate_one: Callable generating an answer for a single chat prompt.
        prompts: Prompts to time.
        n_samples: Maximum number of prompts to time.

    Returns:
        A :class:`LatencyStats` over the measured calls.
    """
    durations: list[float] = []
    for prompt in list(prompts)[:n_samples]:
        start = time.perf_counter()
        generate_one(prompt)
        durations.append(time.perf_counter() - start)
    return summarize_latencies(durations)
