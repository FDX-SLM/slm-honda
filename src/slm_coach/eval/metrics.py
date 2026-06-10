"""Aggregate per-sample evaluation scores into the per-mode breakdown.

The per-mode breakdown is the most important evaluation output: it reveals which of the
seven conversation modes the model is weak in, so the data team knows which slices to
reinforce. Aggregation here is pure Python/NumPy (no GPU, no API) and is unit-tested.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from slm_coach.eval.rubric import CRITERIA, to_ten_scale, weighted_average


@dataclass
class SampleScore:
    """A single evaluated gold case: its mode plus per-criterion 1-5 scores."""

    sample_id: str
    mode: str
    criteria: dict[str, float]


@dataclass
class ScoreSummary:
    """Aggregated scores for one slice (a mode, or the overall set)."""

    n: int
    weighted_avg_5: float
    weighted_avg_10: float
    per_criterion_mean: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return a JSON-serializable summary."""
        return {
            "n": self.n,
            "weighted_avg_5": round(self.weighted_avg_5, 4),
            "weighted_avg_10": round(self.weighted_avg_10, 4),
            "per_criterion_mean": {k: round(v, 4) for k, v in self.per_criterion_mean.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ScoreSummary:
        """Reconstruct a summary from its :meth:`as_dict` form (e.g. a saved ``report.json``)."""
        return cls(
            n=int(data.get("n", 0)),
            weighted_avg_5=float(data.get("weighted_avg_5", 0.0)),
            weighted_avg_10=float(data.get("weighted_avg_10", 0.0)),
            per_criterion_mean=dict(data.get("per_criterion_mean", {})),
        )


@dataclass
class ModeBreakdown:
    """The full per-mode breakdown plus the overall summary."""

    overall: ScoreSummary
    per_mode: dict[str, ScoreSummary] = field(default_factory=dict)

    def as_dict(self) -> dict:
        """Return a JSON-serializable representation."""
        return {
            "overall": self.overall.as_dict(),
            "per_mode": {mode: s.as_dict() for mode, s in self.per_mode.items()},
        }

    def weakest_modes(self, k: int = 3) -> list[tuple[str, float]]:
        """Return the ``k`` lowest-scoring modes as ``(mode, score_10)`` pairs."""
        ranked = sorted(self.per_mode.items(), key=lambda kv: kv[1].weighted_avg_10)
        return [(mode, summary.weighted_avg_10) for mode, summary in ranked[:k]]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModeBreakdown:
        """Reconstruct a breakdown from a saved ``report.json`` payload (for chart regeneration)."""
        return cls(
            overall=ScoreSummary.from_dict(data.get("overall", {})),
            per_mode={
                mode: ScoreSummary.from_dict(summary)
                for mode, summary in data.get("per_mode", {}).items()
            },
        )


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean (0.0 for an empty sequence)."""
    return sum(values) / len(values) if values else 0.0


def _summarize(samples: Sequence[SampleScore], weights: Mapping[str, float]) -> ScoreSummary:
    """Summarize a list of samples into per-criterion means and a weighted average."""
    per_criterion: dict[str, float] = {}
    for criterion in CRITERIA:
        per_criterion[criterion] = _mean(
            [s.criteria[criterion] for s in samples if criterion in s.criteria]
        )
    weighted_5 = weighted_average(per_criterion, weights) if samples else 0.0
    return ScoreSummary(
        n=len(samples),
        weighted_avg_5=weighted_5,
        weighted_avg_10=to_ten_scale(weighted_5),
        per_criterion_mean=per_criterion,
    )


def aggregate_by_mode(
    samples: Iterable[SampleScore],
    weights: Mapping[str, float],
) -> ModeBreakdown:
    """Aggregate sample scores into per-mode and overall summaries.

    Args:
        samples: Per-sample rubric scores (each tagged with its conversation ``mode``).
        weights: Per-criterion weights from ``configs/eval.yaml``.

    Returns:
        A :class:`ModeBreakdown` with one summary per observed mode plus the overall set.
    """
    sample_list = list(samples)
    by_mode: dict[str, list[SampleScore]] = {}
    for sample in sample_list:
        by_mode.setdefault(sample.mode, []).append(sample)

    per_mode = {
        mode: _summarize(mode_samples, weights) for mode, mode_samples in sorted(by_mode.items())
    }
    overall = _summarize(sample_list, weights)
    return ModeBreakdown(overall=overall, per_mode=per_mode)


def judge_disagreement(per_judge_scores: Mapping[str, float]) -> float:
    """Return the spread (max - min) of a criterion's scores across judges.

    Args:
        per_judge_scores: Mapping of judge name to its score for one criterion/sample.

    Returns:
        The max-minus-min disagreement (0.0 for fewer than two judges).
    """
    values = list(per_judge_scores.values())
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def pairwise_winrate(verdicts: Sequence[str]) -> dict[str, float]:
    """Aggregate pairwise A/B verdicts into win/tie/loss rates.

    Convention: the model under test is answer ``"A"`` and the baseline is answer ``"B"``.

    Args:
        verdicts: Per-case verdicts: ``"A"`` (model wins), ``"B"`` (baseline wins), or ``"tie"``.

    Returns:
        Mapping with ``win``/``tie``/``loss`` rates in ``[0, 1]`` and the sample count ``n``.
    """
    total = len(verdicts)
    if total == 0:
        return {"win": 0.0, "tie": 0.0, "loss": 0.0, "n": 0.0}
    wins = sum(1 for v in verdicts if v == "A")
    ties = sum(1 for v in verdicts if v == "tie")
    return {
        "win": wins / total,
        "tie": ties / total,
        "loss": (total - wins - ties) / total,
        "n": float(total),
    }
