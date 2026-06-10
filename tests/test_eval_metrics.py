"""Tests for per-mode score aggregation (the primary eval output)."""

from __future__ import annotations

import pytest

from slm_coach.eval.metrics import (
    SampleScore,
    aggregate_by_mode,
    judge_disagreement,
    pairwise_winrate,
)
from slm_coach.eval.rubric import CRITERIA, DEFAULT_WEIGHTS


def _uniform(value: float) -> dict[str, float]:
    return {criterion: value for criterion in CRITERIA}


def test_aggregate_by_mode_separates_modes():
    samples = [
        SampleScore("c1", "comparison", _uniform(4)),
        SampleScore("c2", "comparison", _uniform(4)),
        SampleScore("p1", "purchase_intent", _uniform(2)),
    ]
    breakdown = aggregate_by_mode(samples, DEFAULT_WEIGHTS)

    assert set(breakdown.per_mode) == {"comparison", "purchase_intent"}

    comparison = breakdown.per_mode["comparison"]
    assert comparison.n == 2
    assert comparison.weighted_avg_5 == pytest.approx(4.0)
    assert comparison.weighted_avg_10 == pytest.approx(7.5)

    purchase = breakdown.per_mode["purchase_intent"]
    assert purchase.n == 1
    assert purchase.weighted_avg_5 == pytest.approx(2.0)
    assert purchase.weighted_avg_10 == pytest.approx(2.5)


def test_overall_summary_and_weakest_modes():
    samples = [
        SampleScore("c1", "comparison", _uniform(4)),
        SampleScore("c2", "comparison", _uniform(4)),
        SampleScore("p1", "purchase_intent", _uniform(2)),
    ]
    breakdown = aggregate_by_mode(samples, DEFAULT_WEIGHTS)

    assert breakdown.overall.n == 3
    assert breakdown.overall.per_criterion_mean["factuality"] == pytest.approx(10 / 3)
    assert breakdown.overall.weighted_avg_10 == pytest.approx((10 / 3 - 1) / 4 * 10)

    assert breakdown.weakest_modes(1) == [("purchase_intent", pytest.approx(2.5))]


def test_per_criterion_weighting_changes_score():
    # A sample strong on factuality, weak on format; factuality is weighted 4x format.
    scores = dict.fromkeys(CRITERIA, 3.0)
    scores["factuality"] = 5.0
    scores["format"] = 1.0
    sample = SampleScore("s1", "comparison", scores)

    breakdown = aggregate_by_mode([sample], DEFAULT_WEIGHTS)
    # Weighted mean must exceed the unweighted mean because the high score carries more weight.
    unweighted = sum(scores.values()) / len(scores)
    assert breakdown.per_mode["comparison"].weighted_avg_5 > unweighted


def test_disagreement_and_pairwise_helpers():
    assert judge_disagreement({"gpt": 4.0, "gemini": 2.0}) == pytest.approx(2.0)
    assert judge_disagreement({"gpt": 4.0}) == 0.0

    rates = pairwise_winrate(["A", "A", "B", "tie"])
    assert rates["win"] == pytest.approx(0.5)
    assert rates["loss"] == pytest.approx(0.25)
    assert rates["tie"] == pytest.approx(0.25)
    assert rates["n"] == 4.0
    assert pairwise_winrate([])["win"] == 0.0
