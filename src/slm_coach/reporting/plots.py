"""Matplotlib charts for training curves and evaluation breakdowns.

Matplotlib is an optional ``viz``-extra dependency, imported lazily behind the non-interactive
``Agg`` backend. If it is not installed, every plotting function logs an install hint and returns
an empty list, so training and evaluation never fail for want of a charting library.

Training charts: ``loss_curve.png`` (train + eval loss), ``eval_metric.png`` (rubric score over
steps), ``lr_schedule.png``. Evaluation charts: ``per_mode.png`` (score per conversation mode —
the primary view), ``criteria.png`` (per-criterion means), ``pairwise.png`` (win/tie/loss vs the
gold reference).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from slm_coach.eval.metrics import ModeBreakdown

logger = get_logger(__name__)


def _pyplot() -> Any | None:
    """Return ``matplotlib.pyplot`` on the Agg backend, or ``None`` if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib not installed; skipping charts (install the 'viz' extra: "
            "uv sync --extra viz)"
        )
        return None
    return plt


def _series(rows: Sequence[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    """Extract aligned ``(step, value)`` series for ``key`` from per-step rows."""
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if key in row and "step" in row:
            xs.append(row["step"])
            ys.append(row[key])
    return xs, ys


def _save(fig: Any, path: Path, plt: Any) -> Path:
    """Tighten, save, and close a figure; return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Wrote chart", extra={"path": str(path)})
    return path


def plot_training_curves(rows: Sequence[dict[str, Any]], out_dir: str | Path) -> list[Path]:
    """Plot loss curves, the eval-rubric curve, and the LR schedule from per-step rows.

    Args:
        rows: Per-step rows from :func:`slm_coach.reporting.tables.read_trainer_log`.
        out_dir: Directory to write the PNGs into.

    Returns:
        Paths of the charts written (empty if matplotlib is missing or there is no data).
    """
    plt = _pyplot()
    if plt is None or not rows:
        return []
    out = Path(out_dir)
    written: list[Path] = []

    train_x, train_y = _series(rows, "loss")
    eval_x, eval_y = _series(rows, "eval_loss")
    if train_y or eval_y:
        fig, ax = plt.subplots(figsize=(8, 5))
        if train_y:
            ax.plot(train_x, train_y, marker="o", markersize=3, label="train loss")
        if eval_y:
            ax.plot(eval_x, eval_y, marker="s", markersize=3, label="eval loss")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Training & eval loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        written.append(_save(fig, out / "loss_curve.png", plt))

    metric_x, metric_y = _series(rows, "eval_rubric_avg")
    if metric_y:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(metric_x, metric_y, marker="o", markersize=3, color="tab:green")
        ax.set_xlabel("step")
        ax.set_ylabel("eval_rubric_avg (1-5)")
        ax.set_ylim(1, 5)
        ax.set_title("Eval rubric score during training")
        ax.grid(True, alpha=0.3)
        written.append(_save(fig, out / "eval_metric.png", plt))

    lr_x, lr_y = _series(rows, "learning_rate")
    if lr_y:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(lr_x, lr_y, color="tab:orange")
        ax.set_xlabel("step")
        ax.set_ylabel("learning rate")
        ax.set_title("Learning-rate schedule")
        ax.grid(True, alpha=0.3)
        written.append(_save(fig, out / "lr_schedule.png", plt))

    grad_x, grad_y = _series(rows, "grad_norm")
    if grad_y:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grad_x, grad_y, color="tab:red", linewidth=1.0)
        ax.set_xlabel("step")
        ax.set_ylabel("gradient norm")
        ax.set_title("Gradient norm during training")
        ax.grid(True, alpha=0.3)
        written.append(_save(fig, out / "grad_norm.png", plt))

    return written


def plot_eval(
    breakdown: ModeBreakdown, out_dir: str | Path, *, extras: dict[str, Any] | None = None
) -> list[Path]:
    """Plot the per-mode bar chart, per-criterion means, and pairwise win-rate.

    Args:
        breakdown: The aggregated per-mode breakdown.
        out_dir: Directory to write the PNGs into.
        extras: Optional extras dict (a ``pairwise_vs_reference`` entry adds the pairwise chart).

    Returns:
        Paths of the charts written (empty if matplotlib is missing).
    """
    plt = _pyplot()
    if plt is None:
        return []
    out = Path(out_dir)
    extras = extras or {}
    written: list[Path] = []

    modes = sorted(breakdown.per_mode.items(), key=lambda kv: kv[1].weighted_avg_10)
    if modes:
        names = [mode for mode, _ in modes]
        scores = [summary.weighted_avg_10 for _, summary in modes]
        fig, ax = plt.subplots(figsize=(8, max(3.0, 0.5 * len(names) + 1.0)))
        ax.barh(names, scores, color="tab:blue")
        ax.set_xlabel("score /10")
        ax.set_xlim(0, 10)
        ax.set_title("Per-mode score (weakest at top)")
        for index, value in enumerate(scores):
            ax.text(min(value + 0.1, 9.5), index, f"{value:.1f}", va="center")
        ax.grid(True, axis="x", alpha=0.3)
        written.append(_save(fig, out / "per_mode.png", plt))

    criteria = breakdown.overall.per_criterion_mean
    if criteria:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(list(criteria), list(criteria.values()), color="tab:purple")
        ax.set_ylabel("mean /5")
        ax.set_ylim(0, 5)
        ax.set_title("Per-criterion mean")
        ax.tick_params(axis="x", rotation=45)
        plt.setp(ax.get_xticklabels(), ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        written.append(_save(fig, out / "criteria.png", plt))

    pairwise = extras.get("pairwise_vs_reference")
    if pairwise:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        left = 0.0
        for label, color in (("win", "tab:green"), ("tie", "tab:gray"), ("loss", "tab:red")):
            value = float(pairwise.get(label, 0.0)) * 100.0
            ax.barh(
                ["model vs gold"], [value], left=left, color=color, label=f"{label} {value:.0f}%"
            )
            left += value
        ax.set_xlim(0, 100)
        ax.set_xlabel("%")
        ax.set_title("Pairwise win-rate vs reference")
        ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25))
        written.append(_save(fig, out / "pairwise.png", plt))

    return written
