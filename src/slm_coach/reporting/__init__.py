"""Metric reporting: CSV tables + PNG charts for training and evaluation.

Tables use the stdlib :mod:`csv` (always available); charts use matplotlib from the optional
``viz`` extra and degrade to a no-op (empty list) when it is not installed. The high-level
:func:`export_training_artifacts` / :func:`export_eval_artifacts` are called automatically at the
end of training / evaluation, and can be re-run on a finished run via ``scripts/plot_metrics.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.reporting.plots import plot_eval, plot_training_curves
from slm_coach.reporting.tables import (
    read_trainer_log,
    write_criteria_table,
    write_per_mode_table,
    write_per_sample_table,
    write_training_table,
)
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from slm_coach.eval.metrics import ModeBreakdown

logger = get_logger(__name__)

__all__ = [
    "export_eval_artifacts",
    "export_training_artifacts",
    "plot_eval",
    "plot_training_curves",
    "read_trainer_log",
    "write_criteria_table",
    "write_per_mode_table",
    "write_per_sample_table",
    "write_training_table",
]


def export_training_artifacts(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    make_tables: bool = True,
    make_plots: bool = True,
) -> dict[str, list[Path]]:
    """Read a training run's log and write its metric table + curve charts.

    Args:
        run_dir: Training output directory containing ``trainer_state.json``.
        out_dir: Where to write artifacts (defaults to ``<run_dir>/metrics``).
        make_tables: Write ``training_log.csv``.
        make_plots: Write loss/eval-metric/LR charts (needs the ``viz`` extra).

    Returns:
        Mapping ``{"tables": [...], "plots": [...]}`` of written paths.
    """
    out = Path(out_dir) if out_dir is not None else Path(run_dir) / "metrics"
    rows = read_trainer_log(run_dir)
    if not rows:
        logger.warning("No training log to export", extra={"run_dir": str(run_dir)})
        return {"tables": [], "plots": []}
    tables: list[Path] = [write_training_table(rows, out)] if make_tables else []
    plots: list[Path] = plot_training_curves(rows, out) if make_plots else []
    logger.info(
        "Exported training artifacts",
        extra={"out": str(out), "tables": len(tables), "plots": len(plots)},
    )
    return {"tables": tables, "plots": plots}


def export_eval_artifacts(
    out_dir: str | Path,
    *,
    breakdown: ModeBreakdown,
    extras: dict[str, Any] | None = None,
    sample_rows: Sequence[dict[str, Any]] | None = None,
    make_tables: bool = True,
    make_plots: bool = True,
) -> dict[str, list[Path]]:
    """Write evaluation metric tables (per-mode/criteria/per-sample) + charts.

    Args:
        out_dir: Report directory to write artifacts alongside ``report.md``/``report.json``.
        breakdown: The aggregated per-mode breakdown.
        extras: Optional extras dict (e.g. ``pairwise_vs_reference`` for the pairwise chart).
        sample_rows: Optional per-case rows for ``per_sample.csv``.
        make_tables: Write the CSV tables.
        make_plots: Write the PNG charts (needs the ``viz`` extra).

    Returns:
        Mapping ``{"tables": [...], "plots": [...]}`` of written paths.
    """
    out = Path(out_dir)
    tables: list[Path] = []
    if make_tables:
        tables.append(write_per_mode_table(breakdown, out))
        tables.append(write_criteria_table(breakdown, out))
        if sample_rows:
            tables.append(write_per_sample_table(sample_rows, out))
    plots: list[Path] = plot_eval(breakdown, out, extras=extras) if make_plots else []
    logger.info(
        "Exported eval artifacts",
        extra={"out": str(out), "tables": len(tables), "plots": len(plots)},
    )
    return {"tables": tables, "plots": plots}
