"""Tabular (CSV) export of training and evaluation metrics.

Reads the cumulative ``log_history`` that ``transformers`` writes to ``trainer_state.json`` and
flattens it into a per-step ``training_log.csv``; flattens the evaluation per-mode breakdown and
per-sample scores into CSVs. Uses the stdlib :mod:`csv` only (no extra dependency), so these
always run regardless of which optional extras are installed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.eval.rubric import CRITERIA
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from slm_coach.eval.metrics import ModeBreakdown

logger = get_logger(__name__)

#: Summary-only keys emitted once at the end of training; excluded from the per-step curve table.
_SUMMARY_KEYS = frozenset(
    {
        "train_runtime",
        "train_samples_per_second",
        "train_steps_per_second",
        "train_loss",
        "train_tokens_per_second",
        "total_flos",
        "eval_runtime",
        "eval_samples_per_second",
        "eval_steps_per_second",
    }
)

#: Preferred leading column order for the per-step training-curve table.
_TRAIN_COLUMNS = (
    "step",
    "epoch",
    "loss",
    "eval_loss",
    "learning_rate",
    "grad_norm",
    "eval_rubric_avg",
)

#: Preferred leading column order for the per-sample evaluation table.
_SAMPLE_COLUMNS = (
    "id",
    "sample_id",
    "mode",
    "score_5",
    "score_10",
    *CRITERIA,
    "prompt",
    "answer",
    "reference",
)


def find_trainer_state(run_dir: str | Path) -> Path | None:
    """Locate the most complete ``trainer_state.json`` under a run directory.

    Prefers ``<run_dir>/trainer_state.json`` (written by ``trainer.save_state()``); otherwise
    falls back to the highest-step ``checkpoint-*/trainer_state.json`` (whose ``log_history`` is
    cumulative).

    Args:
        run_dir: A training output directory.

    Returns:
        The path to the chosen ``trainer_state.json``, or ``None`` if none exists.
    """
    run = Path(run_dir)
    direct = run / "trainer_state.json"
    if direct.is_file():
        return direct
    candidates = list(run.glob("checkpoint-*/trainer_state.json")) or list(
        run.glob("**/trainer_state.json")
    )
    if not candidates:
        return None

    def _step(path: Path) -> int:
        try:
            return int(path.parent.name.split("-")[-1])
        except ValueError:
            return -1

    return max(candidates, key=_step)


def read_trainer_log(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read and flatten a training run's ``log_history`` into per-step rows.

    Args:
        run_dir: A training output directory containing ``trainer_state.json``.

    Returns:
        One merged dict per step (train + eval entries for the same step are combined), sorted by
        step; an empty list if no log is found.
    """
    state_path = find_trainer_state(run_dir)
    if state_path is None:
        logger.warning("No trainer_state.json found", extra={"run_dir": str(run_dir)})
        return []
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read trainer_state.json", extra={"error": str(exc)})
        return []
    return merge_log_history(state.get("log_history", []))


def merge_log_history(log_history: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge ``transformers`` log entries (separate train/eval rows) into one row per step.

    Args:
        log_history: The raw ``log_history`` list from ``trainer_state.json``.

    Returns:
        Per-step rows with numeric metrics only (summary/runtime keys dropped), sorted by step.
    """
    by_step: dict[int, dict[str, Any]] = {}
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        row = by_step.setdefault(step, {"step": step})
        for key, value in entry.items():
            if key == "step" or key in _SUMMARY_KEYS:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                row[key] = value
    return [by_step[step] for step in sorted(by_step)]


def _ordered_columns(rows: Sequence[dict[str, Any]], preferred: Sequence[str]) -> list[str]:
    """Order columns: known ``preferred`` keys first (in order), then the rest alphabetically."""
    seen: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.append(key)
    lead = [c for c in preferred if c in seen]
    rest = sorted(c for c in seen if c not in lead)
    return lead + rest


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> Path:
    """Write ``rows`` to ``path`` as CSV with the given header ``columns``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def write_training_table(
    rows: Sequence[dict[str, Any]], out_dir: str | Path, *, filename: str = "training_log.csv"
) -> Path:
    """Write the per-step training metrics table (loss/eval_loss/lr/rubric over steps).

    Args:
        rows: Per-step rows from :func:`read_trainer_log`.
        out_dir: Directory to write into.
        filename: Output file name.

    Returns:
        The path to the written CSV.
    """
    path = Path(out_dir) / filename
    columns = _ordered_columns(rows, _TRAIN_COLUMNS)
    _write_csv(path, columns, rows)
    logger.info("Wrote training table", extra={"path": str(path), "rows": len(rows)})
    return path


def write_run_facts(
    facts: Mapping[str, Any], out_dir: str | Path, *, filename: str = "run_facts"
) -> list[Path]:
    """Write the training run-facts table (config knobs for the report).

    Records the things a training report should state plainly — base model, method (full/LoRA/
    QLoRA), precision, gradient checkpointing, effective batch, scheduler, masking, data counts —
    as both ``run_facts.csv`` (key,value) and ``run_facts.md`` (a readable two-column table).

    Args:
        facts: Flat mapping of fact name → value.
        out_dir: Directory to write into.
        filename: Base file name (``.csv`` and ``.md`` are appended).

    Returns:
        The written file paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{filename}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["fact", "value"])
        for key, value in facts.items():
            writer.writerow([key, value])
    md_path = out / f"{filename}.md"
    lines = ["# Training run facts", "", "| Fact | Value |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in facts.items()]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote run facts", extra={"csv": str(csv_path)})
    return [csv_path, md_path]


def write_training_summary(
    rows: Sequence[dict[str, Any]], out_dir: str | Path, *, filename: str = "training_summary.md"
) -> Path:
    """Write a short end-of-run summary (final/best loss, eval, steps) for the report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    losses = [r["loss"] for r in rows if "loss" in r]
    eval_losses = [r["eval_loss"] for r in rows if "eval_loss" in r]
    rubric = [r["eval_rubric_avg"] for r in rows if "eval_rubric_avg" in r]
    steps = [r["step"] for r in rows if "step" in r]
    lines = ["# Training summary", ""]
    if steps:
        lines.append(f"- Steps logged: {len(rows)} (max step {max(steps)})")
    if losses:
        lines.append(f"- Train loss: first {losses[0]:.4f} → last {losses[-1]:.4f}")
    if eval_losses:
        lines.append(f"- Eval loss: best {min(eval_losses):.4f} (last {eval_losses[-1]:.4f})")
    if rubric:
        lines.append(f"- Eval rubric/oracle metric: best {max(rubric):.4f}")
    path = out / filename
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote training summary", extra={"path": str(path)})
    return path


def write_per_mode_table(
    breakdown: ModeBreakdown, out_dir: str | Path, *, filename: str = "per_mode.csv"
) -> Path:
    """Write the per-mode score table (overall row first, then one row per mode).

    Args:
        breakdown: The aggregated per-mode breakdown.
        out_dir: Directory to write into.
        filename: Output file name.

    Returns:
        The path to the written CSV.
    """
    columns = ["mode", "n", "score_5", "score_10", *CRITERIA]
    rows: list[dict[str, Any]] = []
    for label, summary in [("overall", breakdown.overall), *sorted(breakdown.per_mode.items())]:
        row: dict[str, Any] = {
            "mode": label,
            "n": summary.n,
            "score_5": round(summary.weighted_avg_5, 4),
            "score_10": round(summary.weighted_avg_10, 4),
        }
        row.update({c: round(summary.per_criterion_mean.get(c, 0.0), 4) for c in CRITERIA})
        rows.append(row)
    path = Path(out_dir) / filename
    _write_csv(path, columns, rows)
    logger.info("Wrote per-mode table", extra={"path": str(path)})
    return path


def write_criteria_table(
    breakdown: ModeBreakdown, out_dir: str | Path, *, filename: str = "criteria.csv"
) -> Path:
    """Write the overall per-criterion mean table (one row per rubric criterion)."""
    columns = ["criterion", "mean_5"]
    rows = [
        {"criterion": criterion, "mean_5": round(mean, 4)}
        for criterion, mean in breakdown.overall.per_criterion_mean.items()
    ]
    path = Path(out_dir) / filename
    _write_csv(path, columns, rows)
    logger.info("Wrote criteria table", extra={"path": str(path)})
    return path


def write_per_sample_table(
    rows: Sequence[dict[str, Any]], out_dir: str | Path, *, filename: str = "per_sample.csv"
) -> Path:
    """Write the per-gold-case evaluation table (id, mode, scores, answer, reference).

    Args:
        rows: One dict per evaluated case (keys are ordered by :data:`_SAMPLE_COLUMNS`).
        out_dir: Directory to write into.
        filename: Output file name.

    Returns:
        The path to the written CSV.
    """
    columns = _ordered_columns(rows, _SAMPLE_COLUMNS)
    path = Path(out_dir) / filename
    _write_csv(path, columns, rows)
    logger.info("Wrote per-sample table", extra={"path": str(path), "rows": len(rows)})
    return path
