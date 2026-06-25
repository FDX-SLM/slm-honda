"""CLI: Honda Entitlement Resolver evaluation (spec §7, Appendix B Step).

Thin entrypoint — delegates to :func:`slm_coach.eval.honda.run_honda_eval`. Generates offline,
scores deterministically against the oracle + ground truth (RC accuracy + confusion + cue
faithfulness + no-fabricated-telemetry + runbook completeness + calibration + latency), and writes
``outputs/eval/<run>/report.{md,json}`` + ``per_sample.csv``.

    uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/sft/best
    uv run python scripts/evaluate.py --config configs/eval.yaml --model any --mock   # offline
    uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/sft/best \
        --hard --run-name eval_hard   # the 20 hand-written hard cases (reported separately)
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from slm_coach.config import load_eval_config
from slm_coach.eval.honda import run_honda_eval
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap
from slm_coach.utils.seed import set_seed

app = typer.Typer(add_completion=False, help="Evaluate a checkpoint (Honda resolver KPIs).")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Path to the evaluation config YAML."),
    model: Path = typer.Option(..., "--model", help="Checkpoint to evaluate (offline)."),
    base: str = typer.Option(
        None, "--base", help="Registry base for recommended sampling (alias or HF id)."
    ),
    hard: bool = typer.Option(
        False, "--hard", help="Evaluate the eval_hard set (data/gold/eval_hard.jsonl)."
    ),
    gold: Path = typer.Option(None, "--gold", help="Override the gold file to evaluate."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve config + plan only (no GPU/API)."),
    mock: bool = typer.Option(False, "--mock", help="Offline mock: replay gold references."),
    run_name: str = typer.Option(None, "--run-name", help="Report dir name; default timestamp."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Run the Honda evaluation pipeline."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    cfg = load_eval_config(config)
    set_seed(cfg.seed)
    gold_override = gold or (Path("data/gold/eval_hard.jsonl") if hard else None)
    logger.info(
        "Starting Honda eval",
        extra={"model": str(model), "gold": str(gold_override or cfg.gold), "mock": mock},
    )
    if dry_run:
        console.print_json(cfg.model_dump_json(indent=2))
        console.print(f"[green]Dry run OK — would evaluate {model}.[/green]")
        raise typer.Exit()
    report_dir = run_honda_eval(
        cfg, model, mock=mock, run_name=run_name, base=base, gold_override=gold_override
    )
    console.print(f"[green]Report written to {report_dir}[/green]")


if __name__ == "__main__":
    app()
