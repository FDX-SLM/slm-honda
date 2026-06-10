"""CLI: (re)generate metric tables + charts from a finished run.

Reads a training run directory (its ``trainer_state.json``) and/or an evaluation ``report.json``
and writes CSV tables + PNG charts. Useful to regenerate artifacts after the fact, with a
different output location, or once matplotlib (the ``viz`` extra) has been installed.

Examples::

    uv run python scripts/plot_metrics.py --run-dir checkpoints/sft_lora_smoke
    uv run python scripts/plot_metrics.py --report outputs/eval/run1/report.json
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from slm_coach.eval.metrics import ModeBreakdown
from slm_coach.reporting import export_eval_artifacts, export_training_artifacts
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap

app = typer.Typer(add_completion=False, help="Regenerate metric tables + charts from a run.")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    run_dir: Path | None = typer.Option(
        None, "--run-dir", help="Training run dir containing trainer_state.json."
    ),
    report: Path | None = typer.Option(
        None, "--report", help="An eval report.json to regenerate tables/charts from."
    ),
    out_dir: Path | None = typer.Option(
        None, "--out-dir", help="Output dir (defaults to <run-dir>/metrics or the report's dir)."
    ),
    no_plots: bool = typer.Option(False, "--no-plots", help="Write CSV tables only (skip charts)."),
    no_tables: bool = typer.Option(False, "--no-tables", help="Write charts only (skip CSVs)."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Regenerate training and/or evaluation metric artifacts."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    if not run_dir and not report:
        console.print("[red]Provide at least one of --run-dir or --report.[/red]")
        raise typer.Exit(code=1)

    if run_dir:
        result = export_training_artifacts(
            run_dir, out_dir, make_tables=not no_tables, make_plots=not no_plots
        )
        console.print(
            f"[green]Training: {len(result['tables'])} table(s), "
            f"{len(result['plots'])} chart(s).[/green]"
        )

    if report:
        payload = json.loads(Path(report).read_text(encoding="utf-8"))
        breakdown = ModeBreakdown.from_dict(payload)
        target = out_dir or Path(report).parent
        result = export_eval_artifacts(
            target,
            breakdown=breakdown,
            extras=payload.get("extras", {}),
            make_tables=not no_tables,
            make_plots=not no_plots,
        )
        console.print(
            f"[green]Eval: {len(result['tables'])} table(s), "
            f"{len(result['plots'])} chart(s) in {target}.[/green]"
        )


if __name__ == "__main__":
    app()
