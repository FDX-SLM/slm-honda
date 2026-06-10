"""CLI: validate the delivered JSONL against the data contract (docs/SPEC.md §3, Step 1).

Thin entrypoint — parses args and calls :mod:`slm_coach.data.loader`. Reports valid vs invalid
records and the distribution by ``mode`` and ``data_type`` so a skewed delivery is caught
before any GPU time is spent.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from slm_coach.data.loader import DEFAULT_DATA_TYPES, validate_data_dir
from slm_coach.data.schema import Mode
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap

app = typer.Typer(add_completion=False, help="Validate delivered JSONL against the data contract.")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    data_dir: Path = typer.Option(Path("data"), "--data-dir", help="Root data directory."),
    data_type: list[str] = typer.Option(
        list(DEFAULT_DATA_TYPES), "--data-type", help="Data-type splits to validate."
    ),
    report: Path | None = typer.Option(
        None, "--report", help="Optional path to write a JSON validation report."
    ),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Validate every requested split and print a per-type summary."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    stats_by_type = validate_data_dir(data_dir, data_types=data_type)

    total_invalid = 0
    seen_modes: set[str] = set()
    for dtype, stats in stats_by_type.items():
        total_invalid += stats.invalid
        seen_modes.update(stats.by_mode)
        _print_split_table(dtype, stats)

    _print_mode_coverage(seen_modes)

    if report is not None:
        payload = {dtype: stats.as_dict() for dtype, stats in stats_by_type.items()}
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]Wrote JSON report to {report}[/green]")

    if total_invalid > 0:
        console.print(f"[red]Validation found {total_invalid} invalid record(s).[/red]")
        raise typer.Exit(code=1)
    console.print("[green]All records valid.[/green]")


def _print_split_table(
    dtype: str, stats
) -> None:  # noqa: ANN001 - ValidationStats (avoid import cycle in sig)
    """Render one split's validation stats as a rich table."""
    table = Table(title=f"{dtype}  (valid={stats.valid}  invalid={stats.invalid})")
    table.add_column("mode")
    table.add_column("count", justify="right")
    for mode, count in sorted(stats.by_mode.items()):
        table.add_row(mode, str(count))
    if not stats.by_mode:
        table.add_row("—", "0")
    console.print(table)
    for lineno, message in stats.errors[:10]:
        console.print(f"  [yellow]line {lineno}[/yellow]: {message}")


def _print_mode_coverage(seen_modes: set[str]) -> None:
    """Warn if any of the 7 canonical modes is missing from the delivery."""
    all_modes = {m.value for m in Mode}
    missing = sorted(all_modes - seen_modes)
    if missing:
        console.print(f"[yellow]Modes with no data: {missing}[/yellow]")
    else:
        console.print("[green]All 7 conversation modes are represented.[/green]")


if __name__ == "__main__":
    app()
