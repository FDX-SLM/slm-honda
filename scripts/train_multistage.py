"""CLI: T2 multi-stage QLoRA SFT — the real SFT (docs/SPEC.md §11, Step 3).

Thin entrypoint — delegates to :func:`slm_coach.training.multistage.run_multistage_training`.
``--dry-run`` resolves the config and prints the curriculum plan without a GPU.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from slm_coach.config import load_sft_config
from slm_coach.training.multistage import run_multistage_training
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap
from slm_coach.utils.seed import set_seed

app = typer.Typer(add_completion=False, help="Train T2: multi-stage QLoRA SFT (curriculum).")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Path to the multi-stage SFT config YAML."),
    resume: Path | None = typer.Option(None, "--resume", help="Checkpoint dir to resume from."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve config + plan only (no GPU)."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Run the multi-stage SFT curriculum."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    cfg = load_sft_config(config)
    set_seed(cfg.seed)
    stage_names = [s.name for s in cfg.stages]
    logger.info("Starting multi-stage SFT", extra={"run_name": cfg.run_name, "stages": stage_names})
    if dry_run:
        console.print_json(cfg.model_dump_json(indent=2))
        console.print(f"[green]Dry run OK — curriculum stages: {stage_names}[/green]")
        raise typer.Exit()
    best = run_multistage_training(cfg, resume=resume)
    console.print(f"[green]Best checkpoint: {best}[/green]")


if __name__ == "__main__":
    app()
