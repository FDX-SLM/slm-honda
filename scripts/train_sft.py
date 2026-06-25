"""CLI: T1 fast LoRA SFT baseline (docs/SPEC.md §11, Step 2).

Thin entrypoint — parses args, loads the config, seeds, then delegates to
:func:`slm_coach.training.sft.run_sft_training`. ``--dry-run`` resolves and prints the plan
without touching a GPU.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from slm_coach.config import load_sft_config
from slm_coach.model_registry import apply_to_config
from slm_coach.training.sft import run_sft_training
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap
from slm_coach.utils.seed import set_seed

app = typer.Typer(add_completion=False, help="Train T1: fast LoRA SFT baseline.")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", help="Path to the SFT config YAML."),
    base: str = typer.Option(
        None, "--base", help="Override base model via the registry (alias or HF id); model-agnostic."
    ),
    resume: Path | None = typer.Option(None, "--resume", help="Checkpoint dir to resume from."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve config + plan only (no GPU)."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Run single-stage LoRA SFT."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    cfg = load_sft_config(config)
    if base:
        spec = apply_to_config(cfg, base)
        # Keep per-model runs from overwriting each other's checkpoints/reports.
        cfg.run_name = f"{cfg.run_name}_{spec.key}"
        logger.info("Base model overridden", extra={"base": spec.hf_id, "notes": spec.notes})
    set_seed(cfg.seed)
    logger.info("Starting SFT", extra={"run_name": cfg.run_name, "model": cfg.model_name})
    if dry_run:
        console.print_json(cfg.model_dump_json(indent=2))
        console.print("[green]Dry run OK — config resolved, no training launched.[/green]")
        raise typer.Exit()
    best = run_sft_training(cfg, resume=resume)
    console.print(f"[green]Best checkpoint: {best}[/green]")


if __name__ == "__main__":
    app()
