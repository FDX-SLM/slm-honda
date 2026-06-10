"""CLI: export / quantize the best checkpoint — the final deliverable (docs/SPEC.md §11, Step 6).

Thin entrypoint — merges the LoRA adapter into FP16, then quantizes to AWQ INT4 and/or
GGUF Q4_K_M. Delegates to :mod:`slm_coach.export.merge` and :mod:`slm_coach.export.quantize`.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from slm_coach.export.merge import merge_lora_to_fp16
from slm_coach.export.quantize import load_calibration_texts, quantize_awq, quantize_gguf
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap

app = typer.Typer(add_completion=False, help="Export & quantize a checkpoint (AWQ INT4 + GGUF).")
console = Console()
logger = get_logger(__name__)

_SUPPORTED_FORMATS = {"awq", "gguf"}


@app.command()
def main(
    checkpoint: Path = typer.Option(..., "--checkpoint", help="Best checkpoint to export."),
    formats: str = typer.Option("awq,gguf", "--formats", help="Comma list: awq,gguf."),
    output: Path = typer.Option(Path("outputs/exported"), "--output", help="Output root dir."),
    calib_data: Path | None = typer.Option(
        None, "--calib-data", help="JSONL/TXT of in-domain texts for AWQ calibration."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the export plan only (no GPU)."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Merge the adapter to FP16, then quantize to the requested formats."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    requested = [f.strip().lower() for f in formats.split(",") if f.strip()]
    unknown = sorted(set(requested) - _SUPPORTED_FORMATS)
    if unknown:
        raise typer.BadParameter(
            f"unknown formats {unknown}; supported: {sorted(_SUPPORTED_FORMATS)}"
        )
    logger.info("Starting export", extra={"checkpoint": str(checkpoint), "formats": requested})

    if dry_run:
        console.print(
            f"[green]Dry run OK — merge {checkpoint} -> FP16, then export {requested}.[/green]"
        )
        raise typer.Exit()

    calib = load_calibration_texts(calib_data) if calib_data else None
    fp16_dir = merge_lora_to_fp16(checkpoint, output / "fp16")
    if "awq" in requested:
        quantize_awq(fp16_dir, output / "awq", calib_data=calib)
    if "gguf" in requested:
        quantize_gguf(fp16_dir, output / "gguf")
    console.print(f"[green]Exported {requested} under {output}[/green]")


if __name__ == "__main__":
    app()
