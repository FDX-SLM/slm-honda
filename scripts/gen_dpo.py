"""CLI: generate the DPO preference pairs from ground truth (spec §5.6, Appendix B).

Sinh 6 loại cặp (cue_dropped / fabricated_telemetry / overconfident / missing_fields /
forced_guess / overpromise). ``chosen`` qua oracle; ``rejected`` cố tình mắc lỗi.

    uv run python scripts/gen_dpo.py --seed 42 --out data/preference/dpo_pairs.jsonl
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console

from slm_coach.datagen.dpo import generate_dpo
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap
from slm_coach.utils.seed import set_seed

app = typer.Typer(add_completion=False, help="Generate DPO preference pairs from ground truth.")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    out: Path = typer.Option(
        Path("data/preference/dpo_pairs.jsonl"), "--out", help="Output JSONL path."
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
    limit: int = typer.Option(None, "--limit", help="Cap TOTAL pairs (smoke); default ~600."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Generate and write the DPO JSONL."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    set_seed(seed)
    records = generate_dpo(seed, limit=limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["bad_type"] for r in records)
    console.print(f"[green]Wrote {len(records)} DPO pairs -> {out}[/green]")
    console.print("Per-type distribution:")
    for bad_type, n in sorted(counts.items()):
        console.print(f"  {bad_type:22s} {n}")


if __name__ == "__main__":
    app()
