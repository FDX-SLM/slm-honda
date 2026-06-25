"""CLI: generate the SFT dataset from ground truth (spec §5.1–5.5, Appendix B).

Sinh 5 nhóm SFT (complaint→resolution, knowledge, differential, distractor, abstention) từ
``slm_coach.ground_truth``; mọi mẫu có resolution package đều qua oracle (§4) trước khi ghi. Output
``messages`` chat format, tiếng Anh, model-agnostic.

    uv run python scripts/gen_sft.py --seed 42 --out data/sft/train_sft.jsonl
    uv run python scripts/gen_sft.py --seed 42 --limit 30 --out data/sft/smoke.jsonl   # smoke
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console

from slm_coach.datagen.sft import generate_sft
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap
from slm_coach.utils.seed import set_seed

app = typer.Typer(add_completion=False, help="Generate the SFT dataset from ground truth.")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    out: Path = typer.Option(Path("data/sft/train_sft.jsonl"), "--out", help="Output JSONL path."),
    seed: int = typer.Option(42, "--seed", help="RNG seed."),
    limit: int = typer.Option(None, "--limit", help="Cap TOTAL records (smoke); default full mix."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Generate and write the SFT JSONL."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    set_seed(seed)
    records = generate_sft(seed, limit=limit)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = Counter(r["mode"] for r in records)
    console.print(f"[green]Wrote {len(records)} SFT records -> {out}[/green]")
    console.print("Per-slice distribution:")
    for slice_tag, n in sorted(counts.items()):
        console.print(f"  {slice_tag:14s} {n}")


if __name__ == "__main__":
    app()
