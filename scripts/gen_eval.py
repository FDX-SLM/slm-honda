"""CLI: generate the evaluation sets (spec §7, Appendix B).

``eval.jsonl`` (seed 999, balanced 3 RC + abstention) → dùng làm ``data/gold/gold_test.jsonl``;
``eval_hard.jsonl`` (20 hand-written messy complaints) báo cáo riêng.

    uv run python scripts/gen_eval.py --seed 999 --out data/gold/gold_test.jsonl
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console

from slm_coach.datagen.evalset import generate_eval, generate_eval_hard
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap
from slm_coach.utils.seed import set_seed

app = typer.Typer(add_completion=False, help="Generate the eval + eval_hard sets.")
console = Console()
logger = get_logger(__name__)


def _write(records: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


@app.command()
def main(
    out: Path = typer.Option(
        Path("data/gold/gold_test.jsonl"), "--out", help="Eval (gold) JSONL path."
    ),
    hard_out: Path = typer.Option(
        Path("data/gold/eval_hard.jsonl"), "--hard-out", help="eval_hard JSONL path."
    ),
    seed: int = typer.Option(999, "--seed", help="RNG seed (held-out; default 999)."),
    limit: int = typer.Option(None, "--limit", help="Cap eval size (smoke); default 180."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Generate and write the eval + eval_hard JSONL files."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    set_seed(seed)
    eval_records = generate_eval(seed, limit=limit)
    hard_records = generate_eval_hard()
    _write(eval_records, out)
    _write(hard_records, hard_out)
    counts = Counter(r["leading_root_cause"] for r in eval_records)
    console.print(f"[green]Wrote {len(eval_records)} eval -> {out}[/green]")
    console.print(f"[green]Wrote {len(hard_records)} eval_hard -> {hard_out}[/green]")
    console.print("Eval label distribution:")
    for label, n in sorted(counts.items()):
        console.print(f"  {label:26s} {n}")


if __name__ == "__main__":
    app()
