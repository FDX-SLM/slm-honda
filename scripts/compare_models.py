"""CLI: build a leaderboard across the 4 base models (spec §6.5).

Đọc các ``outputs/eval/<run>/report.json`` (Honda eval) → in leaderboard nội bộ (RC accuracy,
cue-faithfulness, no-fabrication, runbook-completeness, abstention, artifact, latency) để chọn con
tốt nhất cho demo. Xếp hạng theo (rc_accuracy_clear, no_fabrication_rate).

    uv run python scripts/compare_models.py                          # all runs under outputs/eval
    uv run python scripts/compare_models.py --eval-root outputs/eval --out outputs/eval/leaderboard.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import typer
from rich.console import Console

from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap

app = typer.Typer(add_completion=False, help="Leaderboard across evaluated base models.")
console = Console()
logger = get_logger(__name__)

#: (json key, column label, higher-is-better).
_COLUMNS: list[tuple[str, str, bool]] = [
    ("rc_accuracy_clear", "RC-acc(clear)", True),
    ("rc_accuracy", "RC-acc(all)", True),
    ("cue_faithfulness", "cue-faith", True),
    ("no_fabrication_rate", "no-fab", True),
    ("runbook_completeness", "rb-complete", True),
    ("abstention_hallucination", "abst-hall", False),
    ("artifact_valid_at_1", "artifact@1", True),
    ("ece", "ECE", False),
]


def _load(report_root: Path, explicit: list[Path] | None) -> list[dict]:
    paths = list(explicit) if explicit else sorted(report_root.glob("*/report.json"))
    rows: list[dict] = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Skipping unreadable report", extra={"path": str(p)})
            continue
        model = str(data.get("extras", {}).get("model", p.parent.name))
        lat = data.get("latency") or {}
        rows.append(
            {
                "run": p.parent.name,
                "model": model,
                "latency_p95": lat.get("p95"),
                **{k: data.get(k) for k, _, _ in _COLUMNS},
            }
        )
    return rows


def _rank(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            r.get("rc_accuracy_clear") or 0.0,
            r.get("no_fabrication_rate") or 0.0,
        ),
        reverse=True,
    )


def _fmt(v: object) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"


def _markdown(rows: list[dict]) -> str:
    header = "| Rank | Run | Model | " + " | ".join(label for _, label, _ in _COLUMNS) + " | p95(s) |"
    sep = "| ---: | --- | --- " + "| ---: " * (len(_COLUMNS) + 1) + "|"
    lines = ["# Model leaderboard (Honda Entitlement Resolver)", "", header, sep]
    for i, r in enumerate(_rank(rows), 1):
        cells = " | ".join(_fmt(r.get(k)) for k, _, _ in _COLUMNS)
        lines.append(f"| {i} | {r['run']} | {r['model']} | {cells} | {_fmt(r.get('latency_p95'))} |")
    if rows:
        best = _rank(rows)[0]
        lines += ["", f"**Winner:** {best['model']} ({best['run']}) — pick this for the demo."]
    return "\n".join(lines)


@app.command()
def main(
    eval_root: Path = typer.Option(
        Path("outputs/eval"), "--eval-root", help="Directory of <run>/report.json subdirs."
    ),
    report: list[Path] = typer.Option(
        None, "--report", help="Explicit report.json path(s); repeat. Overrides --eval-root."
    ),
    out: Path = typer.Option(
        Path("outputs/eval/leaderboard.md"), "--out", help="Where to write the leaderboard."
    ),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Render a leaderboard from Honda eval reports."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    rows = _load(eval_root, list(report) if report else None)
    if not rows:
        console.print(f"[red]No report.json found (looked under {eval_root}).[/red]")
        raise typer.Exit(code=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = _markdown(rows)
    out.write_text(md, encoding="utf-8")
    csv_out = out.with_suffix(".csv")
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        cols = ["run", "model", *[k for k, _, _ in _COLUMNS], "latency_p95"]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in _rank(rows):
            w.writerow(r)
    console.print(md)
    console.print(f"[green]Leaderboard written to {out} (+ {csv_out})[/green]")


if __name__ == "__main__":
    app()
