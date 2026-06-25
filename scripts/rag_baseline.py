"""CLI: RAG baseline over the incident history (spec §8 split-screen foil).

Chạy RAG "copy nearest ticket" trên gold/eval set, in RC accuracy + confusion + vài ca RAG sai
(cue-flip) để đối chứng với SLM closed-book. Index lấy từ ground_truth.INCIDENTS (offline, không
cần chroma/internet).

    uv run python scripts/rag_baseline.py --gold data/gold/gold_test.jsonl
    uv run python scripts/rag_baseline.py --gold data/gold/eval_hard.jsonl --out outputs/rag/hard
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console

from slm_coach.eval.honda import load_honda_gold
from slm_coach.eval.rag import RagBaseline
from slm_coach.ground_truth import ALL_LABELS
from slm_coach.utils.logging import configure_logging, get_logger
from slm_coach.utils.runtime import bootstrap

app = typer.Typer(add_completion=False, help="RAG baseline (copy nearest ticket) over gold.")
console = Console()
logger = get_logger(__name__)


@app.command()
def main(
    gold: Path = typer.Option(
        Path("data/gold/gold_test.jsonl"), "--gold", help="Gold/eval JSONL to score against."
    ),
    out: Path = typer.Option(Path("outputs/rag"), "--out", help="Output report dir."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON-structured logs."),
) -> None:
    """Score the RAG baseline and write a report."""
    bootstrap()
    configure_logging(json_logs=json_logs)
    cases = load_honda_gold(gold)
    rag = RagBaseline()

    correct = 0
    confusion: dict[str, Counter] = {g: Counter() for g in ALL_LABELS}
    mistakes: list[dict] = []
    for c in cases:
        pred = rag.predict(c["complaint"])
        pred_rc = pred["leading_root_cause"]
        gold_rc = c["gold_rc"]
        confusion.setdefault(gold_rc, Counter())[pred_rc] += 1
        if pred_rc == gold_rc:
            correct += 1
        elif len(mistakes) < 8:
            mistakes.append(
                {
                    "complaint": c["complaint"][:120],
                    "gold": gold_rc,
                    "rag_pred": pred_rc,
                    "copied_from": pred["retrieved_incident"],
                }
            )
    acc = correct / len(cases) if cases else 0.0

    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG baseline report (copy nearest ticket)",
        "",
        f"- Cases: {len(cases)}",
        f"- **RAG RC accuracy: {acc:.3f}** (cue-blind — sai trên ca cue-flip)",
        "",
        "## Where RAG fails (cue-flip: surface similar, wrong RC)",
        "",
        "| Gold RC | RAG predicted | Copied from | Complaint (excerpt) |",
        "| --- | --- | --- | --- |",
    ]
    for m in mistakes:
        lines.append(
            f"| {m['gold']} | {m['rag_pred']} | {m['copied_from']} | {m['complaint']} |"
        )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "report.json").write_text(
        json.dumps(
            {"accuracy": acc, "n": len(cases), "confusion": {g: dict(c) for g, c in confusion.items()}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]RAG RC accuracy: {acc:.3f} on {len(cases)} cases -> {out}/report.md[/green]")
    for m in mistakes[:5]:
        console.print(f"  [red]gold={m['gold']} rag={m['rag_pred']}[/red] (copied {m['copied_from']})")


if __name__ == "__main__":
    app()
