r"""Honda Entitlement Resolver evaluation (spec §7) — deterministic oracle KPIs.

Khác bản sale (LLM-rubric per-mode), eval này **deterministic**: chấm output model bằng
:mod:`slm_coach.oracle` so với :mod:`slm_coach.ground_truth`. KPI:

* RC accuracy + confusion matrix 3-RC + ABSTAIN (chứng minh phân biệt theo cue).
* cue-grounding faithfulness (% evidence thật sự có trong lời than). Target ≥95%.
* no-fabricated-telemetry rate (KPI honesty quan trọng nhất). Target ≥98%.
* runbook completeness & fidelity (đủ field + khớp runbook gold). Target ≥95%.
* calibration: ECE + overconfident-wrong rate (phạt conf>0.85 mà sai).
* abstention hallucination (ca mơ hồ/ngoài catalog mà ép RC). Target <10%.
* artifact valid@1 (RCA/work-order/email/mermaid present + JSON parse). Target ≥90%.
* latency p50/p95 (offline ``model.generate``). Target <1.5s.

Phần tính metric thuần Python (không numpy) để test offline; numpy/torch chỉ nạp ở real-eval path.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.ground_truth import ALL_LABELS, ABSTAIN, ROOT_CAUSES, SYSTEM_PROMPT
from slm_coach.oracle import (
    check_resolution,
    find_fabricated_telemetry,
    is_grounded,
    parse_output,
    runbook_complete,
    runbook_fidelity_ok,
    self_service_ok,
)
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_coach.config import EvalFileConfig

logger = get_logger(__name__)

_PARSE_FAIL = "PARSE_FAIL"
_ARTIFACT_KEYS = ("rca_md", "work_order_md", "customer_email", "diagram_mermaid")


@dataclass
class SampleEval:
    """Per-case evaluation result."""

    id: str
    slice: str
    gold_rc: str
    pred_rc: str
    correct: bool
    confidence: float | None
    cue_grounded_frac: float
    no_fabricated_telemetry: bool
    runbook_complete: bool
    runbook_fidelity: bool
    artifacts_valid: bool
    self_service_ok: bool = False
    telemetry_hits: list[str] = field(default_factory=list)


@dataclass
class HondaReport:
    """Aggregate Honda eval metrics (the report payload)."""

    n: int
    rc_accuracy: float
    rc_accuracy_clear: float  # accuracy on cases whose gold is a concrete RC
    cue_faithfulness: float
    no_fabrication_rate: float
    runbook_completeness: float
    runbook_fidelity: float
    artifact_valid_at_1: float
    self_service_present_rate: float
    abstention_hallucination: float
    ece: float
    overconfident_wrong_rate: float
    parse_fail_rate: float
    per_slice_accuracy: dict[str, float] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    latency: dict[str, float] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gold loading
# ---------------------------------------------------------------------------


def load_honda_gold(path: str | Path) -> list[dict[str, Any]]:
    """Load gold/eval records (raw dicts with ``leading_root_cause`` + user complaint)."""
    from slm_coach.data.loader import load_gold

    out: list[dict[str, Any]] = []
    for obj in load_gold(path):
        messages = obj.get("messages") or [{"role": "user", "content": obj.get("prompt", "")}]
        complaint = "\n".join(m["content"] for m in messages if m.get("role") == "user")
        out.append(
            {
                "id": obj.get("id", f"case-{len(out)}"),
                "slice": obj.get("mode", "unknown"),
                "complaint": complaint,
                "gold_rc": obj.get("leading_root_cause", ABSTAIN),
                "reference": obj.get("reference", ""),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Per-sample scoring
# ---------------------------------------------------------------------------


def score_sample(case: dict[str, Any], answer: str) -> SampleEval:
    """Score one model answer against a gold case using the oracle + ground truth."""
    complaint = case["complaint"]
    gold_rc = case["gold_rc"]
    think, resolution = parse_output(answer)

    if resolution is None:
        return SampleEval(
            id=case["id"],
            slice=case["slice"],
            gold_rc=gold_rc,
            pred_rc=_PARSE_FAIL,
            correct=False,
            confidence=None,
            cue_grounded_frac=0.0,
            no_fabricated_telemetry=not find_fabricated_telemetry(think),
            runbook_complete=False,
            runbook_fidelity=False,
            artifacts_valid=False,
            self_service_ok=False,
        )

    diag = resolution.get("diagnosis", {}) or {}
    pred_rc = str(diag.get("leading_root_cause", "")).strip() or _PARSE_FAIL
    try:
        conf = float(diag.get("confidence"))
    except (TypeError, ValueError):
        conf = None

    res = check_resolution(complaint, think, resolution)

    evidence = diag.get("evidence_in_ticket", []) or []
    if evidence:
        grounded = sum(1 for e in evidence if is_grounded(str(e), complaint))
        cue_frac = grounded / len(evidence)
    else:
        cue_frac = 1.0  # nothing claimed → nothing to fabricate

    complete = runbook_complete(resolution) if pred_rc in ROOT_CAUSES else (pred_rc == ABSTAIN)
    fidelity = runbook_fidelity_ok(resolution, pred_rc) if pred_rc in ROOT_CAUSES else (
        pred_rc == ABSTAIN
    )
    artifacts = resolution.get("artifacts", {}) or {}
    artifacts_valid = all(str(artifacts.get(k, "")).strip() for k in _ARTIFACT_KEYS)

    return SampleEval(
        id=case["id"],
        slice=case["slice"],
        gold_rc=gold_rc,
        pred_rc=pred_rc,
        correct=(pred_rc == gold_rc),
        confidence=conf,
        cue_grounded_frac=cue_frac,
        no_fabricated_telemetry=res.no_fabricated_telemetry,
        runbook_complete=complete,
        runbook_fidelity=fidelity,
        artifacts_valid=artifacts_valid,
        self_service_ok=self_service_ok(resolution),
        telemetry_hits=res.telemetry_hits,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _expected_calibration_error(samples: Sequence[SampleEval], n_bins: int = 10) -> float:
    """Bucketed ECE: |avg confidence − accuracy| weighted by bucket size."""
    scored = [s for s in samples if s.confidence is not None]
    if not scored:
        return 0.0
    bins: list[list[SampleEval]] = [[] for _ in range(n_bins)]
    for s in scored:
        idx = min(n_bins - 1, int(s.confidence * n_bins))
        bins[idx].append(s)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = _mean([s.confidence for s in bucket])
        acc = _mean([1.0 if s.correct else 0.0 for s in bucket])
        ece += (len(bucket) / len(scored)) * abs(avg_conf - acc)
    return ece


def aggregate(samples: Sequence[SampleEval]) -> HondaReport:
    """Aggregate per-sample results into the Honda report (all KPIs)."""
    n = len(samples)
    clear = [s for s in samples if s.gold_rc in ROOT_CAUSES]
    abstain_cases = [s for s in samples if s.gold_rc == ABSTAIN]

    # Confusion matrix over gold × pred (pred may be PARSE_FAIL).
    confusion: dict[str, dict[str, int]] = {
        g: {p: 0 for p in (*ALL_LABELS, _PARSE_FAIL)} for g in ALL_LABELS
    }
    for s in samples:
        if s.gold_rc in confusion:
            confusion[s.gold_rc][s.pred_rc] = confusion[s.gold_rc].get(s.pred_rc, 0) + 1

    # Per-slice accuracy.
    by_slice: dict[str, list[SampleEval]] = {}
    for s in samples:
        by_slice.setdefault(s.slice, []).append(s)
    per_slice_acc = {
        slc: _mean([1.0 if s.correct else 0.0 for s in items])
        for slc, items in sorted(by_slice.items())
    }

    overconfident_wrong = [
        s for s in samples if s.confidence is not None and s.confidence > 0.85 and not s.correct
    ]
    rc_pred = [s for s in samples if s.pred_rc in ROOT_CAUSES]

    return HondaReport(
        n=n,
        rc_accuracy=_mean([1.0 if s.correct else 0.0 for s in samples]),
        rc_accuracy_clear=_mean([1.0 if s.correct else 0.0 for s in clear]),
        cue_faithfulness=_mean([s.cue_grounded_frac for s in samples]),
        no_fabrication_rate=_mean([1.0 if s.no_fabricated_telemetry else 0.0 for s in samples]),
        runbook_completeness=_mean([1.0 if s.runbook_complete else 0.0 for s in rc_pred]) if rc_pred else 0.0,
        runbook_fidelity=_mean([1.0 if s.runbook_fidelity else 0.0 for s in rc_pred]) if rc_pred else 0.0,
        artifact_valid_at_1=_mean([1.0 if s.artifacts_valid else 0.0 for s in samples]),
        self_service_present_rate=_mean([1.0 if s.self_service_ok else 0.0 for s in samples]),
        abstention_hallucination=(
            _mean([1.0 if s.pred_rc != ABSTAIN else 0.0 for s in abstain_cases])
            if abstain_cases
            else 0.0
        ),
        ece=_expected_calibration_error(samples),
        overconfident_wrong_rate=len(overconfident_wrong) / len(samples) if samples else 0.0,
        parse_fail_rate=_mean([1.0 if s.pred_rc == _PARSE_FAIL else 0.0 for s in samples]),
        per_slice_accuracy=per_slice_acc,
        confusion=confusion,
    )


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

_TARGETS = {
    "rc_accuracy_clear": (">=", 0.85),
    "cue_faithfulness": (">=", 0.95),
    "no_fabrication_rate": (">=", 0.98),
    "runbook_completeness": (">=", 0.95),
    "abstention_hallucination": ("<", 0.10),
    "artifact_valid_at_1": (">=", 0.90),
}


def _verdict(metric: str, value: float) -> str:
    """PASS/FAIL marker vs the spec target for a metric (blank if untargeted)."""
    if metric not in _TARGETS:
        return ""
    op, target = _TARGETS[metric]
    ok = value >= target if op == ">=" else value < target
    return "✅" if ok else "❌"


def _confusion_md(confusion: dict[str, dict[str, int]]) -> list[str]:
    cols = [*ALL_LABELS, _PARSE_FAIL]
    header = "| gold ＼ pred | " + " | ".join(c.replace("_", " ")[:10] for c in cols) + " |"
    sep = "| --- " + "| ---: " * len(cols) + "|"
    lines = [header, sep]
    for g in ALL_LABELS:
        row = confusion.get(g, {})
        lines.append(
            f"| {g.replace('_', ' ')[:14]} | " + " | ".join(str(row.get(c, 0)) for c in cols) + " |"
        )
    return lines


def write_honda_report(
    out_dir: str | Path, report: HondaReport, *, extras: dict[str, Any] | None = None
) -> tuple[Path, Path]:
    """Write ``report.md`` + ``report.json`` for a Honda eval run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extras = extras or {}
    report.extras = extras

    kpi_rows = [
        ("RC accuracy (all)", report.rc_accuracy, "rc_accuracy"),
        ("RC accuracy (clear-cue)", report.rc_accuracy_clear, "rc_accuracy_clear"),
        ("Cue-grounding faithfulness", report.cue_faithfulness, "cue_faithfulness"),
        ("No-fabricated-telemetry rate", report.no_fabrication_rate, "no_fabrication_rate"),
        ("Runbook completeness", report.runbook_completeness, "runbook_completeness"),
        ("Runbook fidelity", report.runbook_fidelity, "runbook_fidelity"),
        ("Artifact valid@1", report.artifact_valid_at_1, "artifact_valid_at_1"),
        ("Abstention hallucination", report.abstention_hallucination, "abstention_hallucination"),
        ("Calibration ECE", report.ece, "ece"),
        ("Overconfident-wrong rate", report.overconfident_wrong_rate, "overconfident_wrong_rate"),
        ("Parse-fail rate", report.parse_fail_rate, "parse_fail_rate"),
    ]
    lines: list[str] = [
        "# Honda Entitlement Resolver — evaluation report",
        "",
        f"- **Model:** {extras.get('model', 'n/a')}",
        f"- **Cases:** {report.n}",
        "",
        "## KPIs",
        "",
        "| Metric | Value | Target |",
        "| --- | ---: | :---: |",
    ]
    for label, value, key in kpi_rows:
        lines.append(f"| {label} | {value:.3f} | {_verdict(key, value)} |")
    lines += [
        "",
        "## Confusion matrix (3 RC + ABSTAIN)",
        "",
        *_confusion_md(report.confusion),
        "",
        "## Per-slice accuracy",
        "",
        "| Slice | Accuracy |",
        "| --- | ---: |",
    ]
    for slc, acc in report.per_slice_accuracy.items():
        lines.append(f"| {slc} | {acc:.3f} |")
    if report.latency:
        lat = report.latency
        lines += [
            "",
            "## Generation latency (offline)",
            "",
            f"- p50: {lat.get('p50', 0):.3f}s | p95: {lat.get('p95', 0):.3f}s "
            f"| mean: {lat.get('mean', 0):.3f}s (n={lat.get('n', 0)}) — target <1.5s",
        ]

    md_path = out / "report.md"
    json_path = out / "report.json"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    payload = asdict(report)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote Honda eval report", extra={"md": str(md_path)})
    return md_path, json_path


def write_per_sample_csv(out_dir: str | Path, samples: Sequence[SampleEval]) -> Path:
    """Write a per-case CSV (id, slice, gold/pred, flags) for manual inspection."""
    import csv

    out = Path(out_dir) / "per_sample.csv"
    cols = [
        "id", "slice", "gold_rc", "pred_rc", "correct", "confidence", "cue_grounded_frac",
        "no_fabricated_telemetry", "runbook_complete", "runbook_fidelity", "artifacts_valid",
        "self_service_ok", "telemetry_hits",
    ]
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in samples:
            row = asdict(s)
            row["telemetry_hits"] = "; ".join(s.telemetry_hits)
            w.writerow(row)
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _mock_answer(case: dict[str, Any]) -> str:
    """Deterministic offline answer for ``--mock``: replay the gold reference, else build it."""
    if case.get("reference"):
        return case["reference"]
    # eval_hard has no reference text → build the ideal package for the gold label.
    import random

    from slm_coach.datagen.core import assistant_content, build_abstention, build_case

    rng = random.Random(hash(case["id"]) & 0xFFFF)
    gold = case["gold_rc"]
    built = build_abstention(rng) if gold == ABSTAIN else build_case(rng, gold)
    return assistant_content(built.think, built.resolution)


def run_honda_eval(
    config: EvalFileConfig,
    model_path: str | Path,
    *,
    dry_run: bool = False,
    mock: bool = False,
    run_name: str | None = None,
    base: str | None = None,
    gold_override: str | Path | None = None,
) -> Path:
    """Run the Honda evaluation pipeline and write the report.

    Args:
        config: Validated eval config.
        model_path: Checkpoint to evaluate (offline) — ignored under ``mock``.
        dry_run: Load gold + log plan only.
        mock: Replay gold references (no GPU) to exercise the metric pipeline.
        run_name: Report subdir (default timestamp).
        base: Optional registry base to apply recommended sampling.
        gold_override: Evaluate a different gold file (e.g. eval_hard) instead of config.gold.

    Returns:
        Path to the report directory.
    """
    report_root = Path(config.report_dir) / "eval"
    run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
    report_dir = report_root / run_name

    gold_path = Path(gold_override or config.gold)
    cases = load_honda_gold(gold_path) if gold_path.is_file() else []
    if not cases and not dry_run:
        raise FileNotFoundError(f"Gold set not found or empty: {gold_path}")

    logger.info(
        "Honda eval plan",
        extra={"model": str(model_path), "n_cases": len(cases), "mock": mock, "gold": str(gold_path)},
    )
    if dry_run:
        logger.info("Dry run: skipping generation", extra={"report_dir": str(report_dir)})
        return report_dir

    system_prompt = config.system_prompt or SYSTEM_PROMPT
    gen_prompts = [
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": c["complaint"]}]
        for c in cases
    ]

    latency_stats: dict[str, float] | None = None
    if mock:
        answers = [_mock_answer(c) for c in cases]
    else:
        from slm_coach.config import GenerationConfig
        from slm_coach.eval.inference import batch_generate, load_for_inference

        gen_cfg: GenerationConfig = config.generation
        if base:
            from slm_coach.model_registry import apply_sampling

            apply_sampling(gen_cfg, base)
        model, tokenizer = load_for_inference(model_path)
        answers = batch_generate(model, tokenizer, gen_prompts, gen_cfg)
        if config.latency.measure:
            latency_stats = _measure_latency(model, tokenizer, gen_prompts, config)

    samples = [score_sample(c, a) for c, a in zip(cases, answers, strict=False)]
    report = aggregate(samples)
    report.latency = latency_stats
    extras = {"model": str(model_path), "gold": str(gold_path), "mock": mock}
    write_honda_report(report_dir, report, extras=extras)
    write_per_sample_csv(report_dir, samples)
    _log_eval_to_langfuse(config, run_name, cases, answers, samples, model_path=str(model_path))
    logger.info("Honda eval complete", extra={"report_dir": str(report_dir)})
    return report_dir


def _log_eval_to_langfuse(
    config: EvalFileConfig,
    run_name: str,
    cases: Sequence[dict[str, Any]],
    answers: Sequence[str],
    samples: Sequence[SampleEval],
    *,
    model_path: str,
) -> None:
    """Push each evaluated sample (full prompt → generated answer + scores) to Langfuse.

    No-op when the tracking extra is missing or the Langfuse keys are unset (Tracker degrades
    gracefully), so eval never depends on it. Unlike the training-time callback this logs the FULL
    generation for EVERY case, not a single truncated peek.
    """
    from slm_coach.tracking import init_tracking

    tracker = init_tracking(config, run_name=run_name)
    if not tracker.langfuse_enabled:
        return
    try:
        for c, answer, s in zip(cases, answers, samples, strict=False):
            tracker.log_generation(
                name="eval_sample",
                prompt=c["complaint"],
                completion=answer,
                eval_id=s.id,
                slice=s.slice,
                gold_rc=s.gold_rc,
                pred_rc=s.pred_rc,
                correct=s.correct,
                confidence=s.confidence,
                no_fabricated_telemetry=s.no_fabricated_telemetry,
                model=model_path,
            )
    finally:
        tracker.close()


def _measure_latency(model: Any, tokenizer: Any, prompts: list, config: EvalFileConfig) -> dict:
    """Measure p50/p95 of single-prompt generation (offline timing)."""
    from slm_coach.eval.inference import batch_generate
    from slm_coach.eval.latency import measure_generation_latency

    def _one(prompt: list[dict[str, str]]) -> str:
        return batch_generate(model, tokenizer, [prompt], config.generation)[0]

    return measure_generation_latency(_one, prompts, config.latency.n_samples).as_dict()
