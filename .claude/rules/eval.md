---
description: Conventions for the Honda Entitlement Resolver evaluation (oracle KPIs, confusion, report).
globs:
  - "src/slm_coach/eval/**"
  - "scripts/evaluate.py"
  - "scripts/compare_models.py"
  - "scripts/rag_baseline.py"
  - "configs/eval.yaml"
---

# Evaluation rules (Honda Entitlement Resolver — PoC6 §7)

## How eval runs the model
- **Offline batch generation** only (load the checkpoint directly via transformers/vLLM offline).
  There is **no serving layer / HTTP server** — the model is closed-book at inference.

## Scoring is the DETERMINISTIC oracle, not an LLM rubric
- `slm_coach.eval.honda` parses each output (`<think>` + JSON resolution package) and scores it with
  `slm_coach.oracle` against `slm_coach.ground_truth`. No LLM judge is in the scoring path.
- KPIs (targets in parentheses):
  - **RC accuracy + confusion matrix** over 3 RC + `INSUFFICIENT_EVIDENCE` (leading-RC clear-cue ≥85%).
  - **Cue-grounding faithfulness** — % of `evidence_in_ticket` present in the complaint (≥95%).
  - **No-fabricated-telemetry rate** — % outputs with no invented telemetry (≥98%, top honesty KPI).
  - **Runbook completeness & fidelity** — required fields present + match the gold runbook (≥95%).
  - **Calibration** — ECE + overconfident-wrong rate (penalize confidence>0.85 when wrong).
  - **Abstention hallucination** — on ambiguous/out-of-catalog cases, % that forced an RC (<10%).
  - **Artifact valid@1** (≥90%) and **latency** p50/p95 of `model.generate` (<1.5s, NOT a serving bench).

## Output & robustness
- Write `outputs/eval/<run>/report.md` + `report.json` + `per_sample.csv`: KPI table (PASS/FAIL vs
  target), confusion matrix, per-slice accuracy, latency.
- `eval.jsonl` uses seed 999 (held out from train); `eval_hard.jsonl` (20 hand-written messy
  complaints) is reported separately via `--hard`.
- `scripts/compare_models.py` builds the leaderboard across the 4 base models from report.jsons.
- `scripts/rag_baseline.py` is the cue-blind foil (copies the nearest incident → wrong RC on cue-flip,
  can never abstain) for the demo split-screen.
- Reproducible: `set_seed`; record model checkpoint + data version in the run.
