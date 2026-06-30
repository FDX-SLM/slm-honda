# CLAUDE.md — Honda Entitlement Resolver SLM (PoC6)

Persistent project memory. Keep this short. The authoritative spec is **`PoC6_SLM_BUILD_SPEC.pdf`**
(read it before scaffolding or structural changes).

## What this repo is
Fine-tunes and evaluates a **closed-book diagnostic SLM** for Honda support. Input = a **raw customer
complaint** (no error code, no logs). The model reads linguistic **cues** → infers a **root cause**
(calibrated differential + abstention) → emits a **resolution package** (`<think>` + JSON diagnosis +
runbook business fields + RCA/work-order/email/mermaid artifacts). Pipeline: **generate → train →
evaluate → export**, and **stops at producing the model**. Data/runbooks/output are **English**;
comments/explanations may be Vietnamese.

## Hard scope boundaries
- IN: **data generation from ground truth** (gated by the oracle), training (SFT LoRA/QLoRA → DPO),
  evaluation (oracle KPIs), export/quantize, RAG-baseline + demo.
- OUT: no serving / inference-runtime / API harness at inference (closed-book). **No Makefile.**

## Workflow — `uv` only
- Install: `uv sync` (CPU core is enough to generate data, validate, dry-run, test). Run: `uv run <cmd>`.
- Tests: `uv run pytest`. Lint/format: `uv run ruff check .` / `uv run black .`. Add a dep: `uv add <pkg>`.

## Three root causes (differentiated by cue — §1.2)
`TCU_OFFLINE` (RB-TCU-04, garage/no-signal/timeout) · `ENTITLEMENT_CACHE_STALE` (RB-CACHE-02,
active-on-web-not-app/intermittent) · `ELIGIBILITY_RULE_CONFLICT` (RB-ELIG-05, keeps prompting
Subscribe / region-trim-plan combo) · `INSUFFICIENT_EVIDENCE` (abstain: no cue or out-of-catalog).

## Ground truth & oracle (non-negotiable)
- All facts (systems, runbooks §2.1, `render_runbook` §2.3, cue library, eligibility matrix, incidents,
  Appendix A system prompt) live in `src/slm_coach/ground_truth.py` — **never invent facts**.
- Every generated sample passes `src/slm_coach/oracle.py` (§4): cue-grounding, **no fabricated
  telemetry** (KPI honesty), RC↔cue match, runbook fidelity, calibration (≤0.85; abstain ≤0.45).
- The model reasons only from cues in the complaint; `<think>` must never assert telemetry
  (timestamps, "record found", "delivered at T+28s"). `mode` is a slice tag — never trained.

## Conventions
- Config-driven (`configs/*.yaml`). **Model-agnostic**: switch base with `--base` via
  `model_registry.py`; render via `tokenizer.apply_chat_template` (never hardcode templates).
- Mask loss on the assistant turn only (incl. `<think>`); keep `<think>` literal for non-native models.
- Secrets in `.env` only. `data/`, `checkpoints/`, `outputs/` gitignored. Full type hints +
  Google-style docstrings; structured logging via `utils/logging.py` (no `print` in `src/`).
- `set_seed` at every entrypoint. GPU code degrades to dry-run/mock without a GPU.

## The 4 base models (model-agnostic, §6.1)
`Qwen/Qwen3.5-9B` (qwen) · `google/gemma-4-12B-it` (gemma) · `microsoft/phi-4` (phi) ·
`ibm-granite/granite-4.1-8b` (granite). No Mistral. Verify HF ids before downloading.

## Definition of done
`uv sync` clean, `uv run pytest` green, every CLI has `--help` and runs in dry-run/mock without
GPU/keys. gen scripts produce 5 SFT groups + 6 DPO types (oracle-gated, balanced 3 RC). Training writes
resumable checkpoints (best + last + `meta.json`) **and report artifacts** (run_facts incl.
gradient_checkpointing, training_log/summary, loss/lr/grad_norm charts). Eval produces the oracle-KPI
report (confusion 3-RC + ABSTAIN, cue-faithfulness ≥95%, no-fabrication ≥98%, runbook-completeness
≥95%, abstention hallucination <10%, latency <1.5s). `compare_models` ranks the 4 bases. Demo runs the
SLM-vs-RAG split-screen. README ends with the step-by-step run guide.
