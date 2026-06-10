# CLAUDE.md — SLM Sales Coach (fine-tuning & eval pipeline)

Persistent project memory. Keep this short; full design lives in `docs/SPEC.md` (read it before scaffolding or making structural changes).

## What this repo is
Fine-tunes and evaluates a Vietnamese iPhone sales-coach SLM (base: Qwen3.5-9B). The pipeline goes **data → train → evaluate → export**, and **stops at producing the model**.

## Hard scope boundaries
- IN: data loading/validation, training (T1 LoRA SFT, T2 multi-stage QLoRA SFT, T3 DPO/ORPO), eval, export/quantize.
- OUT: data **generation** (distillation, personas, dual-agent) — another team owns it. No serving / inference-runtime / API harness. **No Makefile.**

## Workflow — `uv` only
- Install: `uv sync`. Run anything: `uv run <cmd>` (e.g. `uv run python scripts/train_sft.py --config configs/sft_lora.yaml`).
- Tests: `uv run pytest`. Lint/format: `uv run ruff check .` / `uv run black .`.
- Add a dep: `uv add <pkg>`. Never edit a venv by hand.

## Conventions (non-negotiable)
- Config-driven: every hyperparameter lives in `configs/*.yaml`. Never hardcode params/paths/secrets.
- Secrets in `.env` only (never commit). `data/`, `checkpoints/`, `outputs/` are gitignored.
- Full type hints + Google-style docstrings. Structured logging via `utils/logging.py` — no `print` in `src/`.
- `set_seed` at every train/eval entrypoint. `scripts/` stay thin (parse args → call `src/slm_coach`).
- GPU code must degrade to dry-run/mock when no GPU is present.

## Data is consumed, not created
Canonical JSONL under `data/{sft,reasoning,preference}/` + `data/gold/gold_test.jsonl`. Keep only `audit_status == "approved"`. `mode` is metadata — it must never enter the training sequence. See `docs/SPEC.md` §3 for the exact schema.

## Training shape (reminder)
Two phases: SFT (T1/T2) → alignment (T3). DPO needs an SFT checkpoint as its start; ORPO is monolithic. Rule of thumb: ≥800 preference pairs → ORPO, else DPO. Detailed rules auto-load from `.claude/rules/` when editing `src/slm_coach/training` or `src/slm_coach/eval`.

## Definition of done
`uv sync` clean, `uv run pytest` green, every CLI has `--help` and runs in dry-run without GPU/keys, training writes complete resumable checkpoints (best + last + `meta.json`), eval produces a per-mode report, export yields AWQ INT4 + GGUF Q4_K_M. README ends with the step-by-step run guide.
