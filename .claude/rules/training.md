---
description: Conventions for the fine-tuning code (SFT, multi-stage, alignment, checkpoints).
globs:
  - "src/slm_coach/training/**"
  - "scripts/train_*.py"
  - "configs/sft_*.yaml"
  - "configs/align_*.yaml"
---

# Training rules

## Two-phase shape
- Phase 1 = SFT: `sft` + `reasoning` data (same SFT loss). T1 = LoRA SFT (fast baseline); T2 = multi-stage QLoRA SFT (curriculum: broad sft → + reasoning).
- Phase 2 = Alignment on `preference`: **DPO requires an SFT checkpoint as its starting point**; **ORPO is monolithic** (single stage). Rule of thumb: ≥800 pairs → ORPO, else DPO. Method is chosen via config, not hardcoded.

## Checkpointing (must be complete & resumable)
- Save adapter (LoRA) + tokenizer + config + trainer state + optimizer/scheduler state.
- `save_strategy="steps"` (+ end of each epoch); `save_total_limit` set; always keep distinct **best** and **last**.
- `load_best_model_at_end=True`, best chosen by `metric_for_best_model` from config.
- Every checkpoint writes `meta.json`: config snapshot, git commit, data version, seed, metrics.
- Support `--resume <path>` to continue from a checkpoint with optimizer state intact.

## Data into training
- Use TRL trainers (`SFTTrainer`, `DPOTrainer`, `ORPOTrainer`); load base via Unsloth + Flash Attention 2; QLoRA 4-bit for T2.
- **Multi-turn masking**: compute loss on *every* assistant turn (train-on-responses-only), not just the last.
- `reasoning` records: fold `<think>\n{reasoning}\n</think>\n{response}` into the assistant turn; keep some non-thinking examples too (config flag). **Never** train the `why` field.
- `mode`, `persona`, and other metadata must **never** enter the tokenized training sequence.

## Always
- Read every hyperparameter from `configs/*.yaml`. Call `set_seed`. Persist params/metrics as CSV tables + PNG charts via `slm_coach.reporting`; log sample generations to Langfuse.
- Run the eval-during-training callback every `eval_steps` and respect early stopping.
- GPU-dependent paths must degrade to a dry-run/mock when no GPU is available.
