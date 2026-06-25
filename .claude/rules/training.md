---
description: Conventions for the Honda Entitlement Resolver training (SFT, DPO, checkpoints, model-agnostic).
globs:
  - "src/slm_coach/training/**"
  - "scripts/train_*.py"
  - "src/slm_coach/model_registry.py"
  - "configs/sft.yaml"
  - "configs/dpo.yaml"
---

# Training rules (Honda Entitlement Resolver — PoC6 §6)

## Two-phase shape
- Phase 1 = SFT on the 5 SFT groups (same SFT loss): complaint→resolution, knowledge augmentation,
  differential, distractors, abstention. LoRA (fast) or QLoRA 4-bit (config-driven via `quant`).
- Phase 2 = DPO on the 6 preference types. **DPO requires an SFT checkpoint as its starting point**
  (`--sft-checkpoint`). ~600 pairs → DPO (not ORPO). Method is config-driven, never hardcoded.

## Model-agnostic (the key principle, §6.2)
- **One shared dataset** in neutral `messages` format — never rewritten per model.
- **Never hardcode the chat template.** Each base renders via
  `tokenizer.apply_chat_template(messages, tokenize=False)`. Switch base with `--base` (alias or HF id)
  resolved by `model_registry.py` (the 4 models: qwen, gemma, phi, granite — no Mistral).
- Keep `<think>` literal inside `assistant.content` so non-native models (Gemma/Granite) learn it;
  verify the template does not escape/swallow `<` `>`.

## Data into training
- Use TRL trainers (`SFTTrainer`, `DPOTrainer`); QLoRA 4-bit for the big bases.
- **Mask loss on the assistant turn only** (incl. the `<think>` block) via `assistant_only_loss` /
  train-on-responses-only; degrade gracefully if the template lacks generation markers.
- `mode` (slice tag), `persona`, and other metadata must **never** enter the tokenized sequence.
- The model must never be trained to assert telemetry — that is enforced upstream by the oracle at
  generation time, so trust the data but keep the system prompt's honesty rules intact.

## Checkpointing (complete & resumable)
- Save adapter + tokenizer + config + trainer state + optimizer/scheduler. `save_strategy="steps"`
  (+ epoch end); `save_total_limit` set; always keep distinct **best** and **last**.
- `load_best_model_at_end=True`, best by `metric_for_best_model`. Every checkpoint writes `meta.json`
  (config snapshot, git commit, data version, seed, metrics). Support `--resume <path>`.

## Always
- Read every hyperparameter from `configs/*.yaml`. Call `set_seed`. Run the eval-during-training
  callback every `eval_steps` and respect early stopping.
- Persist report artifacts via `slm_coach.reporting`: `run_facts` (base, method, precision,
  **gradient_checkpointing**, effective batch, masking, data counts), `training_log.csv`,
  `training_summary.md`, and charts (loss / eval metric / LR / grad_norm). Log samples to Langfuse.
- GPU-dependent paths degrade to a dry-run/mock when no GPU is available.
