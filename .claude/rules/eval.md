---
description: Conventions for the evaluation code (gold test, rubric, judges, per-mode, PII, report).
globs:
  - "src/slm_coach/eval/**"
  - "scripts/evaluate.py"
  - "configs/eval.yaml"
---

# Evaluation rules

## How eval runs the model
- Use **offline batch generation** only (load the checkpoint directly via transformers or vLLM offline mode). There is **no serving layer / HTTP server / runtime harness** in this repo.

## Scoring
- 7-criteria rubric (1–5 each): factuality, helpfulness, tone, completeness, safety, format, language_quality. Weights come from `configs/eval.yaml`.
- **Multi-judge LLM-as-judge: use GPT + Gemini ONLY.** **Never use Claude or DeepSeek as judges** — they are the teacher models that generated the data, so judging with them causes circular / self-preference bias. Aggregate by mean/majority and record judge disagreement.
- **Per-mode breakdown is the primary output**: report scores sliced by each of the 7 conversation modes (e.g. objection_handling 6.2/10) so the data team knows which slices to reinforce.

## KPIs
- PII leak rate via presidio with the Vietnamese custom recognizers (CCCD, phone, license plate, card...). Target < 0.5% (configurable).
- Optional offline generation latency: p50/p95 of `model.generate` (NOT a serving benchmark).

## Output & robustness
- Write `outputs/eval/<run>/report.md` + `report.json`: per-mode table, mean rubric, judge agreement, PII leak rate, latency, and comparison vs a previous baseline if present.
- Judge API keys come from `.env`; handle rate limits / errors with retries and fail gracefully.
- Provide a custom `lm-eval-harness` task so checkpoints can be benchmarked in a standardized way.
- Reproducible: `set_seed`; record model checkpoint + data version in the run.
