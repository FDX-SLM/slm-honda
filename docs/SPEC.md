> ⚠️ **DEPRECATED — historical only.** This repo now implements the **Honda Entitlement Resolver**
> (PoC6). The authoritative spec is **`PoC6_SLM_BUILD_SPEC.pdf`**; the run guide is [README.md](../README.md)
> and [RUNBOOK.md](RUNBOOK.md). The sales-coach text below is kept for engine-design reference only
> (config/training/eval/export plumbing is reused) — its domain, data-contract, and eval rubric no
> longer apply.

# Repo Specification: Fine-tuning & Evaluation Pipeline — SLM Sales Coach (iPhone)

> **This file is a brief for Claude Code.** Read it fully before generating any code. Your task: scaffold a complete, professional, runnable Python repository for **fine-tuning** and **evaluating** a Small Language Model (SLM) that acts as an iPhone sales advisor in Vietnamese. **Do NOT generate the data-creation parts** (another team owns those). The pipeline stops at **producing the model** (train → evaluate → export/quantize). There is **no serving/inference-runtime layer** in scope. The project is managed with **`uv`** (no Makefile). When finished, **print the step-by-step run guide (Section 11) at the end of the README**.

---

## 0. Instructions for Claude Code

1. Create the exact folder structure in Section 5, with every file (including empty `__init__.py`, `.gitignore`, `.env.example`).
2. Manage the project with **`uv`**: provide a `pyproject.toml` with all dependencies and project scripts; assume `uv sync` creates the environment and `uv run ...` executes commands. **Do NOT create a Makefile.**
3. The code must be **runnable**: clean imports, every CLI starts, `uv run pytest` passes basic unit tests. GPU-dependent paths must degrade safely (dry-run/mock) when no GPU is present.
4. Follow the coding standards in Section 10 (type hints, docstrings, logging, config-driven, tests).
5. All training/eval parameters come from **YAML configs** (Section 9) — never hardcode them.
6. Honor the **data contract** in Section 3 — this repo *consumes* data, it does not create it.
7. Finally, write a clear `README.md` that ends with exactly the step-by-step guide in Section 11 (commands + explanations).

---

## 1. Project context (brief)

A POC building an SLM (base model: **Qwen3.5-9B**) that acts as an iPhone sales coach in Vietnamese, using a distillation-heavy approach. Training data is produced by a separate team in three shapes (SFT, reasoning, preference) and delivered as JSONL. This repo handles **training, evaluation, and model export**, with strong requirements for complete checkpointing and stage-wise evaluation.

Training philosophy (two phases):
- **Phase 1 — SFT**: learn `sft` + `reasoning` (same SFT loss). Track T1 = fast LoRA SFT; Track T2 = multi-stage QLoRA SFT with a curriculum (broad-coverage data first → reasoning later).
- **Phase 2 — Alignment**: starting from the SFT checkpoint, train on `preference` using **ORPO** (default when ≥800 pairs) or **DPO** (2-pass when <800). Alignment teaches no new knowledge; it only sharpens preferences.

---

## 2. Scope

### IN scope (generate full code)
- Load, validate, and convert canonical data into trainer formats (consume existing JSONL).
- Fine-tuning: T1 LoRA SFT, T2 multi-stage QLoRA SFT, T3 DPO/ORPO.
- **Eval during training** + **complete checkpointing** (fully resumable, keeps best).
- Full evaluation: gold test set, 7-criteria rubric, multi-judge LLM-as-judge, **per-mode scoring**, PII leak rate, optional offline generation latency, report output.
- Export/quantize: merge LoRA → FP16 safetensors → AWQ INT4 + GGUF Q4_K_M. **This is the final deliverable.**
- Tracking: Langfuse (sample generations); quantitative metrics persisted as CSV tables + PNG charts. Tests. YAML configs. CLI via `uv run`.

### OUT of scope (do NOT generate)
- Data generation: distillation prompt templates, persona libraries, dual-agent simulation, calling teacher LLMs.
- Catalog scraping; creating the gold test set (the repo only *reads* an existing gold test set).
- **Any serving / inference-runtime / API harness** (no HTTP server, no production guardrail layer). Evaluation performs its own **offline batch generation** by loading the checkpoint directly — it does not need a serving layer.
- **No Makefile** (the project uses `uv` only).
- Note: data *loading & validation* IS in scope (training needs it); only data *creation* and *serving* are out.

---

## 3. Data contract (format the data team delivers — the repo must read it correctly)

Data is **JSON Lines** (one record per line), placed under `data/` (gitignored). Every record shares a common metadata block; content fields differ by `data_type`.

**Common metadata:** `id, data_type, mode, persona, source, lang, version, audit_status`
(`mode` ∈ the 7 conversation modes; it is metadata only — it must NOT be fed into the training sequence.)

**`data_type = "sft"`** (single-turn or multi-turn):
```json
{"id":"...","data_type":"sft","conversation_type":"single|multi_turn","mode":"...","persona":"P0x",
 "messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
 "lang":"vi","version":"v1","audit_status":"approved"}
```

**`data_type = "reasoning"`**:
```json
{"id":"...","data_type":"reasoning","mode":"...","persona":"P0x",
 "situation":"...","reasoning":["step1","step2"],"response":"...","why":"<audit-only, NOT trained>",
 "lang":"vi","version":"v1","audit_status":"approved"}
```

**`data_type = "preference"`** (for DPO/ORPO):
```json
{"id":"...","data_type":"preference","mode":"...","persona":"P0x","bad_type":"pushy",
 "prompt":[{"role":"user","content":"..."}],
 "chosen":[{"role":"assistant","content":"..."}],
 "rejected":[{"role":"assistant","content":"..."}],
 "lang":"vi","version":"v1","audit_status":"approved"}
```

**Expected data directory layout:**
```
data/sft/*.jsonl
data/reasoning/*.jsonl
data/preference/*.jsonl
data/gold/gold_test.jsonl     # gold test set for eval (labeled with mode for sliced scoring)
```

The repo ships a `data/README.md` describing this contract and, when training, keeps only `audit_status == "approved"` records (configurable filter).

---

## 4. Tech stack

- Python ≥ 3.11, managed entirely with **`uv`** (`pyproject.toml` + `uv.lock`). Pin versions in `pyproject.toml`; do not hardcode versions in code.
- Training: `transformers`, `trl`, `peft`, `unsloth`, `bitsandbytes`, `accelerate`, `datasets`, Flash Attention 2.
- Evaluation: `lm-eval-harness` (custom task), LLM-as-judge via API (OpenAI + Google GenAI), `presidio-analyzer/anonymizer` for PII leak detection.
- Config: `pydantic` (v2) + `pyyaml` (or `omegaconf`).
- CLI: `typer` (or `argparse`). Tracking: `langfuse`. Charts: `matplotlib`.
- Export: `autoawq` (AWQ INT4) + `llama.cpp` conversion (GGUF Q4_K_M).
- Quality: `ruff`, `black`, `pytest`, `pre-commit` — all invoked through `uv run`.

Define console-script entry points in `pyproject.toml` if helpful, but the canonical interface is `uv run python scripts/<name>.py`.

---

## 5. Target folder structure

```
slm-sales-coach/
├── README.md
├── pyproject.toml               # uv-managed; all deps + version pins here
├── uv.lock                      # generated by `uv lock` / `uv sync`
├── .gitignore
├── .env.example                 # API keys for judge models, Langfuse keys
├── .pre-commit-config.yaml
├── configs/
│   ├── base.yaml                # shared: base model, paths, seed, tracking
│   ├── sft_lora.yaml            # T1
│   ├── sft_multistage.yaml      # T2 (declares curriculum stages)
│   ├── align_orpo.yaml          # T3 — ORPO
│   ├── align_dpo.yaml           # T3 — DPO
│   └── eval.yaml                # evaluation config
├── src/slm_coach/
│   ├── __init__.py
│   ├── config.py                # pydantic models for all configs; merge base + override loader
│   ├── tracking.py              # init Langfuse; log sample generations (metrics → reporting/)
│   ├── reporting/               # metric CSV tables + PNG charts (loss/eval curves, per-mode bars)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── schema.py            # pydantic CanonicalRecord (sft/reasoning/preference) + validation
│   │   ├── loader.py            # read JSONL, validate, filter audit_status, split by data_type
│   │   ├── formatting.py        # canonical -> TRL formats: chatml messages; reasoning -> <think>; preference
│   │   └── mixture.py           # mixing & curriculum: single/multi ratio, stage ordering, sampling weights
│   ├── training/
│   │   ├── __init__.py
│   │   ├── model.py             # load base (Unsloth + FA2), attach LoRA/QLoRA, merge, save adapter+tokenizer
│   │   ├── sft.py               # SFTTrainer (TRL) — train-on-responses-only + multi-turn masking
│   │   ├── multistage.py        # orchestrate curriculum stages, chain checkpoints between stages
│   │   ├── align.py             # DPOTrainer / ORPOTrainer (selected via config)
│   │   └── callbacks.py         # EvalDuringTraining callback, checkpoint policy, early stopping
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── inference.py         # OFFLINE batch generation: load checkpoint, generate answers (no server)
│   │   ├── runner.py            # load gold test, drive generation, collect results
│   │   ├── rubric.py            # 7-criteria rubric + scoring scale
│   │   ├── judge.py             # multi-judge LLM-as-judge (see Section 8 — avoid circular bias)
│   │   ├── pii_guard.py         # measure PII leak rate via presidio (reuse VN recognizers)
│   │   ├── latency.py           # OPTIONAL offline generation latency (p50/p95 of generate calls)
│   │   ├── metrics.py           # aggregation: per-mode scores, totals, KPIs
│   │   ├── harness_task.py      # custom task for lm-eval-harness
│   │   └── report.py            # write eval report (markdown + json) to outputs/
│   ├── export/
│   │   ├── __init__.py
│   │   ├── merge.py             # merge LoRA -> FP16 safetensors
│   │   └── quantize.py          # AWQ INT4 (autoawq) + GGUF Q4_K_M (llama.cpp)
│   └── utils/
│       ├── __init__.py
│       ├── logging.py           # structured logging
│       └── seed.py              # set_seed for reproducibility
├── scripts/                     # thin CLI entrypoints (parse args + call src)
│   ├── validate_data.py
│   ├── train_sft.py
│   ├── train_multistage.py
│   ├── train_align.py
│   ├── evaluate.py
│   └── export_model.py
├── tests/
│   ├── __init__.py
│   ├── test_schema.py           # valid/invalid schema
│   ├── test_formatting.py       # canonical -> TRL format, multi-turn masking
│   ├── test_loader.py           # read/filter/split
│   ├── test_mixture.py
│   └── test_eval_metrics.py     # per-mode aggregation correctness
├── notebooks/
│   └── smoke_test.ipynb         # end-to-end run on a tiny model
├── data/                        # (gitignored) populated by the data team — includes data/README.md
├── checkpoints/                 # (gitignored)
└── outputs/                     # (gitignored) eval reports, exported models
```

---

## 6. Per-module requirements

**`config.py`** — pydantic models: `BaseConfig` (model_name, output_dir, seed, tracking), `LoRAConfig`, `SFTConfig`, `MultiStageConfig` (list of `StageConfig`), `AlignConfig` (method: "dpo"|"orpo", beta...), `EvalConfig`. A `load_config(path)` function reads `base.yaml` then merges the specific config's overrides.

**`data/schema.py`** — `CanonicalRecord` (validated per `data_type`), enum of the 7 modes, role/content validation. `validate_file(path)` returns stats (valid/invalid counts, distribution by mode/data_type).

**`data/formatting.py`** —
- `to_sft_dataset`: canonical sft → `messages` (ChatML) with the chat template applied; for `reasoning`, fold `<think>\n{reasoning}\n</think>\n{response}` into the assistant turn (config flag to toggle thinking, so non-thinking examples also exist).
- `to_preference_dataset`: canonical preference → `{prompt, chosen, rejected}` (explicit prompt).
- Support **multi-turn masking**: compute loss on *every* assistant turn (train-on-responses-only), not just the last one.

**`data/mixture.py`** — mix by configured ratio (e.g. ~2/3 multi-turn, 1/3 single for SFT); define stage ordering for the curriculum (stage 1: broad sft; stage 2: + reasoning + hard modes).

**`training/model.py`** — load base via Unsloth + Flash Attention 2; attach LoRA (T1) or 4-bit QLoRA (T2); save (adapter + tokenizer + config), merge (→ FP16 safetensors).

**`training/sft.py` / `multistage.py` / `align.py`** — wrap TRL `SFTTrainer` / `DPOTrainer` / `ORPOTrainer`. `align.py` selects DPO vs ORPO via config; DPO requires `--sft-checkpoint` as the starting point, ORPO is monolithic (single stage). `multistage.py` runs stages sequentially, each initialized from the previous stage's checkpoint.

**`training/callbacks.py`** — see Section 7.

**`eval/inference.py`** — load a checkpoint and run **offline batch generation** (transformers, or vLLM in offline mode) to produce model answers for the gold test. No HTTP server, no runtime harness.

**`eval/*` (others)** — see Section 8.

---

## 7. Fine-tuning requirements (complete checkpointing + eval)

**Checkpointing (must be complete):**
- `save_strategy = "steps"` (configurable `save_steps`) **and** at the end of each epoch.
- Save **adapter (LoRA) + tokenizer + config + trainer state + optimizer/scheduler state** so training is fully **resumable** (`--resume <path>`).
- `save_total_limit` to cap disk usage; always keep a distinct **best** and **last** checkpoint.
- `load_best_model_at_end = True`, select best by `metric_for_best_model` (e.g. mean rubric score or judge win-rate on a small eval set).
- Each checkpoint ships a `meta.json`: config, git commit, data version, seed, metrics at that step.

**Eval during training (required):**
- An `EvalDuringTraining` callback runs every `eval_steps`: take a **gold-test subset** (fast), generate answers, score with a reduced rubric + (optional) judge, inject the metric for best-model selection, and log a sample generation to Langfuse.
- `early_stopping` on the primary metric (configurable patience).
- Log train/eval loss, learning rate, and sample generations (via Langfuse) for qualitative tracking.

**Reproducibility:** set a global seed; record all config + data version in each checkpoint's `meta.json`.

---

## 8. Evaluation requirements (complete, per plan)

**Gold test set** (`data/gold/gold_test.jsonl`, labeled with `mode`): the model generates answers for each case via `eval/inference.py` (offline).

**7-criteria rubric** (`rubric.py`), each on a 1–5 scale: factuality, helpfulness, tone, completeness, safety, format, language_quality (Vietnamese). Weighted scoring is configurable.

**Multi-judge LLM-as-judge** (`judge.py`):
- Use **GPT + Gemini** as judges. **Do NOT use Claude/DeepSeek as judges** — they are the teachers that produced the data, so using them risks *circular / self-preference bias*.
- Support rubric scoring (per-criterion) and pairwise (A/B) comparison when needed.
- Aggregate by mean/majority across judges; record judge disagreement.

**Per-mode scoring** (`metrics.py`): aggregate scores **broken down by the 7 modes** (e.g. objection_handling 6.2/10, comparison 8.9/10) to reveal where the model is weak → feedback to the data team to reinforce the right slices. This is the most important eval output.

**PII leak rate** (`pii_guard.py`): run presidio (reuse the Vietnamese custom recognizers: CCCD, phone, license plate, card...) over generated outputs to measure the PII leak rate — target KPI < 0.5%.

**Latency (optional, offline)** (`latency.py`): measure p50/p95 of `model.generate` calls during offline generation. This is raw generation timing, NOT a serving benchmark.

**Report** (`report.py`): write `outputs/eval/<run>/report.md` + `report.json` containing: per-mode score table, mean rubric scores, judge agreement, PII leak rate, latency, and a comparison against a previous baseline if available.

**lm-eval-harness** (`harness_task.py`): register a custom task so checkpoints can be run inside the lm-eval-harness framework (for standardized benchmarking + cross-checkpoint comparison).

---

## 9. Config-driven design (condensed examples — Claude Code generates full versions)

`configs/base.yaml`
```yaml
model_name: "Qwen/Qwen3.5-9B"
output_dir: "checkpoints"
seed: 42
tracking: { langfuse: true }
data: { dir: "data", keep_audit_status: ["approved"], lang: "vi" }
```

`configs/sft_multistage.yaml`
```yaml
defaults: base.yaml
lora: { r: 16, alpha: 32, dropout: 0.05, target_modules: "all-linear" }
quant: { load_in_4bit: true }            # QLoRA for T2
sft: { epochs: 2, lr: 2.0e-4, batch_size: 4, grad_accum: 4,
       max_seq_len: 4096, train_on_responses_only: true, multiturn_masking: true,
       save_steps: 200, eval_steps: 200, save_total_limit: 3,
       load_best_model_at_end: true, metric_for_best_model: "rubric_avg",
       early_stopping_patience: 3 }
stages:                                   # curriculum
  - { name: "broad",     include: ["sft"],              mix: { multi_turn: 0.66, single: 0.34 } }
  - { name: "reasoning", include: ["sft","reasoning"],  reasoning_thinking: true }
```

`configs/align_orpo.yaml`
```yaml
defaults: base.yaml
align: { method: "orpo", beta: 0.1, lr: 5.0e-6, epochs: 1 }   # >=800 pairs -> ORPO
sft_checkpoint: null                      # ORPO is monolithic; for DPO point to the SFT checkpoint
```

`configs/eval.yaml`
```yaml
defaults: base.yaml
gold: "data/gold/gold_test.jsonl"
rubric_weights: { factuality: 2, safety: 2, helpfulness: 1.5, tone: 1, completeness: 1, format: 0.5, language_quality: 1 }
judges: ["gpt", "gemini"]                 # do NOT use claude/deepseek
pii: { enabled: true, target_leak_rate: 0.005 }
latency: { measure: true, n_samples: 50 }
per_mode_breakdown: true
```

---

## 10. Coding standards

- Full type hints; Google-style docstrings on every public function/class.
- Structured logging (`utils/logging.py`); no `print` in `src/`.
- Strictly config-driven; no hardcoded paths/hyperparameters/secrets.
- Secrets via `.env` (`.env.example` lists required keys: `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `LANGFUSE_*`).
- `set_seed` at every train/eval entrypoint.
- Unit tests for schema, formatting (incl. multi-turn masking), loader, mixture, per-mode metrics.
- `ruff` + `black` + `pre-commit`, all run via `uv run` (e.g. `uv run ruff check .`). Minimal CI (lint + test) if generating `.github/workflows`, also using `uv`.
- `scripts/` only parse args + call `src/`; logic lives in `src/slm_coach`.
- GPU-dependent code must degrade safely without a GPU (allow dry-run/mock to test the pipeline).

---

## 11. How to run — step by step (print this section verbatim at the end of the README)

> Everything runs through `uv`. `uv run <cmd>` executes `<cmd>` inside the project's virtual environment. You never need to manually activate a venv.

### Step 0 — One-time setup

```bash
# Install uv (one time, if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repo root: create the venv and install all dependencies from pyproject.toml
uv sync

# Create your local secrets file and fill in the judge API keys + tracking URIs
cp .env.example .env
#   edit .env: OPENAI_API_KEY, GOOGLE_API_KEY (for eval judges), LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST

# Sanity check: run the unit tests (no GPU / no API keys needed)
uv run pytest
```

**What this does:** `uv sync` builds an isolated environment with the exact pinned dependencies. The tests confirm the data schema, formatting/masking, loader, and metrics logic all work before you spend any GPU time. **Check:** all tests pass.

### Step 1 — Validate the data (from the data team)

```bash
uv run python scripts/validate_data.py --data-dir data/
```

**Purpose:** confirm the delivered JSONL matches the data contract (Section 3) before training. **Requires:** `data/{sft,reasoning,preference}/*.jsonl` present. **Produces:** a console/JSON report of valid vs invalid records and the distribution by `mode` and `data_type`. **Check:** invalid count is 0 (or acceptable) and all 7 modes are represented — a skewed distribution here predicts a weak model later.

### Step 2 — (Optional) Train T1: fast LoRA SFT baseline

```bash
uv run python scripts/train_sft.py --config configs/sft_lora.yaml
```

**Purpose:** a quick, cheap baseline to confirm the whole training loop works and to give you an early reference point. **Requires:** valid SFT data, a GPU. **Produces:** checkpoints under `checkpoints/sft_lora/` (keeps `best` and `last`). **Check:** training loss decreases and `checkpoints/sft_lora/best/` exists. You can skip straight to Step 3 if you don't need the baseline.

### Step 3 — Train T2: multi-stage QLoRA SFT (the real SFT)

```bash
uv run python scripts/train_multistage.py --config configs/sft_multistage.yaml
```

**Purpose:** the main supervised model. It runs the curriculum: stage `broad` (sft only, ~2/3 multi-turn) → stage `reasoning` (sft + reasoning, with `<think>` folding). Each stage starts from the previous stage's checkpoint. **Requires:** valid SFT + reasoning data, a GPU. **Produces:** `checkpoints/sft_multistage/best/` and `.../last/`, plus a `meta.json` per checkpoint. Eval-during-training runs every `eval_steps` and logs to MLflow. **Check:** `checkpoints/sft_multistage/best/` exists; the rubric metric improved across stages in MLflow.

### Step 4 — Alignment (Phase 2): ORPO or DPO

Choose based on how many preference pairs you have (the rule of thumb: **≥800 pairs → ORPO**, otherwise **DPO**).

```bash
# Option A — ORPO (monolithic; recommended when you have enough preference pairs)
uv run python scripts/train_align.py --config configs/align_orpo.yaml

# Option B — DPO (needs the SFT checkpoint from Step 3 as its starting point)
uv run python scripts/train_align.py --config configs/align_dpo.yaml \
       --sft-checkpoint checkpoints/sft_multistage/best
```

**Purpose:** sharpen the model toward senior-style answers (chosen) over weak ones (rejected). DPO must start from the SFT checkpoint; ORPO combines the SFT-like and preference objectives in one stage. **Requires:** valid preference data, a GPU. **Produces:** `checkpoints/align_orpo/best/` (or `align_dpo/best/`). **Check:** the aligned checkpoint exists and scores higher than the SFT checkpoint when you run Step 5.

### Step 5 — Evaluate

```bash
uv run python scripts/evaluate.py --config configs/eval.yaml \
       --model checkpoints/align_orpo/best
```

**Purpose:** the full evaluation. It loads the checkpoint, generates answers for the gold test offline, scores them with the 7-criteria rubric + GPT/Gemini judges, computes the **per-mode breakdown**, PII leak rate, and (optional) latency. **Requires:** `data/gold/gold_test.jsonl`, judge API keys in `.env`. **Produces:** `outputs/eval/<run>/report.md` + `report.json`. **Check:** open `report.md` — the per-mode table tells you exactly which modes are weak so the data team can reinforce them. You can run this on any checkpoint (SFT vs aligned) to compare.

### Step 6 — Export / quantize (final deliverable)

```bash
uv run python scripts/export_model.py --checkpoint checkpoints/align_orpo/best --formats awq,gguf
```

**Purpose:** turn the best checkpoint into deployable model files. It merges the LoRA adapter into FP16 safetensors, then quantizes to AWQ INT4 (GPU) and GGUF Q4_K_M (CPU/edge). **Requires:** the chosen best checkpoint. **Produces:** `outputs/exported/awq/` and `outputs/exported/gguf/`. **Check:** both folders contain the quantized model — this is the end state of the pipeline.

### Utilities

```bash
# Resume an interrupted training run from its last checkpoint
uv run python scripts/train_sft.py --config configs/sft_lora.yaml --resume checkpoints/sft_lora/last

# Lint & format
uv run ruff check .
uv run black .

# Add a new dependency later
uv add <package>
```

**Typical full sequence:** Step 0 → 1 → 3 → 4 → 5 → 6 (Step 2 is an optional baseline).

---

## 12. Definition of Done

1. `uv sync` succeeds; all modules import cleanly.
2. `uv run pytest` passes (schema, formatting + masking, loader, mixture, per-mode metrics).
3. Every CLI script supports `--help`; runs in dry-run/mock without a GPU or API keys.
4. Training produces complete checkpoints (adapter + tokenizer + trainer/optimizer state + `meta.json`), is resumable, and keeps both best + last.
5. Evaluation writes a `report.md` with a **per-mode score table**, rubric, judges (GPT+Gemini), PII leak rate, and latency.
6. Export produces the final quantized model under `outputs/exported/` (AWQ INT4 + GGUF Q4_K_M) — the pipeline's end state.
7. README is clear and ends with exactly the step-by-step guide in Section 11.
8. No Makefile; everything runs through `uv`. No hardcoded secrets; all parameters live in `configs/*.yaml`.
