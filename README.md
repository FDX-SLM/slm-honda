# Honda Entitlement Resolver — SLM fine-tuning & evaluation (PoC6)

Fine-tunes and evaluates a **closed-book diagnostic SLM** for Honda support/operations. Input is a
**raw customer complaint** (no error code, no logs); the model reads the linguistic **cues**, infers
the **root cause** (a calibrated differential, not a guess), and emits a full **resolution package**
(`<think>` reasoning + a JSON diagnosis + runbook business fields + RCA/work-order/email/mermaid
artifacts). It is trained on **100% synthetic data generated from ground truth** and **never invents
telemetry**. Pipeline: **generate → train → evaluate → export**, and it **stops at producing the model**.

> Authoritative spec: `PoC6_SLM_BUILD_SPEC.pdf`. Data/runbooks/output are **English**; code comments
> and explanations may be Vietnamese.

## Three root causes (differentiated by cue, §1.2)

| RC | Runbook | Cue in the complaint |
| --- | --- | --- |
| `TCU_OFFLINE` | RB-TCU-04 | car parked underground/garage, "no signal", remote command times out, shows active but car doesn't respond |
| `ENTITLEMENT_CACHE_STALE` | RB-CACHE-02 | active on web but not the app, worked-then-stopped, intermittent, re-login helps |
| `ELIGIBILITY_RULE_CONFLICT` | RB-ELIG-05 | app keeps prompting Subscribe despite payment, region/trim/plan combo (e.g. CR-V Touring US-West) |
| `INSUFFICIENT_EVIDENCE` (abstain) | — | no distinguishing cue, or out-of-catalog (403, billing, app crash, OTA) → route to a human |

Everything factual (systems, runbooks, incidents, eligibility matrix) lives in
[`src/slm_coach/ground_truth.py`](src/slm_coach/ground_truth.py); every generated sample is gated by
the [graph oracle](src/slm_coach/oracle.py) (no fabricated telemetry, cue-grounded evidence, runbook
fidelity, calibrated confidence).

---

## Install

Everything runs through [`uv`](https://docs.astral.sh/uv/) (Python 3.12, pinned in `.python-version`).

```bash
uv sync                 # CPU core — enough to generate data, validate, dry-run, run pytest
uv run pytest           # no GPU / no API keys
```

On the **GPU training/eval box** (Linux + CUDA) add the heavy extras:

```bash
uv sync --extra train --extra gpu --extra export --extra viz --extra tracking
cp .env.example .env    # optional: Langfuse keys. (Eval is oracle-based — no judge API keys needed.)
```

GPU/optional modules import lazily and degrade safely: every training CLI has `--dry-run` and the
evaluator has `--mock`, so the whole pipeline wires up with **no GPU and no keys**.

---

## Run guide (step by step)

### 0 — Environment
```bash
uv sync --extra train --extra gpu --extra export --extra viz   # GPU box
# or just `uv sync` to generate data / dry-run on CPU
```

### 1 — Generate the data (from ground truth, gated by the oracle)
```bash
uv run python scripts/gen_sft.py  --seed 42  --out data/sft/train_sft.jsonl          # ~2.3k SFT (5 groups)
uv run python scripts/gen_dpo.py  --seed 42  --out data/preference/dpo_pairs.jsonl   # ~600 DPO pairs (6 types)
uv run python scripts/gen_eval.py --seed 999 --out data/gold/gold_test.jsonl         # 180 eval + 20 eval_hard (seed 999)
uv run python scripts/validate_data.py --data-dir data/                              # schema check
```
Smoke (tiny, for a quick look): add `--limit 30`. Each gen script prints its per-slice distribution.

### 2 — (optional) Stratified holdout
```bash
uv run python scripts/split_holdout.py --config configs/sft.yaml   # materializes data/holdout/{train,val}.jsonl
```

### 3 — Train SFT (model-agnostic — switch base with `--base`)
```bash
uv run python scripts/train_sft.py --config configs/sft.yaml --base qwen      # → checkpoints/sft_qwen/best
# also: --base gemma | phi | granite   (Qwen/Qwen3.5-9B · google/gemma-4-12B-it · microsoft/phi-4 · ibm-granite/granite-4.1-8b)
```
Training writes report artifacts under `checkpoints/sft_<base>/metrics/`: `run_facts.csv/.md`
(base, method, precision, **gradient_checkpointing**, effective batch, masking, data counts),
`training_log.csv`, `training_summary.md`, and charts `loss_curve.png` / `eval_metric.png` /
`lr_schedule.png` / `grad_norm.png` (charts need the `viz` extra).

### 4 — Evaluate SFT (oracle KPIs)
```bash
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/sft_qwen/best \
    --base qwen --run-name eval_sft_qwen
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/sft_qwen/best \
    --base qwen --hard --run-name eval_hard_qwen   # 20 hand-written hard cases, reported separately
```
Report: `outputs/eval/<run>/report.{md,json}` + `per_sample.csv` — RC accuracy, confusion (3 RC +
ABSTAIN), cue-grounding faithfulness, no-fabricated-telemetry, runbook completeness/fidelity,
calibration (ECE), abstention hallucination, artifact valid@1, latency p50/p95.

### 5 — DPO alignment (continues the SFT checkpoint)
```bash
uv run python scripts/train_align.py --config configs/dpo.yaml --base qwen \
    --sft-checkpoint checkpoints/sft_qwen/best        # → checkpoints/dpo_qwen/best
uv run python scripts/evaluate.py --config configs/eval.yaml --model checkpoints/dpo_qwen/best \
    --base qwen --run-name eval_dpo_qwen
```

### 6 — Export (the deliverable)
```bash
uv run python scripts/export_model.py --checkpoint checkpoints/dpo_qwen/best --formats gguf,awq
```

### 7 — Compare the 4 base models → pick the winner
```bash
uv run python scripts/compare_models.py --eval-root outputs/eval --out outputs/eval/leaderboard.md
```

### 8 — Demo + RAG baseline (the money shot)
```bash
uv run python scripts/rag_baseline.py --gold data/gold/gold_test.jsonl   # cue-blind foil (lower accuracy, can't abstain)
HONDA_ADAPTER=checkpoints/dpo_qwen/best uv run streamlit run app.py       # split-screen SLM vs RAG
```
The demo runs in **DEMO mode from ground truth** if no adapter/GPU is present, so it works offline for
screenshots.

---

## Train all 4 bases in one loop

```bash
for M in qwen gemma phi granite; do
  uv run python scripts/train_sft.py   --config configs/sft.yaml --base $M
  uv run python scripts/train_align.py --config configs/dpo.yaml --base $M --sft-checkpoint checkpoints/sft_$M/best
  uv run python scripts/evaluate.py    --config configs/eval.yaml --model checkpoints/dpo_$M/best --base $M --run-name eval_dpo_$M
done
uv run python scripts/compare_models.py --eval-root outputs/eval
```

## Project layout

```
src/slm_coach/
  ground_truth.py   3 RC + 3 runbooks (§2.1) + render_runbook (§2.3) + cue library + eligibility matrix + incidents + system prompt
  oracle.py         graph oracle (§4): cue-grounding · no-fabricated-telemetry · RC↔cue · runbook fidelity · calibration
  model_registry.py the 4 base models → {hf_id, dtype, sampling, think_native, notes}
  datagen/          core (complaint→<think>→resolution+artifacts) · sft (5 groups) · dpo (6 types) · evalset (eval + eval_hard)
  data/             schema · loader · formatting (apply_chat_template, <think>, multi-turn masking) · split
  training/         model (Unsloth optional, SDPA attention, LoRA/QLoRA) · sft · align (DPO) · callbacks
  eval/             honda (oracle KPIs + report) · inference (offline batch) · rag (baseline) · latency
  export/           merge (LoRA→FP16) · quantize (AWQ INT4 + GGUF Q4_K_M)
  reporting/        run_facts + training_log/summary + per-mode/eval tables · charts (loss/lr/grad_norm/per-mode)
scripts/            gen_sft · gen_dpo · gen_eval · validate_data · split_holdout · train_sft · train_align
                    evaluate · export_model · compare_models (leaderboard) · compare_pair (head-to-head) · rag_baseline
app.py              Streamlit demo (split-screen SLM vs RAG)
```

## Conventions

- **Config-driven:** every hyperparameter lives in `configs/*.yaml`. Switch the base model with
  `--base` (registry) — data and chat template are never hardcoded (`tokenizer.apply_chat_template`).
- **Honesty:** the model reasons only from cues in the complaint; the oracle rejects any sample that
  invents telemetry or an ungrounded cue. Confidence ≤ 0.85 from a raw complaint; abstention ≤ 0.45.
- **Secrets in `.env` only.** `data/`, `checkpoints/`, `outputs/` are gitignored.

See [docs/RUNBOOK.md](docs/RUNBOOK.md) for the end-to-end pipeline detail and
[data/README.md](data/README.md) for the data contract.
