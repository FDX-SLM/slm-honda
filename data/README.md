# Data contract (generated from ground truth)

This directory is **gitignored**. Unlike the earlier sales PoC (which consumed data), PoC6
**generates** all data from [`src/slm_coach/ground_truth.py`](../src/slm_coach/ground_truth.py) and
gates every sample through the [graph oracle](../src/slm_coach/oracle.py). All content is **English**.

```
data/
├── sft/*.jsonl          # supervised fine-tuning (5 groups, §5.1–5.5)
├── preference/*.jsonl   # DPO preference pairs (6 types, §5.6)
├── gold/
│   ├── gold_test.jsonl  # eval set (seed 999), labeled with the expected root cause
│   └── eval_hard.jsonl  # 20 hand-written messy complaints (cue hidden in noise)
└── samples/             # inspection-only (NOT trained): e.g. authored_sft.jsonl
```

> ⚠️ Training globs **every** `*.jsonl` under `data/sft/` (and `data/preference/`). Keep **only the
> training file** there. The pure-authored inspection copy (`--source authored`) goes to
> `data/samples/`, not `data/sft/`, so it is not double-loaded into training.

Generate + validate:

```bash
uv run python scripts/gen_sft.py  --seed 42  --out data/sft/train_sft.jsonl
uv run python scripts/gen_dpo.py  --seed 42  --out data/preference/dpo_pairs.jsonl
uv run python scripts/gen_eval.py --seed 999 --out data/gold/gold_test.jsonl
uv run python scripts/validate_data.py --data-dir data/
```

## Common metadata (every record)

`id, data_type, mode, persona, lang, version, audit_status`

- `mode` is a **slice tag** (metadata only — never enters the training sequence):
  `tcu_offline, cache_stale, eligibility, abstention, knowledge, differential, distractor`.
- `lang` is `en`; only `audit_status == "approved"` records are trained on.

## `data_type = "sft"`

```json
{"id":"sft-cr-00001","data_type":"sft","conversation_type":"single","mode":"tcu_offline",
 "persona":"internal_agent","lang":"en","version":"poc6-v1","audit_status":"approved",
 "messages":[
   {"role":"system","content":"<Appendix A system prompt>"},
   {"role":"user","content":"<raw customer complaint>"},
   {"role":"assistant","content":"<think>...reasoning from cues...</think>{<resolution-package JSON>}"}]}
```

Loss is computed only on the assistant turn (train-on-responses-only); the `<think>` block stays
literal inside `assistant.content` so non-`<think>`-native models learn it too. Knowledge-augmentation
records (`mode: "knowledge"`) are plain `user`/`assistant` Q&A over runbook fields (no system turn).

## `data_type = "preference"` (DPO)

```json
{"id":"dpo-fabricated_telemetry-00007","data_type":"preference","mode":"tcu_offline",
 "bad_type":"fabricated_telemetry","persona":"internal_agent","lang":"en","version":"poc6-v1",
 "audit_status":"approved",
 "prompt":[{"role":"system","content":"..."},{"role":"user","content":"..."}],
 "chosen":[{"role":"assistant","content":"<think>calibrated...</think>{...}"}],
 "rejected":[{"role":"assistant","content":"<think>webhook delivered at T+28s...</think>{...}"}]}
```

`bad_type` ∈ `cue_dropped, fabricated_telemetry, overconfident, missing_fields, forced_guess,
overpromise`. `chosen` passes the oracle; `rejected` deliberately commits that one error.

## Gold / eval (`data/gold/gold_test.jsonl`)

```json
{"id":"eval-00001","mode":"tcu_offline","leading_root_cause":"TCU_OFFLINE",
 "messages":[{"role":"user","content":"<complaint>"}],
 "reference":"<gold assistant package>","persona":"internal_agent"}
```

`leading_root_cause` is the ground-truth label the evaluator scores against (RC accuracy + confusion).
`eval_hard.jsonl` has the same shape with an empty `reference` (the label is the ground truth).
