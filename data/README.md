# Data contract (consumed, not created)

This directory is **gitignored** and populated by the **data team**. This repo only *reads and
validates* the data — it never creates it (see `docs/SPEC.md` §2–§3). Place files as:

```
data/
├── sft/*.jsonl          # supervised fine-tuning (single-turn or multi-turn)
├── reasoning/*.jsonl    # chain-of-thought records
├── preference/*.jsonl   # preference pairs for DPO/ORPO
└── gold/gold_test.jsonl # gold test set for evaluation (each case labeled with `mode`)
```

All files are **JSON Lines** (one record per line). Validate a delivery with:

```bash
uv run python scripts/validate_data.py --data-dir data/
```

## Common metadata (every record)

`id, data_type, mode, persona, source, lang, version, audit_status`

- `mode` ∈ the 7 conversation modes (metadata only — it must **never** enter the training
  sequence): `purchase_intent, comparison, objection_handling, upsell, after_sales,
  complex_query, edge_case`.
- Only records with `audit_status == "approved"` are used for training (configurable via
  `data.keep_audit_status` in `configs/base.yaml`).

## `data_type = "sft"` (single-turn or multi-turn)

```json
{"id":"...","data_type":"sft","conversation_type":"single|multi_turn","mode":"...","persona":"P0x",
 "messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}],
 "lang":"vi","version":"v1","audit_status":"approved"}
```

Loss is computed on **every** assistant turn (train-on-responses-only + multi-turn masking).

## `data_type = "reasoning"`

```json
{"id":"...","data_type":"reasoning","mode":"...","persona":"P0x",
 "situation":"...","reasoning":["step1","step2"],"response":"...","why":"<audit-only, NOT trained>",
 "lang":"vi","version":"v1","audit_status":"approved"}
```

At training time the reasoning is folded into the assistant turn as
`<think>\n{reasoning}\n</think>\n{response}` (toggleable). The `why` field is **audit-only and
never trained**.

## `data_type = "preference"` (DPO/ORPO)

```json
{"id":"...","data_type":"preference","mode":"...","persona":"P0x","bad_type":"pushy",
 "prompt":[{"role":"user","content":"..."}],
 "chosen":[{"role":"assistant","content":"..."}],
 "rejected":[{"role":"assistant","content":"..."}],
 "lang":"vi","version":"v1","audit_status":"approved"}
```

## Gold test set (`data/gold/gold_test.jsonl`)

Used by `scripts/evaluate.py`. Canonical (pinned) shape — one object per line, each labeled with
`mode`. `messages` is the prompt context (no assistant turn); `reference` is the senior/gold
answer the judges compare against:

```json
{"id":"...","mode":"...","messages":[{"role":"user","content":"..."}],
 "reference":"<câu trả lời chuẩn của sales senior>","persona":"P0x"}
```

For convenience the loader also accepts a `prompt`/`question`/`situation` field instead of
`messages`, and a trailing assistant turn or a `response`/`answer` field as the `reference`
(see `slm_coach.data.schema.GoldCase`). Validate the whole set the same way as the rest:
`uv run python scripts/validate_data.py` validates the training splits; gold is validated when
you run evaluation.
