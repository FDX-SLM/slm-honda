# Honda Entitlement Resolver — evaluation report

- **Model:** checkpoints/sft_granite/best
- **Cases:** 180

## KPIs

| Metric | Value | Target |
| --- | ---: | :---: |
| RC accuracy (all) | 0.872 |  |
| RC accuracy (clear-cue) | 0.860 | ✅ |
| Cue-grounding faithfulness | 0.993 | ✅ |
| No-fabricated-telemetry rate | 1.000 | ✅ |
| Runbook completeness | 1.000 | ✅ |
| Runbook fidelity | 1.000 |  |
| Artifact valid@1 | 1.000 | ✅ |
| Abstention hallucination | 0.067 | ✅ |
| Calibration ECE | 0.201 |  |
| Overconfident-wrong rate | 0.000 |  |
| Parse-fail rate | 0.000 |  |

## Confusion matrix (3 RC + ABSTAIN)

| gold ＼ pred | TCU OFFLIN | ENTITLEMEN | ELIGIBILIT | PAYMENT WE | TOKEN SCOP | INSUFFICIE | PARSE FAIL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TCU OFFLINE | 29 | 1 | 0 | 0 | 0 | 0 | 0 |
| ENTITLEMENT CA | 0 | 30 | 0 | 0 | 0 | 0 | 0 |
| ELIGIBILITY RU | 0 | 1 | 11 | 18 | 0 | 0 | 0 |
| PAYMENT WEBHOO | 0 | 1 | 0 | 29 | 0 | 0 | 0 |
| TOKEN SCOPE | 0 | 0 | 0 | 0 | 30 | 0 | 0 |
| INSUFFICIENT E | 2 | 0 | 0 | 0 | 0 | 28 | 0 |

## Per-slice accuracy

| Slice | Accuracy |
| --- | ---: |
| abstention | 0.933 |
| cache_stale | 1.000 |
| eligibility | 0.367 |
| payment_webhook | 0.967 |
| tcu_offline | 0.967 |
| token_scope | 1.000 |