# Honda Entitlement Resolver — evaluation report

- **Model:** checkpoints/sft_gemma/best
- **Cases:** 180

## KPIs

| Metric | Value | Target |
| --- | ---: | :---: |
| RC accuracy (all) | 0.889 |  |
| RC accuracy (clear-cue) | 0.900 | ✅ |
| Cue-grounding faithfulness | 0.998 | ✅ |
| No-fabricated-telemetry rate | 1.000 | ✅ |
| Runbook completeness | 1.000 | ✅ |
| Runbook fidelity | 0.987 |  |
| Artifact valid@1 | 1.000 | ✅ |
| Abstention hallucination | 0.167 | ❌ |
| Calibration ECE | 0.219 |  |
| Overconfident-wrong rate | 0.000 |  |
| Parse-fail rate | 0.000 |  |

## Confusion matrix (3 RC + ABSTAIN)

| gold ＼ pred | TCU OFFLIN | ENTITLEMEN | ELIGIBILIT | PAYMENT WE | TOKEN SCOP | INSUFFICIE | PARSE FAIL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TCU OFFLINE | 30 | 0 | 0 | 0 | 0 | 0 | 0 |
| ENTITLEMENT CA | 1 | 29 | 0 | 0 | 0 | 0 | 0 |
| ELIGIBILITY RU | 0 | 2 | 16 | 12 | 0 | 0 | 0 |
| PAYMENT WEBHOO | 0 | 0 | 0 | 30 | 0 | 0 | 0 |
| TOKEN SCOPE | 0 | 0 | 0 | 0 | 30 | 0 | 0 |
| INSUFFICIENT E | 1 | 0 | 0 | 0 | 0 | 25 | 0 |

## Per-slice accuracy

| Slice | Accuracy |
| --- | ---: |
| abstention | 0.833 |
| cache_stale | 0.967 |
| eligibility | 0.533 |
| payment_webhook | 1.000 |
| tcu_offline | 1.000 |
| token_scope | 1.000 |