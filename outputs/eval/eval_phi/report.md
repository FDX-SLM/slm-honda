# Honda Entitlement Resolver — evaluation report

- **Model:** checkpoints/sft_phi/best
- **Cases:** 180

## KPIs

| Metric | Value | Target |
| --- | ---: | :---: |
| RC accuracy (all) | 0.806 |  |
| RC accuracy (clear-cue) | 0.820 | ❌ |
| Cue-grounding faithfulness | 0.981 | ✅ |
| No-fabricated-telemetry rate | 1.000 | ✅ |
| Runbook completeness | 1.000 | ✅ |
| Runbook fidelity | 0.994 |  |
| Artifact valid@1 | 1.000 | ✅ |
| Abstention hallucination | 0.267 | ❌ |
| Calibration ECE | 0.136 |  |
| Overconfident-wrong rate | 0.000 |  |
| Parse-fail rate | 0.000 |  |

## Confusion matrix (3 RC + ABSTAIN)

| gold ＼ pred | TCU OFFLIN | ENTITLEMEN | ELIGIBILIT | PAYMENT WE | TOKEN SCOP | INSUFFICIE | PARSE FAIL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TCU OFFLINE | 29 | 0 | 1 | 0 | 0 | 0 | 0 |
| ENTITLEMENT CA | 3 | 27 | 0 | 0 | 0 | 0 | 0 |
| ELIGIBILITY RU | 1 | 6 | 7 | 16 | 0 | 0 | 0 |
| PAYMENT WEBHOO | 0 | 0 | 0 | 30 | 0 | 0 | 0 |
| TOKEN SCOPE | 0 | 0 | 0 | 0 | 30 | 0 | 0 |
| INSUFFICIENT E | 3 | 2 | 0 | 1 | 0 | 22 | 0 |

## Per-slice accuracy

| Slice | Accuracy |
| --- | ---: |
| abstention | 0.733 |
| cache_stale | 0.900 |
| eligibility | 0.233 |
| payment_webhook | 1.000 |
| tcu_offline | 0.967 |
| token_scope | 1.000 |