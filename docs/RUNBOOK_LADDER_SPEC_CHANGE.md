# Spec change request — 5 root-cause classes × sub-causes, + customer self-service ladder

> For **claude web** to fold into `PoC6_SLM_BUILD_SPEC`. This is the design source of truth for an
> additive expansion along **two axes** (breadth: 3→5 RC classes; depth: each class → multiple
> sub-causes). It removes nothing; it enriches §1, §2, §3.2, §4, §7.
> Training data is **hand-distilled by Claude** (no template generator, no API); the rule-based
> generator is kept only as a backup and must stay consistent with this schema.

---

## 0. Why
Eval showed two gaps: (1) the diagnosis can hinge on a single cue, and (2) with only 3 RCs and one
runbook each, 1000 records repeat the same handful of answers/solutions. Fix with two axes:

- **Breadth (3 → 5 RC classes):** one class per failure point on the dependency chain, each with a
  distinct *language cue* so the model still differentiates from the complaint alone (the "wow").
- **Depth (each class → 4–7 sub-causes):** a real root-cause class manifests many ways; each sub-cause
  has its own *sub-cue*, *internal fix*, *severity/ETA*, and *tailored escalation* — so reasoning is a
  two-tier differential (class → sub-cause) and solutions are genuinely varied.

Plus the customer-facing **self-service ladder** (deep, step-by-step, easy→hard) from the previous change.

---

## 1. The 5 RC classes (one per pipeline node)
`SSP → HONDA_PAY → ENTITLEMENT_SVC → IAM_HIDAS → CCS_PORTAL → TCU_VEHICLE`

| Node | RC class | Distinct top-level cue | Owner | New? |
|---|---|---|---|---|
| HONDA_PAY | `PAYMENT_WEBHOOK_LOST` | paid **very recently**; "just bought it / this morning"; often self-resolves | DSD Payments Team | 🆕 |
| ENTITLEMENT_SVC | `ELIGIBILITY_RULE_CONFLICT` | keeps prompting Subscribe; region/trim/plan combo; **never created, persistent** | Entitlement Platform Team | ✓ |
| IAM_HIDAS | `TOKEN_SCOPE` | **error 403 / permission denied / logged out when opening the feature** (login otherwise fine) | HG Identity Team | 🆕 (was abstain) |
| CCS_PORTAL | `ENTITLEMENT_CACHE_STALE` | **active on web ≠ app**; intermittent; re-login helps | Entitlement Platform Team | ✓ |
| TCU_VEHICLE | `TCU_OFFLINE` | car in garage/underground; **car-side timeout**; app shows active | HG Connected Vehicle Team | ✓ |

`INSUFFICIENT_EVIDENCE` (abstain) stays for genuinely cue-less / out-of-catalog tickets.

The five top-level cues do **not** overlap, so class-level diagnosis from the complaint remains clean.

---

## 2. Sub-cause taxonomy (depth) — `sub_cause` per class
`leading_root_cause` stays the **class** (for cue-classification + eval). Add `sub_cause` (the specific
failure mode). owner_team / support_contact / escalation / runbook_id stay **per class** (fidelity-locked);
**why_technical, fix_steps, severity, eta_ttr, and the ladder's escalation rung vary by sub-cause.**

### PAYMENT_WEBHOOK_LOST (RB-PAY-01)
| sub_cause | sub-cue | internal fix | sev / ETA |
|---|---|---|---|
| `webhook_delayed` | "bought minutes ago", brand-new | wait; auto-retry usually settles | S4 / minutes (self-resolves) |
| `webhook_dropped` | "bought this morning, hours later still nothing" | replay payment→entitlement event | S3 / ~1h |
| `payment_pending_review` | "card shows pending / processing / on hold" | wait for settlement or manually release | S3 / hours |
| `duplicate_charge_no_grant` | "charged twice but nothing activated" | dedupe charge, grant entitlement, refund dup | S3 / ~1h |
| `partial_provision` | "shows in my orders but the feature is off" | re-run the entitlement-creation step | S3 / ~1h |

### ELIGIBILITY_RULE_CONFLICT (RB-ELIG-05)
| sub_cause | sub-cue | internal fix | sev / ETA |
|---|---|---|---|
| `region_not_in_matrix` | names a region ("Canada / US-West") | add region to eligibility matrix | S3 / 2-3h |
| `trim_not_in_matrix` | premium trim ("CR-V Touring") | add trim to matrix | S3 / 2-3h |
| `plan_tier_not_enabled` | "elite / top tier" for that combo | enable plan tier for the combo | S3 / 2-3h |
| `matrix_stale_new_model_year` | "2026 / just-released model" | update matrix for the new model year | S2 / ~1 day |
| `rule_misconfig_bug` | "this combo should be allowed" | fix rule logic (code deploy) | S2 / 1 business day |
| `promo_bundle_edge` | "got it through a promo/bundle" | manual entitlement grant | S3 / 2-3h |

### TOKEN_SCOPE (RB-IAM-03)
| sub_cause | sub-cue | internal fix | sev / ETA |
|---|---|---|---|
| `stale_scope` | "403 / permission denied" on the feature, login fine | force a token/scope refresh | S3 / <1h |
| `token_expired` | "keeps logging me out when I tap it" | re-auth / clear the session | S3 / <1h |
| `scope_mapping_bug` | "403 even after re-login" | fix entitlement→scope mapping, reissue | S2 / hours |
| `multi_account_mismatch` | "two accounts / wrong email signed in" | sign into / link the correct account | S3 / <1h |

### ENTITLEMENT_CACHE_STALE (RB-CACHE-02)
| sub_cause | sub-cue | internal fix | sev / ETA |
|---|---|---|---|
| `app_client_cache` | "reinstall / re-login fixes it" | client cache clear (customer self-serve) | S3 / minutes |
| `ccs_server_cache_ttl` | "web active, app not, persists after re-login" | force server cache invalidation | S3 / <30m |
| `cdn_edge_cache` | "fine on some devices/areas, stale on others" | purge edge cache | S3 / <30m |
| `multi_device_sync_lag` | "works on my tablet but not my phone" | refresh the lagging device | S4 / minutes |
| `invalidation_missed` | "changed/upgraded plan, app never updated" | manual invalidation after the change | S3 / <30m |

### TCU_OFFLINE (RB-TCU-04)
| sub_cause | sub-cue | internal fix | sev / ETA |
|---|---|---|---|
| `no_signal_garage` | "underground / garage / basement" | move car to open sky | S4 / <15m |
| `weak_signal_remote` | "rural / weak coverage area" | relocate to better coverage | S4 / <15m |
| `tcu_asleep` | "not driven all week / sat idle" | drive or start engine to wake the unit | S4 / <15m |
| `tcu_firmware_hang` | "outside with good signal, still times out" | ignition/power cycle | S3 / ~30m |
| `tcu_hardware_fault` | "tried everything for days, still dead" | RMA path | S3 / days |
| `low_12v_battery` | "battery weak / car parked for months" | charge the 12V battery | S4 / hours |
| `carrier_outage` | "whole area's cell network is down" | wait / check carrier status | S3 / varies |

(≈ 27 sub-causes total + abstain → ample diversity for 700–1000 records.)

---

## 3. Resolution package change (§3.2)
Add `sub_cause` (string, one of the class's sub-causes) and keep `customer_self_service` (deep ladder):

```json
"diagnosis": {
  "leading_root_cause": "TCU_OFFLINE",          // class (cue-classification + eval)
  "sub_cause": "tcu_hardware_fault",            // NEW: the specific failure mode
  "confidence": 0.7,
  "differential": [ ... ],
  "evidence_in_ticket": [ ... ],
  "to_confirm": [ ... ]
},
"...runbook fields (severity/eta/fix_steps vary by sub_cause; owner/support/escalation per class)...",
"customer_self_service": [ ...deep ordered ladder, escalation rung tailored to the sub_cause... ]
```

---

## 4. Two-tier `<think>` reasoning
The trace now does a **class diagnosis then a sub-cause sub-differential**, e.g.:
> "Cue: app shows active but the car times out → class is TCU_OFFLINE. Going deeper: they say the car
> has been **outside with good signal for days and it still times out** — that rules out environmental
> signal loss and points to a **firmware hang or hardware fault** rather than just 'move the car'. I
> lean hardware fault, raise severity, and route to RMA instead of the open-sky step."

This is where richness comes from: not 1-of-5 boxes, but *which box, then which failure inside it*.

---

## 5. Customer self-service ladder (unchanged shape, sub-cause-aware tail)
Deep, ordered, easy→hard, 5–7 rungs: early rungs **rule out** other classes (self-triage), middle rungs
**resolve** the leading class, last rung **escalates**. The **escalation rung is tailored to the
sub-cause** (e.g., hardware_fault → "contact support for a hardware check / RMA"; webhook_dropped →
"no need to repay; we'll replay your activation"). Customer-safe actions only; no internal ops; no
fabricated telemetry; no over-promise.

---

## 6. Oracle / eval (§4 / §7)
- §1.2/§1.3: add cue signatures for `PAYMENT_WEBHOOK_LOST` (recency) and `TOKEN_SCOPE` (403/permission).
  Note: 403 moves **from** abstention **into** TOKEN_SCOPE.
- `ROOT_CAUSES` / `ALL_LABELS`: now 5 classes.
- §4 rules unchanged in spirit; add: `sub_cause` must be a known sub-cause of the asserted class;
  fidelity still locks owner/support/escalation per class (severity/eta/fix_steps may vary by sub-cause);
  ladder fabrication + over-promise + ≥3 ordered rungs still enforced.
- §7 KPIs: keep RC-class accuracy + confusion; add **sub-cause accuracy** (within the correct class) and
  `self_service_present_rate`. Calibration/abstention/no-fabrication unchanged.

---

## 7. Code changes (in repo, after this spec is confirmed)
| File | Change |
|---|---|
| `ground_truth.py` | 2 new base runbooks (RB-PAY-01, RB-IAM-03); `SUB_CAUSES` taxonomy with per-sub-cause overrides; cue lib + ladder for new classes; `customer_self_service_ladder(rc, sub_cause)` |
| `oracle.py` | 5-class `ROOT_CAUSES`/`ALL_LABELS`; cue signatures for the 2 new classes; `sub_cause` validation |
| `datagen/core.py` | `build_resolution(rc, sub_cause, ...)` applies overrides + emits `sub_cause`; new-class openers/cues/differential-alt so the backup generator stays consistent |
| `eval/honda.py` | sub-cause accuracy KPI; `sub_cause` in per_sample.csv |
| (data) | `data/sft/distilled_claude.jsonl` — Claude-hand-distilled, balanced across 5 classes × sub-causes, deep ladders; target 700–1000 |

---

## 8. Important
All class names, sub-causes, owner teams, severities, and compensation are **[PoC] synthetic** — the
model learns to be internally consistent with them; FA/USK must validate against the real Honda
architecture before production (PM2.0). Two existing behaviors must be preserved after training:
**(1) correct error detection** (class + sub-cause) and **(2) strong internal support** (full runbook
package) — alongside the new **(3) deep, easy→hard customer guidance**.
