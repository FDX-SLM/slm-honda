r"""Ground truth for the Honda Entitlement Resolver PoC (PoC6 BUILD SPEC §1–§2).

Đây là **nguồn chân lý duy nhất** (single source of truth) cho toàn bộ pipeline:
generator (``slm_coach.datagen``) trích field từ đây để sinh data, và oracle
(``slm_coach.oracle``) chấm mọi mẫu sinh ra so với đây. Mọi "sự thật" (systems, root
cause, runbook, incident, eligibility matrix) nằm trong file này — code không bịa fact, chỉ
sinh biến thể diễn đạt từ ground truth.

Mọi nội dung train/runbook viết **tiếng Anh** (theo spec §0). Comment/giải thích tiếng Việt.

Bố cục:
* :data:`SYSTEMS` — chuỗi phụ thuộc 6 node (§1.1).
* :data:`SEVERITY`, :data:`PRIORITY`, :data:`CHURN_RISK_LEVELS` — thang dùng chung (§2.2).
* :data:`ROOT_CAUSES` / :data:`ABSTAIN` — enum 3 RC + abstention.
* :data:`CUE_LIBRARY` — manh mối ngôn ngữ (customer cues) cho từng RC (§1.2) + paraphrase pool.
* :data:`ELIGIBILITY_MATRIX` — ma trận eligibility cho RC-5 (§1.4).
* :data:`INCIDENTS` — incident seeds, ≥4 mỗi RC (§1.5).
* :data:`RUNBOOKS` — structured dict đầy đủ field §2.1 cho 3 RC, kèm :func:`render_runbook`
  sinh bản document markdown chuyên nghiệp (§2.3).
* :data:`SYSTEM_PROMPT` — system prompt Phụ lục A.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# §1.1 Dependency chain — SYSTEMS
# ---------------------------------------------------------------------------

#: 6-node dependency chain (synthetic / PoC names — see Appendix D of the spec).
#: SSP → HONDA_PAY → ENTITLEMENT_SVC → IAM_HIDAS → CCS_PORTAL → TCU_VEHICLE
SYSTEMS: dict[str, dict[str, str]] = {
    "SSP": {
        "full_name": "Subscription Service Platform",
        "owner_team": "DSD Product Team (AHM)",
        "role": "Manages subscription plans (Sport / Elite / Touring)",
    },
    "HONDA_PAY": {
        "full_name": "Honda Pay / Stripe",
        "owner_team": "DSD Payments Team",
        "role": "Collects payment and emits the payment.succeeded webhook",
    },
    "ENTITLEMENT_SVC": {
        "full_name": "Entitlement Service (+cache, eligibility)",
        "owner_team": "Entitlement Platform Team",
        "role": "Creates entitlements, runs eligibility checks, owns the cache",
    },
    "IAM_HIDAS": {
        "full_name": "IAM / HIDAS",
        "owner_team": "HG Identity Team",
        "role": "Issues tokens/scopes derived from the entitlement",
    },
    "CCS_PORTAL": {
        "full_name": "CCS Portal (HondaLink app/web)",
        "owner_team": "DSD Digital Experience Team",
        "role": "Customer-facing UI; reads the entitlement through the cache",
    },
    "TCU_VEHICLE": {
        "full_name": "TCU / Vehicle",
        "owner_team": "HG Connected Vehicle Team",
        "role": "In-vehicle unit receiving push over cellular",
    },
}

# ---------------------------------------------------------------------------
# §2.2 Shared scales
# ---------------------------------------------------------------------------

SEVERITY: dict[str, str] = {
    "S1": "Widespread (many customers / a whole region)",
    "S2": "One combo / a group of customers",
    "S3": "One customer, feature unusable, no workaround",
    "S4": "One customer, has a workaround / environmental / self-resolving",
}

PRIORITY: dict[str, str] = {
    "P1": "Immediate",
    "P2": "Same day (paid customer, no workaround)",
    "P3": "Within 1-2 days",
    "P4": "Backlog",
}

CHURN_RISK_LEVELS: tuple[str, ...] = ("low", "low-medium", "medium", "medium-high", "high")

# ---------------------------------------------------------------------------
# Root-cause classes + abstention
# ---------------------------------------------------------------------------

#: The three in-catalog root-cause classes the model may conclude.
ROOT_CAUSES: tuple[str, ...] = (
    "TCU_OFFLINE",
    "ENTITLEMENT_CACHE_STALE",
    "ELIGIBILITY_RULE_CONFLICT",
)

#: Sentinel for abstention (ambiguous complaint or out-of-catalog).
ABSTAIN: str = "INSUFFICIENT_EVIDENCE"

#: Every legal value of ``diagnosis.leading_root_cause``.
ALL_LABELS: tuple[str, ...] = (*ROOT_CAUSES, ABSTAIN)

#: RC class → runbook id.
RC_TO_RUNBOOK: dict[str, str] = {
    "ENTITLEMENT_CACHE_STALE": "RB-CACHE-02",
    "TCU_OFFLINE": "RB-TCU-04",
    "ELIGIBILITY_RULE_CONFLICT": "RB-ELIG-05",
}

#: Short, stable group tag used for per-slice eval/holdout (mirrors schema ``Mode``).
RC_TO_SLICE: dict[str, str] = {
    "TCU_OFFLINE": "tcu_offline",
    "ENTITLEMENT_CACHE_STALE": "cache_stale",
    "ELIGIBILITY_RULE_CONFLICT": "eligibility",
    ABSTAIN: "abstention",
}

# ---------------------------------------------------------------------------
# §1.2 Cue library — the linguistic clues that let the model differentiate
# ---------------------------------------------------------------------------

#: Customer cues per RC. ``cues`` are the canonical clue phrases (used both to synthesize
#: complaints and to ground ``evidence_in_ticket``); ``distinguishing_from`` is the contrast
#: rule the <think> trace should reflect.
CUE_LIBRARY: dict[str, dict[str, Any]] = {
    "ENTITLEMENT_CACHE_STALE": {
        "one_line": (
            "Entitlement was created but the app/CCS cache has not refreshed, so the app shows "
            "the feature as missing."
        ),
        "cues": [
            "the web/account shows it as ACTIVE but the app does not",
            "it worked before and then suddenly stopped",
            "it is intermittent / flickers",
            "logging out and back in sometimes helps",
            "I can clearly see it is active but it won't turn on",
        ],
        "distinguishing_from": (
            "ELIGIBILITY_RULE_CONFLICT: cache stale means the entitlement DOES exist (active "
            "somewhere); eligibility means it is active NOWHERE and the app keeps asking to buy."
        ),
    },
    "TCU_OFFLINE": {
        "one_line": (
            "Subscription is active but the TCU lost cellular signal, so the vehicle never "
            "received the push."
        ),
        "cues": [
            "the car has been parked in a basement/underground garage for days",
            "the car has no signal",
            "remote commands spin and then time out",
            "the car has not been driven all week",
            "the feature shows active but the car does not respond",
        ],
        "distinguishing_from": (
            "ENTITLEMENT_CACHE_STALE: cache stale means the app itself shows the feature missing; "
            "TCU offline means the app shows it as active but the CAR times out (car-side)."
        ),
    },
    "ELIGIBILITY_RULE_CONFLICT": {
        "one_line": (
            "The webhook arrived but eligibility rejected the region/trim/plan combo, so no "
            "entitlement was created."
        ),
        "cues": [
            "the app keeps prompting Subscribe / asking me to buy again even though I paid",
            "a region- or trim-limited or premium combination (new trim, limited region, top plan)",
            "it happens immediately and persistently, not intermittently",
            "I already paid, why does it still ask me to buy",
        ],
        "distinguishing_from": (
            "ENTITLEMENT_CACHE_STALE: cache stale means the entitlement DOES exist (active "
            "somewhere); eligibility means it is active nowhere. Persistent for days, not "
            "intermittent, does not self-resolve."
        ),
    },
}

#: Out-of-catalog cues → abstention (NOT a concrete RC). Token 403, billing, app crash, OTA.
OUT_OF_CATALOG_CUES: list[str] = [
    "I keep getting a 403 error",
    "the app logs me out constantly",
    "it says access permission denied",
    "I want a refund / I am disputing the charge",
    "the app crashes on launch",
    "there is a software/OTA update stuck",
]

#: Truly ambiguous / vague complaints (no distinguishing cue) → abstention.
VAGUE_COMPLAINTS: list[str] = [
    "It just doesn't work.",
    "I paid for it and nothing happens.",
    "My subscription is broken, please fix it.",
    "The feature I bought is not available, I don't know why.",
    "Something is wrong with my account.",
]

# ---------------------------------------------------------------------------
# Paraphrase pools — natural-language scaffolding to synthesize raw complaints
# ---------------------------------------------------------------------------

#: Opening lines a customer might use (with a {feature} / {days} slot).
COMPLAINT_OPENERS: list[str] = [
    "I bought {feature} about {days} and it still won't work.",
    "I paid for {feature} {days} ago but I can't use it.",
    "I subscribed to {feature} and it's not working.",
    "I got {feature} on my plan recently and something is off.",
    "Hi, I purchased {feature} but it isn't doing anything.",
]

#: Subscription features the customer talks about (surface detail, not a cue).
FEATURES: list[str] = [
    "Remote Start",
    "Remote Climate",
    "the Touring package",
    "Remote Lock/Unlock",
    "the connected services subscription",
    "Vehicle Finder",
]

#: Time phrases for the {days} slot.
TIME_PHRASES: list[str] = [
    "3 days ago",
    "a couple of days ago",
    "yesterday",
    "last week",
    "about a week ago",
    "a few days ago",
]

# ---------------------------------------------------------------------------
# §1.4 Eligibility matrix (for RC-5)
# ---------------------------------------------------------------------------

ELIGIBILITY_MATRIX: dict[str, Any] = {
    "eligible_combos": [
        {"region": "US-East", "vehicle": "Civic 2025", "plans": ["Sport", "Elite", "Touring"]},
        {"region": "US-East", "vehicle": "CR-V 2025", "plans": ["Sport", "Elite", "Touring"]},
        {"region": "US-West", "vehicle": "Civic 2025", "plans": ["Sport", "Elite", "Touring"]},
        {"region": "US-West", "vehicle": "CR-V 2025", "plans": ["Sport", "Elite"]},
        {"region": "Canada", "vehicle": "Civic 2025", "plans": ["Sport", "Elite"]},
    ],
    "example_trap": {
        "region": "US-West",
        "vehicle": "CR-V 2025",
        "plan": "Touring",
        "note": "Touring is outside the combo -> RC-5",
    },
}


def is_eligible(region: str, vehicle: str, plan: str) -> bool:
    """Return whether a region/vehicle/plan combo is in the eligibility matrix (§1.4)."""
    for combo in ELIGIBILITY_MATRIX["eligible_combos"]:
        if combo["region"] == region and combo["vehicle"] == vehicle and plan in combo["plans"]:
            return True
    return False


#: Trap combos (paid but eligibility rejects) — concrete grounding for RC-5 complaints.
ELIGIBILITY_TRAPS: list[dict[str, str]] = [
    {"region": "US-West", "vehicle": "CR-V 2025", "plan": "Touring"},
    {"region": "Canada", "vehicle": "Civic 2025", "plan": "Touring"},
    {"region": "Canada", "vehicle": "CR-V 2025", "plan": "Elite"},
]

# ---------------------------------------------------------------------------
# §1.5 Incident seeds (≥4 per RC)
# ---------------------------------------------------------------------------

#: Past incidents the model can cite as ``similar_incident`` + use for TTR statistics.
INCIDENTS: list[dict[str, Any]] = [
    # --- TCU_OFFLINE ---
    {
        "id": "INC-0742",
        "root_cause_class": "TCU_OFFLINE",
        "customer_complaint": (
            "Subscription active but remote start times out; car parked in basement garage "
            "for a week"
        ),
        "key_cue": "underground garage + remote timeout",
        "resolution_steps": [
            "check TCU last_seen",
            "ask customer to move vehicle to open sky",
            "re-push entitlement",
            "verify ack",
        ],
        "owner": "HG Connected Vehicle Team",
        "ttr_min": 20,
        "severity": "S4",
        "tags": ["tcu", "offline", "cellular"],
    },
    {
        "id": "INC-0758",
        "root_cause_class": "TCU_OFFLINE",
        "customer_complaint": "Remote climate shows active but the car hasn't moved all week and won't respond",
        "key_cue": "car not driven all week + no response",
        "resolution_steps": [
            "check TCU last_seen",
            "ask customer to drive to open area and start engine",
            "re-push entitlement",
            "verify ack",
        ],
        "owner": "HG Connected Vehicle Team",
        "ttr_min": 15,
        "severity": "S4",
        "tags": ["tcu", "offline"],
    },
    {
        "id": "INC-0771",
        "root_cause_class": "TCU_OFFLINE",
        "customer_complaint": "Customer says the car has no signal in their underground parking and commands keep timing out",
        "key_cue": "no signal underground + timeout",
        "resolution_steps": [
            "confirm entitlement active",
            "check TCU last_seen",
            "move vehicle to open sky",
            "re-push",
        ],
        "owner": "HG Connected Vehicle Team",
        "ttr_min": 25,
        "severity": "S4",
        "tags": ["tcu", "offline", "cellular"],
    },
    {
        "id": "INC-0789",
        "root_cause_class": "TCU_OFFLINE",
        "customer_complaint": "Remote start spins and times out; TCU hardware fault suspected after vehicle came back online but still failed",
        "key_cue": "timeout persists after reconnect -> hardware",
        "resolution_steps": [
            "check TCU last_seen",
            "verify token valid",
            "escalate HG L2 for hardware check",
            "RMA if confirmed",
        ],
        "owner": "HG Connected Vehicle Team",
        "ttr_min": 2880,
        "severity": "S3",
        "tags": ["tcu", "hardware", "rma"],
    },
    # --- ENTITLEMENT_CACHE_STALE ---
    {
        "id": "INC-0610",
        "root_cause_class": "ENTITLEMENT_CACHE_STALE",
        "customer_complaint": "Web account shows the feature active but the app does not; logging out and back in helped briefly",
        "key_cue": "active on web not app + relogin helps",
        "resolution_steps": [
            "confirm entitlement active for VIN",
            "force cache invalidation (DEL ent:{vin})",
            "re-sync push to CCS",
            "ask customer to pull-to-refresh",
        ],
        "owner": "Entitlement Platform Team",
        "ttr_min": 12,
        "severity": "S3",
        "tags": ["cache", "stale", "ccs"],
    },
    {
        "id": "INC-0623",
        "root_cause_class": "ENTITLEMENT_CACHE_STALE",
        "customer_complaint": "Feature worked yesterday then disappeared in the app; intermittent",
        "key_cue": "worked before then stopped + intermittent",
        "resolution_steps": [
            "confirm entitlement active",
            "force cache invalidation",
            "re-sync",
            "verify app",
        ],
        "owner": "Entitlement Platform Team",
        "ttr_min": 10,
        "severity": "S3",
        "tags": ["cache", "stale"],
    },
    {
        "id": "INC-0641",
        "root_cause_class": "ENTITLEMENT_CACHE_STALE",
        "customer_complaint": "Customer can see it active on the website but the app keeps showing it as off",
        "key_cue": "active on web, off in app",
        "resolution_steps": [
            "confirm entitlement active",
            "check cache TTL / last invalidation",
            "force invalidation",
            "re-sync push",
        ],
        "owner": "Entitlement Platform Team",
        "ttr_min": 15,
        "severity": "S3",
        "tags": ["cache", "stale", "ttl"],
    },
    {
        "id": "INC-0659",
        "root_cause_class": "ENTITLEMENT_CACHE_STALE",
        "customer_complaint": "Feature flickers on and off in the app even though account looks fine",
        "key_cue": "flickering / inconsistent active state",
        "resolution_steps": [
            "confirm entitlement active",
            "force cache invalidation",
            "re-sync push to CCS",
            "verify",
        ],
        "owner": "Entitlement Platform Team",
        "ttr_min": 12,
        "severity": "S3",
        "tags": ["cache", "stale"],
    },
    # --- ELIGIBILITY_RULE_CONFLICT ---
    {
        "id": "INC-0501",
        "root_cause_class": "ELIGIBILITY_RULE_CONFLICT",
        "customer_complaint": "Paid for Touring on a CR-V 2025 in US-West but the app keeps prompting Subscribe",
        "key_cue": "paid + app keeps asking to subscribe + trap combo",
        "resolution_steps": [
            "fetch eligibility decision log",
            "confirm wrongful reject vs matrix",
            "update eligibility matrix",
            "replay entitlement creation",
            "audit same-combo customers",
        ],
        "owner": "Entitlement Platform Team",
        "ttr_min": 140,
        "severity": "S3",
        "tags": ["eligibility", "matrix", "replay"],
    },
    {
        "id": "INC-0517",
        "root_cause_class": "ELIGIBILITY_RULE_CONFLICT",
        "customer_complaint": "Customer with a Canada Civic Touring plan paid but never got the feature, app asks to buy again",
        "key_cue": "trap combo + persistent subscribe prompt",
        "resolution_steps": [
            "fetch eligibility decision log",
            "confirm wrongful reject",
            "update matrix",
            "replay entitlement",
        ],
        "owner": "Entitlement Platform Team",
        "ttr_min": 160,
        "severity": "S2",
        "tags": ["eligibility", "region-wide"],
    },
    {
        "id": "INC-0528",
        "root_cause_class": "ELIGIBILITY_RULE_CONFLICT",
        "customer_complaint": "Premium plan on a limited-region trim charged the customer but no entitlement was created",
        "key_cue": "premium/limited combo + paid + no entitlement",
        "resolution_steps": [
            "fetch eligibility decision log",
            "compare against matrix",
            "update matrix",
            "replay + audit",
        ],
        "owner": "Entitlement Platform Team",
        "ttr_min": 150,
        "severity": "S3",
        "tags": ["eligibility", "matrix"],
    },
    {
        "id": "INC-0540",
        "root_cause_class": "ELIGIBILITY_RULE_CONFLICT",
        "customer_complaint": "Region-wide eligibility rule wrongly rejected a whole trim/region group after a plan launch",
        "key_cue": "many customers same combo + immediate persistent",
        "resolution_steps": [
            "fetch decision logs",
            "confirm matrix gap",
            "deploy matrix fix",
            "replay batch + audit",
        ],
        "owner": "Entitlement Platform Team",
        "ttr_min": 480,
        "severity": "S2",
        "tags": ["eligibility", "region-wide", "deploy"],
    },
]


def incidents_for(rc_class: str) -> list[dict[str, Any]]:
    """Return all incident seeds for a root-cause class."""
    return [inc for inc in INCIDENTS if inc["root_cause_class"] == rc_class]


def median_ttr_min(rc_class: str) -> int:
    """Median ``ttr_min`` across the incident seeds of an RC (for TTR statistics)."""
    values = sorted(inc["ttr_min"] for inc in incidents_for(rc_class))
    if not values:
        return 0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) // 2


# ---------------------------------------------------------------------------
# §2.1 / §2.4 — RUNBOOKS (structured dict, full business fields)
# ---------------------------------------------------------------------------

#: One structured runbook per RC. ``render_runbook`` turns each into the §2.3 document.
RUNBOOKS: dict[str, dict[str, Any]] = {
    "ENTITLEMENT_CACHE_STALE": {
        "runbook_id": "RB-CACHE-02",
        "rc_class": "ENTITLEMENT_CACHE_STALE",
        "title": "Entitlement Cache Stale",
        "one_line": CUE_LIBRARY["ENTITLEMENT_CACHE_STALE"]["one_line"],
        "summary": (
            "The subscription was activated successfully, but the app is showing a stale cached "
            "view, so the feature looks missing even though it is enabled. This is a display/sync "
            "issue and resolves quickly."
        ),
        "why_plain": (
            "Your subscription is on, but the app is showing an old cached view, so the feature "
            "looks missing. A refresh fixes it."
        ),
        "why_technical": (
            "Entitlement is active and provisioned, but the CCS/app cache holds a stale view "
            "(TTL not expired or invalidation missed), so the client renders the feature as off."
        ),
        "owner_team": "Entitlement Platform Team",
        "support_contact": "DSD Entitlement on-call (Slack #ent-oncall)",
        "escalation": "DSD L3 if invalidation does not take effect",
        "detection_cues": [
            "the feature appears active on the web/account but not in the app",
            "it worked before and then stopped",
            "behavior is intermittent",
            "logging out and back in (or refreshing) sometimes helps",
            "I can see it's active but it won't turn on",
        ],
        "confirm_checks": [
            "An entitlement record exists and is active for the VIN",
            "The CCS cache TTL has not expired, or check the last cache-invalidation timestamp",
        ],
        "fix_steps": [
            "Confirm the entitlement is active for the VIN",
            "Force cache invalidation (DEL ent:{vin})",
            "Re-sync the entitlement push to CCS",
            "Ask the customer to pull-to-refresh or log out and back in",
            "Verify the app now shows the feature",
        ],
        "eta_ttr": "Under 30 minutes (median ~12 min); often immediate after invalidation",
        "severity": "S3",
        "priority": "P2",
        "churn_risk": {
            "level": "low-medium",
            "why": "the fix is fast, but it is annoying if it recurs",
        },
        "compensation_policy": {
            "offer": "small goodwill gesture only if it recurs 3+ times",
            "when_proactive": False,
            "escalate_if": "recurs three or more times",
            "note": "usually none; not offered proactively",
        },
        "customer_communication": (
            "Reassure the customer this is a display-sync issue with a fast fix; guide them through "
            "refresh/re-login; confirm once it has synced. Do not imply the customer did anything "
            "wrong."
        ),
        "similar_incident": "INC-0610",
        "last_reviewed": "2026-06",
    },
    "TCU_OFFLINE": {
        "runbook_id": "RB-TCU-04",
        "rc_class": "TCU_OFFLINE",
        "title": "TCU Offline",
        "one_line": CUE_LIBRARY["TCU_OFFLINE"]["one_line"],
        "summary": (
            "The subscription is active, but the vehicle's telematics unit (TCU) has been offline "
            "— for example, parked underground — so it never received the activation push. It will "
            "sync within minutes once the vehicle is back in cellular range."
        ),
        "why_plain": (
            "Your subscription is active, but the car has been out of cellular range (e.g. in a "
            "garage), so it hasn't received the activation yet. It syncs within minutes once the "
            "car is back in signal."
        ),
        "why_technical": (
            "Entitlement is active and pushed, but the TCU is offline (no cellular/MQTT) so the "
            "push was not delivered/acked; the vehicle is not synced."
        ),
        "owner_team": "HG Connected Vehicle Team",
        "support_contact": "HG Connected Vehicle on-call",
        "escalation": "HG L2 if a TCU hardware fault is suspected",
        "detection_cues": [
            "the car has been parked in a garage/basement/underground for days",
            "no signal",
            "remote commands spin and then time out",
            "the car has not moved all week",
            "the feature shows as active but the car does not respond",
        ],
        "confirm_checks": [
            "The entitlement is active for the VIN",
            "The TCU last_seen timestamp",
            "The access token is valid",
        ],
        "fix_steps": [
            "Confirm the entitlement is active",
            "Check the TCU last_seen timestamp",
            "Ask the customer to move the vehicle to open sky and start the engine to re-establish "
            "cellular",
            "Trigger a re-push of the entitlement",
            "Verify the TCU acknowledges and the feature works",
        ],
        "eta_ttr": (
            "Under 15 minutes once the vehicle is online; if a hardware fault is found, an RMA "
            "path of several days"
        ),
        "severity": "S4",
        "priority": "P3",
        "churn_risk": {
            "level": "low-medium",
            "why": "frustrating but environmental; good expectation-setting keeps the customer "
            "satisfied",
        },
        "compensation_policy": {
            "offer": "goodwill gesture only if prolonged or a hardware fault is confirmed",
            "when_proactive": False,
            "escalate_if": "issue is prolonged or a hardware fault is confirmed",
            "note": "typically guidance only; focus on restoring connectivity",
        },
        "customer_communication": (
            "Explain that the car needs to reconnect; give simple steps (move to an open area, "
            "start the engine); set the expectation that it syncs within minutes once online; "
            "offer to follow up. Avoid blaming the customer."
        ),
        "similar_incident": "INC-0742",
        "last_reviewed": "2026-06",
    },
    "ELIGIBILITY_RULE_CONFLICT": {
        "runbook_id": "RB-ELIG-05",
        "rc_class": "ELIGIBILITY_RULE_CONFLICT",
        "title": "Eligibility Rule Conflict",
        "one_line": CUE_LIBRARY["ELIGIBILITY_RULE_CONFLICT"]["one_line"],
        "summary": (
            "The customer paid successfully, but an internal eligibility rule blocked activation "
            "because their vehicle/region/plan combination is not enabled. No entitlement record "
            "was ever created, so nothing reaches the app or vehicle. This is a system-side issue, "
            "not a customer or payment error."
        ),
        "why_plain": (
            "Your payment went through, but an internal activation rule blocked your specific "
            "vehicle/region/plan combination, so the feature was never switched on. This is on our "
            "side and we are fixing it."
        ),
        "why_technical": (
            "payment.succeeded was delivered, but the eligibility check rejected the "
            "region/trim/plan combo, so no entitlement record was created; nothing propagates to "
            "IAM/CCS/TCU."
        ),
        "owner_team": "Entitlement Platform Team",
        "support_contact": "DSD Entitlement on-call (Slack #ent-oncall)",
        "escalation": "DSD L2; DSD L3 if the rule change requires an engineering deploy",
        "detection_cues": [
            "the app keeps prompting Subscribe despite payment",
            "a region- or trim-limited or premium combination",
            "the problem is immediate and persistent (not intermittent)",
        ],
        "confirm_checks": [
            "The payment.succeeded webhook was delivered",
            "No entitlement record exists for the VIN",
            "eligibility_decision = not_eligible",
            "The combination is outside the eligibility matrix",
        ],
        "fix_steps": [
            "Fetch the eligibility decision log for the VIN",
            "Confirm the combination was wrongly rejected (compare against the eligibility matrix)",
            "Update the eligibility matrix for the combination",
            "Replay entitlement creation",
            "Verify the app now shows the feature",
            "Audit other customers on the same combination who may be affected",
        ],
        "eta_ttr": (
            "Typically 2-3 hours (median ~140 min) for a matrix config change plus replay; up to "
            "one business day if a code deploy is required"
        ),
        "severity": "S3",
        "priority": "P2",
        "churn_risk": {
            "level": "medium-high",
            "why": "the customer has paid and feels cheated, has often called more than once, and "
            "the risk grows each day it stays unresolved",
        },
        "compensation_policy": {
            "offer": "1-month service credit",
            "when_proactive": True,
            "escalate_if": ">48h -> 3-month credit + written apology + expedited activation",
            "note": "proactively offer the 1-month credit for the affected subscription",
        },
        "customer_communication": (
            "Reassure the customer that payment went through correctly (not their fault); explain "
            "that an internal activation rule is being fixed; commit to a concrete ETA (24h); "
            "proactively offer the credit. Never blame the customer or ask them to repurchase."
        ),
        "similar_incident": "INC-0501",
        "last_reviewed": "2026-06",
    },
}

#: Severity may shift for combo/region-wide eligibility or TCU hardware faults — documented note.
SEVERITY_NOTES: dict[str, str] = {
    "TCU_OFFLINE": "S4 (S3 if a hardware fault)",
    "ELIGIBILITY_RULE_CONFLICT": "S3 (S2 if combo/region-wide)",
    "ENTITLEMENT_CACHE_STALE": "S3",
}


def runbook_for(rc_class: str) -> dict[str, Any]:
    """Return the structured runbook dict for an RC class.

    Raises:
        KeyError: If ``rc_class`` is not one of the three in-catalog RCs.
    """
    return RUNBOOKS[rc_class]


def _numbered(items: list[str]) -> str:
    """Render a list as a numbered markdown block."""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def _prose(items: list[str]) -> str:
    """Render a list as a single prose sentence (semicolon-joined)."""
    return "; ".join(items) + "."


def render_runbook(rc_class: str) -> str:
    """Render the §2.3 runbook document (markdown) for an RC from its structured dict.

    Một nguồn (RUNBOOKS), một bản document — không bao giờ lệch nhau. Dùng cho review/demo/dạy
    model (runbook_id) và để generator trích từng field thành Q&A.

    Args:
        rc_class: One of :data:`ROOT_CAUSES`.

    Returns:
        The professional runbook document as a markdown string.
    """
    rb = RUNBOOKS[rc_class]
    sev = SEVERITY_NOTES.get(rc_class, rb["severity"])
    churn = rb["churn_risk"]
    comp = rb["compensation_policy"]
    proactive = "proactively offered" if comp["when_proactive"] else "not offered proactively"
    return "\n".join(
        [
            f"**{rb['runbook_id']} — {rb['title']}**",
            f"*Owner: {rb['owner_team']} · Severity: {sev} · Priority: {rb['priority']} · "
            f"Last reviewed: {rb['last_reviewed']}*",
            "",
            f"**Summary.** {rb['summary']}",
            "",
            f"**When this applies (detection cues).** {_prose(rb['detection_cues'])}",
            "",
            "**Confirm before acting.**",
            _numbered(rb["confirm_checks"]),
            "",
            "**Resolution steps.**",
            _numbered(rb["fix_steps"]),
            "",
            f"**Time to resolution.** {rb['eta_ttr']}.",
            "",
            f"**Escalation.** {rb['escalation']}.",
            "",
            f"**Customer impact & retention.** Churn risk: {churn['level']} — {churn['why']}.",
            "",
            f"**Goodwill / compensation policy.** Offer: {comp['offer']}; {proactive}; "
            f"escalate if {comp['escalate_if']}. {comp['note']}.",
            "",
            f"**Customer communication.** {rb['customer_communication']}",
        ]
    )


# ---------------------------------------------------------------------------
# Appendix A — system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are the Honda Entitlement Onboarding Resolver, an offline diagnostic assistant used "
    "INTERNALLY by Honda support/operations staff. The input is a raw customer complaint in "
    "natural language; the customer does not know the technical cause. You answer from internal "
    "knowledge only (closed-book) and never call external tools.\n"
    "Reason step by step inside <think>...</think>:\n"
    '(1) read the concrete cues in the complaint (e.g., "parked in a garage", "app shows active '
    'on web but not app", "it keeps asking me to subscribe");\n'
    "(2) form a leading hypothesis among the known root causes, with one or two alternatives "
    "(a differential);\n"
    '(3) you have NO system logs — do NOT invent telemetry values (no timestamps, no "record '
    'found/not found" as fact); reason only from the complaint\'s cues plus your knowledge of how '
    "the system fails;\n"
    "(4) give a calibrated confidence (usually 0.55-0.85 from a complaint alone) and list what a "
    "human should check to confirm;\n"
    "(5) if there is no distinguishing cue or it is outside the known catalog, set "
    "leading_root_cause = INSUFFICIENT_EVIDENCE and route to a human.\n"
    "After </think>, output ONE JSON object matching the resolution-package schema (diagnosis + "
    "runbook fields + artifacts). Do not add text after the JSON."
)
