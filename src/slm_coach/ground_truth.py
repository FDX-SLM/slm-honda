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

#: The five in-catalog root-cause classes the model may conclude (one per pipeline node).
ROOT_CAUSES: tuple[str, ...] = (
    "TCU_OFFLINE",
    "ENTITLEMENT_CACHE_STALE",
    "ELIGIBILITY_RULE_CONFLICT",
    "PAYMENT_WEBHOOK_LOST",
    "TOKEN_SCOPE",
)

#: RCs the rule-based backup generator can synthesize complaints for (full cue/opener data).
#: The 2 newer classes are covered by Claude-hand-distilled data, not the template generator.
GENERATABLE_RCS: tuple[str, ...] = (
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
    "PAYMENT_WEBHOOK_LOST": "RB-PAY-01",
    "TOKEN_SCOPE": "RB-IAM-03",
}

#: Short, stable group tag used for per-slice eval/holdout (mirrors schema ``Mode``).
RC_TO_SLICE: dict[str, str] = {
    "TCU_OFFLINE": "tcu_offline",
    "ENTITLEMENT_CACHE_STALE": "cache_stale",
    "ELIGIBILITY_RULE_CONFLICT": "eligibility",
    "PAYMENT_WEBHOOK_LOST": "payment_webhook",
    "TOKEN_SCOPE": "token_scope",
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
    "PAYMENT_WEBHOOK_LOST": {
        "one_line": (
            "Payment went through, but the payment-to-entitlement event did not complete, so the "
            "entitlement was never created."
        ),
        "cues": [
            "the purchase is very recent (minutes or hours ago)",
            "paid this morning and it still has not switched on",
            "the charge shows pending or processing",
            "charged but nothing activated yet, often self-resolves shortly",
        ],
        "distinguishing_from": (
            "ELIGIBILITY_RULE_CONFLICT: eligibility is a persistent rule rejection that does not "
            "self-resolve; a lost/delayed activation is tied to a very recent purchase and often "
            "settles on its own or after a replay."
        ),
    },
    "TOKEN_SCOPE": {
        "one_line": (
            "The entitlement exists, but the IAM token/scope was not refreshed, so the feature "
            "returns a 403 / permission error."
        ),
        "cues": [
            "error 403 or permission denied when opening the feature",
            "the app logs me out when I tap the feature",
            "sign-in works otherwise but the feature is blocked",
            "still blocked after signing out and back in",
        ],
        "distinguishing_from": (
            "ENTITLEMENT_CACHE_STALE: a cache shows the feature missing/off; a token-scope issue "
            "shows an explicit permission error (403) or logs the user out on the feature."
        ),
    },
}

#: Out-of-catalog cues → abstention (NOT a concrete RC). Token 403, billing, app crash, OTA.
OUT_OF_CATALOG_CUES: list[str] = [
    "I want a refund / I am disputing the charge",
    "I think I was double charged and want my money back",
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
        "customer_action": (
            "In the meantime, please pull down to refresh in the app, or log out and back in, and "
            "the feature should show up."
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
        "customer_action": (
            "When you get a chance, please move the car to an open area and start the engine for a "
            "few minutes so it can reconnect; the feature should sync within minutes."
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
        "customer_action": (
            "There's nothing you need to do on your end; we're correcting the activation rule now "
            "and will confirm within 24 hours."
        ),
        "similar_incident": "INC-0501",
        "last_reviewed": "2026-06",
    },
    "PAYMENT_WEBHOOK_LOST": {
        "runbook_id": "RB-PAY-01",
        "rc_class": "PAYMENT_WEBHOOK_LOST",
        "title": "Payment Succeeded but Activation Event Lost",
        "one_line": "Payment went through, but the payment-to-entitlement event did not complete, so the entitlement was never created.",
        "summary": "The customer paid successfully, but the event that should create the entitlement did not complete, so the feature was never switched on. Recent purchases often settle on their own; older ones need the activation to be replayed.",
        "why_plain": "Your payment went through fine, but the step that turns on your feature did not finish. If you just purchased it, it often completes on its own shortly; otherwise we will re-run the activation for you.",
        "why_technical": "payment is captured, but the payment-to-entitlement event did not complete (delayed or dropped), so no entitlement record was created and nothing propagates downstream.",
        "owner_team": "DSD Payments Team",
        "support_contact": "DSD Payments on-call (Slack #pay-oncall)",
        "escalation": "DSD L2; DSD L3 if the activation must be replayed manually at scale",
        "detection_cues": [
            "the purchase is very recent (minutes or hours ago)",
            "paid this morning and the feature still has not switched on",
            "the charge shows pending or processing",
            "charged but nothing activated, often self-resolves shortly",
        ],
        "confirm_checks": [
            "The payment was captured for the order",
            "Whether an entitlement record was created",
            "Whether the activation event is still in retry",
        ],
        "fix_steps": [
            "Confirm the payment was captured",
            "Check whether the activation event is delayed or dropped",
            "Wait for auto-retry if the purchase is very recent",
            "Replay the payment-to-entitlement event",
            "Verify the entitlement is created and the app shows the feature",
        ],
        "eta_ttr": "Often self-resolves within the hour for very recent purchases; otherwise ~1 hour to replay the activation",
        "severity": "S3",
        "priority": "P2",
        "churn_risk": {"level": "low-medium", "why": "the customer has paid, but it often self-resolves quickly with good expectation-setting"},
        "compensation_policy": {"offer": "goodwill gesture only if it stays unresolved beyond a day", "when_proactive": False, "escalate_if": ">1 day -> 1-month credit", "note": "usually guidance only; reassure that the payment is fine"},
        "customer_communication": "Reassure the customer the payment went through correctly; explain the activation just needs to complete; give a concrete expectation (often within the hour, otherwise we replay it); never ask them to repurchase.",
        "customer_action": "Your payment is fine — please give it a little time; if it does not switch on shortly, we will re-run the activation for you.",
        "similar_incident": "INC-0815",
        "last_reviewed": "2026-06",
    },
    "TOKEN_SCOPE": {
        "runbook_id": "RB-IAM-03",
        "rc_class": "TOKEN_SCOPE",
        "title": "IAM Token / Scope Not Refreshed",
        "one_line": "The entitlement exists, but the IAM token/scope was not refreshed, so the feature returns a 403.",
        "summary": "The subscription is active, but the customer's access token does not yet carry the scope for the feature, so tapping it returns a permission error. Refreshing the token or re-authenticating restores access.",
        "why_plain": "You do have the subscription, but your sign-in session did not pick up the new permission yet, so the app blocks the feature. Signing out and back in usually refreshes it.",
        "why_technical": "the entitlement is active, but the IAM/HIDAS token was issued without the entitlement's scope (stale or expired), so the feature endpoint returns 403 until the token/scope is refreshed.",
        "owner_team": "HG Identity Team",
        "support_contact": "HG Identity on-call (Slack #iam-oncall)",
        "escalation": "HG L2 if the entitlement-to-scope mapping is wrong (needs a reissue)",
        "detection_cues": [
            "error 403 or permission denied when opening the feature",
            "the app logs the customer out when they tap the feature",
            "sign-in works otherwise but the feature is blocked",
            "still blocked after signing out and back in",
        ],
        "confirm_checks": [
            "The entitlement is active for the account",
            "Whether the access token carries the feature scope",
            "Whether the token is expired",
        ],
        "fix_steps": [
            "Confirm the entitlement is active for the account",
            "Check whether the token carries the feature scope",
            "Force a token/scope refresh (re-authenticate)",
            "If the scope mapping is wrong, fix it and reissue the token",
            "Verify the feature no longer returns 403",
        ],
        "eta_ttr": "Under 1 hour; immediate after a token refresh in most cases",
        "severity": "S3",
        "priority": "P2",
        "churn_risk": {"level": "low-medium", "why": "a permission error feels broken, but the fix is fast once the token is refreshed"},
        "compensation_policy": {"offer": "small goodwill gesture only if it recurs", "when_proactive": False, "escalate_if": "recurring -> investigate the scope mapping", "note": "usually none; not offered proactively"},
        "customer_communication": "Reassure the customer they do have access; explain it is a sign-in/permission refresh, not a lost subscription; guide them through a re-login; do not imply they did anything wrong.",
        "customer_action": "Please sign out and back in to refresh your permissions; if the feature still shows a permission error, we will refresh it on our side.",
        "similar_incident": "INC-0923",
        "last_reviewed": "2026-06",
    },
}

#: Severity may shift for combo/region-wide eligibility or TCU hardware faults — documented note.
SEVERITY_NOTES: dict[str, str] = {
    "TCU_OFFLINE": "S4 (S3 if a hardware fault)",
    "ELIGIBILITY_RULE_CONFLICT": "S3 (S2 if combo/region-wide)",
    "ENTITLEMENT_CACHE_STALE": "S3",
    "PAYMENT_WEBHOOK_LOST": "S3 (S2 if many orders affected)",
    "TOKEN_SCOPE": "S3 (S2 if a scope-mapping bug)",
}

# ---------------------------------------------------------------------------
# Sub-causes (failure modes) per RC class. The class is the cue-classification target;
# the sub_cause is the specific failure inside it. owner/support/escalation/runbook_id stay per
# class (fidelity-locked); why_technical / fix_steps / severity / eta_ttr vary by sub_cause.
# ``escalate_note`` tailors the customer ladder's final (escalation) rung to the sub_cause.
# ---------------------------------------------------------------------------
SUB_CAUSES: dict[str, dict[str, dict[str, Any]]] = {
    "TCU_OFFLINE": {
        "no_signal_garage": {"why_technical": "the TCU is in an underground/garage area with no cellular, so the push was never delivered.", "fix_steps": ["Confirm the entitlement is active", "Ask the customer to move the vehicle to open sky", "Re-push the entitlement", "Verify the TCU acknowledges"], "severity": "S4", "eta_ttr": "Under 15 minutes once the vehicle is back in signal", "escalate_note": "if it still fails after the car has clear signal, contact support — we will check the vehicle side."},
        "weak_signal_remote": {"why_technical": "the TCU is in a weak-coverage (rural) area, so the push intermittently fails to deliver.", "fix_steps": ["Confirm the entitlement is active", "Ask the customer to retry from a better-coverage area", "Re-push the entitlement", "Verify delivery"], "severity": "S4", "eta_ttr": "Under 15 minutes from a good-coverage location", "escalate_note": "if better coverage still does not help, contact support to check the vehicle side."},
        "tcu_asleep": {"why_technical": "the TCU entered low-power sleep after a long idle period, so it is not listening for the push.", "fix_steps": ["Confirm the entitlement is active", "Ask the customer to start the engine / drive briefly to wake the TCU", "Re-push the entitlement", "Verify acknowledgement"], "severity": "S4", "eta_ttr": "Under 15 minutes after the vehicle is woken", "escalate_note": "if waking the car does not help, contact support for a vehicle-side check."},
        "tcu_firmware_hang": {"why_technical": "the TCU is online but its telematics stack has hung, so it does not process the push despite good signal.", "fix_steps": ["Confirm the entitlement is active", "Ask the customer to fully power-cycle the vehicle", "Re-push the entitlement", "Verify acknowledgement"], "severity": "S3", "eta_ttr": "About 30 minutes; immediate after a successful power-cycle", "escalate_note": "if a power-cycle with good signal still fails, contact support — this needs a vehicle-side reset."},
        "tcu_hardware_fault": {"why_technical": "the TCU appears faulty: it does not connect even with confirmed good signal over an extended period.", "fix_steps": ["Confirm the entitlement is active", "Confirm the vehicle has had good signal repeatedly", "Open a hardware-fault investigation", "Initiate the RMA path if confirmed"], "severity": "S3", "eta_ttr": "RMA path of several days if a hardware fault is confirmed", "escalate_note": "since it persists with good signal for days, contact support for a hardware check / RMA — no further self-service will help."},
        "low_12v_battery": {"why_technical": "a low 12V battery (long idle) is starving the TCU, so it cannot stay connected.", "fix_steps": ["Confirm the entitlement is active", "Ask the customer to charge/replace the 12V battery", "Re-push the entitlement once the battery is healthy", "Verify"], "severity": "S4", "eta_ttr": "Once the 12V battery is charged (hours)", "escalate_note": "if charging the battery does not restore it, contact support for a vehicle-side check."},
        "carrier_outage": {"why_technical": "a regional cellular outage is preventing the push from reaching the TCU.", "fix_steps": ["Confirm the entitlement is active", "Check for a known carrier outage in the area", "Wait for the carrier to recover, then re-push", "Verify"], "severity": "S3", "eta_ttr": "Depends on carrier recovery", "escalate_note": "if a wider outage is suspected, contact support; this is network-side and will recover."},
    },
    "ENTITLEMENT_CACHE_STALE": {
        "app_client_cache": {"why_technical": "the phone app holds a stale local cache, so it renders the feature as off.", "fix_steps": ["Confirm the entitlement is active for the VIN", "Ask the customer to refresh / re-login / reinstall", "Verify the app shows the feature"], "severity": "S3", "eta_ttr": "Minutes; immediate after a re-login/reinstall", "escalate_note": "if a reinstall and re-login still do not help, contact support to refresh it server-side."},
        "ccs_server_cache_ttl": {"why_technical": "the CCS server cache TTL has not expired, so it keeps serving a stale view despite a fresh client.", "fix_steps": ["Confirm the entitlement is active", "Force a server cache invalidation", "Re-sync to CCS", "Verify"], "severity": "S3", "eta_ttr": "Under 30 minutes; immediate after invalidation", "escalate_note": "if it persists after a reinstall, contact support — we will force a server-side cache refresh."},
        "cdn_edge_cache": {"why_technical": "an edge/CDN node is serving a stale cached response in the customer's region.", "fix_steps": ["Confirm the entitlement is active", "Purge the edge cache for the entitlement", "Verify across devices/regions"], "severity": "S3", "eta_ttr": "Under 30 minutes after an edge purge", "escalate_note": "if some devices/regions work and others do not, contact support to purge the edge cache."},
        "multi_device_sync_lag": {"why_technical": "one device holds a stale view while others are correct — a per-device sync lag.", "fix_steps": ["Confirm the entitlement is active", "Refresh the lagging device", "Verify it matches the others"], "severity": "S4", "eta_ttr": "Minutes after refreshing the lagging device", "escalate_note": "if the one device still lags after a refresh, contact support."},
        "invalidation_missed": {"why_technical": "a plan change did not trigger a cache invalidation, so the old view persists.", "fix_steps": ["Confirm the new entitlement is active", "Manually invalidate the cache for the change", "Re-sync", "Verify"], "severity": "S3", "eta_ttr": "Under 30 minutes after a manual invalidation", "escalate_note": "if the app did not update after a plan change, contact support to invalidate the cache."},
    },
    "ELIGIBILITY_RULE_CONFLICT": {
        "region_not_in_matrix": {"why_technical": "the customer's region is missing from the eligibility matrix, so the combo was rejected and no entitlement created.", "fix_steps": ["Fetch the eligibility decision for the VIN", "Confirm the region was wrongly excluded", "Add the region to the matrix", "Replay entitlement creation", "Verify"], "severity": "S3", "eta_ttr": "2-3 hours for a matrix change plus replay", "escalate_note": "do not repay — this is a region rule on our side; we will fix it and apply a credit."},
        "trim_not_in_matrix": {"why_technical": "the vehicle trim is missing from the eligibility matrix, so the combo was rejected.", "fix_steps": ["Fetch the eligibility decision", "Confirm the trim was wrongly excluded", "Add the trim to the matrix", "Replay entitlement creation", "Verify"], "severity": "S3", "eta_ttr": "2-3 hours for a matrix change plus replay", "escalate_note": "do not repay — this is a trim rule on our side; we will fix it and apply a credit."},
        "plan_tier_not_enabled": {"why_technical": "the plan tier is not enabled for this combo in the matrix, so activation was blocked.", "fix_steps": ["Fetch the eligibility decision", "Confirm the plan tier should be enabled", "Enable the plan tier for the combo", "Replay entitlement creation", "Verify"], "severity": "S3", "eta_ttr": "2-3 hours for a config change plus replay", "escalate_note": "do not repay — we will enable your plan for this combo and apply a credit."},
        "matrix_stale_new_model_year": {"why_technical": "a newly launched model year is not yet in the eligibility matrix, so eligible customers are wrongly rejected.", "fix_steps": ["Fetch the eligibility decision", "Confirm the new model year is missing", "Update the matrix for the new model year", "Replay entitlement creation", "Verify and audit same-combo customers"], "severity": "S2", "eta_ttr": "Up to one business day; affects multiple customers", "escalate_note": "do not repay — your model year just needs adding on our side; we will fix it and apply a credit."},
        "rule_misconfig_bug": {"why_technical": "the eligibility rule logic is misconfigured and wrongly rejects a valid combo; a code fix is required.", "fix_steps": ["Fetch the eligibility decision", "Confirm the rule wrongly rejects a valid combo", "Open an engineering fix for the rule", "Deploy and replay entitlement creation", "Verify and audit affected customers"], "severity": "S2", "eta_ttr": "Up to one business day if a code deploy is required", "escalate_note": "do not repay — this is a rule bug on our side; engineering will fix it and we will apply a credit."},
        "promo_bundle_edge": {"why_technical": "a promo/bundle purchase hit an eligibility edge case, so the entitlement was not created.", "fix_steps": ["Fetch the eligibility decision", "Confirm the promo/bundle should grant the feature", "Manually grant the entitlement", "Verify"], "severity": "S3", "eta_ttr": "2-3 hours for a manual grant", "escalate_note": "do not repay — your promo/bundle should include this; we will grant it and apply a credit if relevant."},
    },
    "PAYMENT_WEBHOOK_LOST": {
        "webhook_delayed": {"why_technical": "the payment-to-entitlement event is delayed in retry for a very recent purchase; it usually settles on its own.", "fix_steps": ["Confirm the payment was captured", "Confirm the activation event is still in retry", "Allow auto-retry to settle", "Verify the entitlement is created"], "severity": "S4", "eta_ttr": "Often within the hour on its own", "escalate_note": "your payment is fine — give it a little time; if it does not switch on shortly, contact support to replay it."},
        "webhook_dropped": {"why_technical": "the payment-to-entitlement event was dropped (not just delayed), so the entitlement was never created.", "fix_steps": ["Confirm the payment was captured", "Confirm no entitlement was created", "Replay the payment-to-entitlement event", "Verify"], "severity": "S3", "eta_ttr": "About 1 hour to replay the activation", "escalate_note": "your payment is fine and you should not repay — contact support to replay the activation."},
        "payment_pending_review": {"why_technical": "the payment is pending/under review, so it has not settled and the activation has not started.", "fix_steps": ["Check the payment status", "Wait for settlement or manually release the hold", "Trigger entitlement creation once settled", "Verify"], "severity": "S3", "eta_ttr": "Hours, depending on settlement", "escalate_note": "your payment is still settling — please wait; contact support if it stays pending."},
        "duplicate_charge_no_grant": {"why_technical": "the customer was charged (possibly twice) but no entitlement was granted due to the failed activation.", "fix_steps": ["Confirm the charge(s)", "De-duplicate the charge", "Grant the entitlement", "Refund any duplicate", "Verify"], "severity": "S3", "eta_ttr": "About 1 hour, plus refund timing for any duplicate", "escalate_note": "do not repay — contact support; we will grant your feature and refund any duplicate charge."},
        "partial_provision": {"why_technical": "the order was created but the entitlement step did not finish, leaving a partial provision.", "fix_steps": ["Confirm the order exists", "Confirm the entitlement step did not complete", "Re-run entitlement creation", "Verify"], "severity": "S3", "eta_ttr": "About 1 hour to re-run the entitlement step", "escalate_note": "your order is there — contact support to finish the activation; no need to repay."},
    },
    "TOKEN_SCOPE": {
        "stale_scope": {"why_technical": "the token was issued before the entitlement, so it lacks the feature scope until refreshed.", "fix_steps": ["Confirm the entitlement is active", "Confirm the token lacks the feature scope", "Force a token/scope refresh", "Verify the 403 clears"], "severity": "S3", "eta_ttr": "Under 1 hour; immediate after a refresh", "escalate_note": "if signing out and back in does not clear the permission error, contact support to refresh your access."},
        "token_expired": {"why_technical": "the session token has expired and the app is not refreshing it, so the feature endpoint returns 403.", "fix_steps": ["Confirm the entitlement is active", "Confirm the token is expired", "Re-authenticate / clear the session", "Verify"], "severity": "S3", "eta_ttr": "Under 1 hour; immediate after re-auth", "escalate_note": "if a fresh sign-in does not help, contact support to reset your session."},
        "scope_mapping_bug": {"why_technical": "the entitlement-to-scope mapping is wrong, so even a refreshed token lacks the scope; needs a fix and reissue.", "fix_steps": ["Confirm the entitlement is active", "Confirm the scope mapping is wrong", "Fix the entitlement-to-scope mapping", "Reissue the token", "Verify"], "severity": "S2", "eta_ttr": "Hours; needs a mapping fix and reissue", "escalate_note": "since it persists after re-login, contact support — we will fix the permission mapping on our side."},
        "multi_account_mismatch": {"why_technical": "the feature is on one account but the customer is signed into another, so the active session lacks the scope.", "fix_steps": ["Identify which account holds the entitlement", "Ask the customer to sign into the correct account (or link accounts)", "Verify access"], "severity": "S3", "eta_ttr": "Under 1 hour", "escalate_note": "check you are signed into the account that bought the feature; contact support to link your accounts if needed."},
    },
}


def sub_causes_for(rc_class: str) -> tuple[str, ...]:
    """Valid sub_cause keys for an RC class (empty tuple for abstention/unknown)."""
    return tuple(SUB_CAUSES.get(rc_class, {}).keys())


def resolve_sub_cause(rc_class: str, sub_cause: str | None) -> dict[str, Any]:
    """Return the per-sub-cause overrides (why_technical/fix_steps/severity/eta_ttr/escalate_note).

    Falls back to the base runbook values when ``sub_cause`` is None or unknown.
    """
    rb = RUNBOOKS.get(rc_class, {})
    base = {
        "why_technical": rb.get("why_technical", ""),
        "fix_steps": rb.get("fix_steps", []),
        "severity": rb.get("severity", ""),
        "eta_ttr": rb.get("eta_ttr", ""),
        "escalate_note": None,
    }
    if sub_cause and sub_cause in SUB_CAUSES.get(rc_class, {}):
        base.update(SUB_CAUSES[rc_class][sub_cause])
    return base


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
# Customer self-service ladder (§3.2 customer_self_service) — a step-by-step
# troubleshooting protocol the customer can follow while support works the ticket.
# Easy → hard; early rungs RULE OUT other causes (self-triage), later rungs RESOLVE
# the leading cause, last rung ESCALATES. Customer-safe actions only (never internal
# ops like DEL ent:{vin}); no fabricated telemetry; no over-promise.
# ---------------------------------------------------------------------------

#: Ordered protocols per leading RC. Each step: tier, action, addresses (which hypothesis the step
#: tests/fixes), expected_time, verify (how the customer knows it worked), if_fails.
_SELF_SERVICE: dict[str, list[dict[str, str]]] = {
    "TCU_OFFLINE": [
        {"action": "Open the app, pull down to refresh, then sign out and sign back in. This clears a stale display if that is all it is.",
         "addresses": "ENTITLEMENT_CACHE_STALE", "expected_time": "~2 min",
         "verify": "the feature now appears and works"},
        {"action": "Check that the subscription still shows Active in the app or your online account, so we know this is a connectivity issue and not a billing one.",
         "addresses": "general", "expected_time": "~1 min",
         "verify": "the subscription shows Active"},
        {"action": "Drive or move the vehicle out of the garage/basement into open air with a clear view of the sky, where it can pick up cellular signal.",
         "addresses": "TCU_OFFLINE", "expected_time": "~5 min",
         "verify": "the car is outdoors with signal"},
        {"action": "With the engine running in open air, wait 3-5 minutes for the car to reconnect, then open the app and retry the remote command.",
         "addresses": "TCU_OFFLINE", "expected_time": "~5 min",
         "verify": "the remote command now succeeds"},
        {"action": "If it still spins, lock and unlock the car once with the physical key fob, then retry the remote command in the app to nudge the car to check in.",
         "addresses": "TCU_OFFLINE", "expected_time": "~3 min",
         "verify": "the command goes through"},
        {"action": "If it is still failing after the car has had signal for about 15 minutes, fully power the vehicle off, wait a minute, restart it, and try once more.",
         "addresses": "TCU_OFFLINE", "expected_time": "~20 min",
         "verify": "the feature responds"},
        {"action": "If none of these work, the problem is not signal-related. Contact support; we will check for a hardware fault and resolve it.",
         "addresses": "escalate", "expected_time": "handled by support",
         "verify": "a specialist takes over"},
    ],
    "ENTITLEMENT_CACHE_STALE": [
        {"action": "In the app, pull down to refresh, then sign out and sign back in. This forces the app to re-read your current entitlements and usually restores the feature immediately.",
         "addresses": "ENTITLEMENT_CACHE_STALE", "expected_time": "~2 min",
         "verify": "the feature appears and works"},
        {"action": "If it flickers back then disappears, fully close the app (swipe it away) and reopen it, rather than just minimizing it.",
         "addresses": "ENTITLEMENT_CACHE_STALE", "expected_time": "~2 min",
         "verify": "the feature stays visible"},
        {"action": "Reinstall the app from the store, then sign in again. This clears cached data stored on the phone.",
         "addresses": "ENTITLEMENT_CACHE_STALE", "expected_time": "~5 min",
         "verify": "the feature now shows after re-login"},
        {"action": "Open the app once while the car is in an area with good signal and the engine is on, so a fresh sync can reach the vehicle too.",
         "addresses": "TCU_OFFLINE", "expected_time": "~5 min",
         "verify": "the feature works end to end"},
        {"action": "Check the website/account. If the feature shows Active there but not in the app after the steps above, note that — it confirms a sync issue on our side.",
         "addresses": "general", "expected_time": "~2 min",
         "verify": "you can see the web vs app mismatch"},
        {"action": "If it still will not show after a reinstall and re-login, contact support — we will force a cache refresh on our side; it is quick.",
         "addresses": "escalate", "expected_time": "handled by support",
         "verify": "a specialist forces the refresh"},
    ],
    "ELIGIBILITY_RULE_CONFLICT": [
        {"action": "First, pull to refresh and sign out and back in, on the off chance it is only a display delay.",
         "addresses": "ENTITLEMENT_CACHE_STALE", "expected_time": "~2 min",
         "verify": "the feature appears (rules out a cache issue)"},
        {"action": "Force-close and reopen the app, or reinstall it, then sign in again.",
         "addresses": "ENTITLEMENT_CACHE_STALE", "expected_time": "~5 min",
         "verify": "still prompts to subscribe (rules out the app/cache)"},
        {"action": "Make sure your car has had cellular signal recently (not parked underground) and retry.",
         "addresses": "TCU_OFFLINE", "expected_time": "~5 min",
         "verify": "still prompts to subscribe (rules out signal)"},
        {"action": "Have your payment confirmation or receipt handy. Your payment is not the problem, but it helps us locate the order quickly.",
         "addresses": "general", "expected_time": "~2 min",
         "verify": "you have the order reference ready"},
        {"action": "If it still keeps asking you to subscribe, please do NOT pay again — this is an activation rule on our side for your specific vehicle/plan. Contact support; we will fix it and apply a service credit.",
         "addresses": "escalate", "expected_time": "handled by support (2-3h)",
         "verify": "a specialist updates the rule and credits the account"},
    ],
}

_SELF_SERVICE["PAYMENT_WEBHOOK_LOST"] = [
    {"action": "In the app, pull down to refresh and sign out and back in, in case it just needs to re-read your purchase.",
     "addresses": "ENTITLEMENT_CACHE_STALE", "expected_time": "~2 min", "verify": "the feature appears"},
    {"action": "Check your order/receipt is showing in the app or your account, so you have the order reference.",
     "addresses": "general", "expected_time": "~2 min", "verify": "you can see the order"},
    {"action": "If you bought it very recently, give it a little time — activation often completes on its own within the hour; retry after a while.",
     "addresses": "PAYMENT_WEBHOOK_LOST", "expected_time": "~30-60 min", "verify": "the feature switches on"},
    {"action": "Make sure your car has had signal (not parked underground) so a completed activation can reach it, then retry.",
     "addresses": "TCU_OFFLINE", "expected_time": "~5 min", "verify": "the feature works end to end"},
    {"action": "If it still has not switched on, do NOT repurchase — contact support; your payment is fine and we will replay the activation.",
     "addresses": "escalate", "expected_time": "handled by support (~1h)", "verify": "a specialist replays the activation"},
]

_SELF_SERVICE["TOKEN_SCOPE"] = [
    {"action": "Sign out of the app and sign back in — this refreshes your permissions and usually clears the error.",
     "addresses": "TOKEN_SCOPE", "expected_time": "~2 min", "verify": "the permission error clears"},
    {"action": "Force-close and reopen the app; if needed, reinstall it and sign in again.",
     "addresses": "TOKEN_SCOPE", "expected_time": "~5 min", "verify": "the feature opens"},
    {"action": "Make sure you are signed into the same account that purchased the feature (check the email on the account).",
     "addresses": "TOKEN_SCOPE", "expected_time": "~2 min", "verify": "you are on the correct account"},
    {"action": "Note the exact error (e.g. a 403 or permission message) so support can pinpoint it.",
     "addresses": "general", "expected_time": "~1 min", "verify": "you have the error details"},
    {"action": "If it still shows a permission error, contact support — we will refresh your access on our side.",
     "addresses": "escalate", "expected_time": "handled by support (<1h)", "verify": "a specialist refreshes your access"},
]

#: Generic safe ladder for abstention (root cause unknown — do not assume any single cause).
_SELF_SERVICE_GENERIC: list[dict[str, str]] = [
    {"action": "Pull to refresh and sign out and back in in the app.",
     "addresses": "general", "expected_time": "~2 min", "verify": "the feature appears"},
    {"action": "Force-close and reopen the app; if needed, reinstall it and sign in again.",
     "addresses": "general", "expected_time": "~5 min", "verify": "the feature appears"},
    {"action": "Make sure the car has cellular signal (move it to open air if it has been parked underground) and retry.",
     "addresses": "general", "expected_time": "~5 min", "verify": "the feature responds"},
    {"action": "Note any error message or code you see, and whether the feature shows active on the website.",
     "addresses": "general", "expected_time": "~2 min", "verify": "you have details to share"},
    {"action": "If none of this helps, a specialist will look into it and may need a few details to pin down the cause.",
     "addresses": "escalate", "expected_time": "handled by support", "verify": "a specialist triages it"},
]


def customer_self_service_ladder(
    leading_rc: str, sub_cause: str | None = None
) -> list[dict[str, Any]]:
    """Return the ordered, step-by-step customer self-service protocol for a leading RC.

    Easy → hard: early rungs rule out other causes (self-triage), middle rungs resolve the leading
    cause, the last rung escalates. For an unknown/abstention RC, return the generic safe ladder.
    When ``sub_cause`` is given, the final (escalation) rung's action is tailored to it (e.g. a
    hardware fault routes to an RMA check rather than the generic hand-off). Each rung carries
    ``tier``, ``action``, ``addresses``, ``expected_time``, ``verify``, ``if_fails``.
    """
    steps = _SELF_SERVICE.get(leading_rc, _SELF_SERVICE_GENERIC)
    note = resolve_sub_cause(leading_rc, sub_cause).get("escalate_note") if sub_cause else None
    out: list[dict[str, Any]] = []
    for i, s in enumerate(steps, start=1):
        last = i == len(steps)
        action = s["action"]
        if last and note:  # tailor the escalation rung to the specific failure mode
            action = note[0].upper() + note[1:]
        out.append({
            "tier": i,
            "action": action,
            "addresses": s["addresses"],
            "expected_time": s["expected_time"],
            "verify": s["verify"],
            "if_fails": (f"go to step {i + 1}" if not last else "a specialist takes over"),
        })
    return out


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
