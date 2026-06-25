r"""Core synthesis: raw complaint → <think> trace → resolution package + artifacts.

Mọi thứ ở đây trích từ :mod:`slm_coach.ground_truth` (không bịa fact). Một :class:`Case` gói:
lời than thô (user turn), trace ``<think>`` (suy luận từ cue, KHÔNG telemetry), và resolution
package (output JSON §3.2). Generator nhóm SFT/DPO/eval lắp ráp từ các hàm này.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any

from slm_coach.ground_truth import (
    ABSTAIN,
    CUE_LIBRARY,
    OUT_OF_CATALOG_CUES,
    ROOT_CAUSES,
    RUNBOOKS,
    SYSTEM_PROMPT,
    VAGUE_COMPLAINTS,
    runbook_for,
)

__all__ = [
    "SYSTEM_PROMPT",
    "Case",
    "assistant_content",
    "build_abstention",
    "build_case",
    "build_resolution",
    "sft_messages",
]

# ---------------------------------------------------------------------------
# Per-RC complaint scaffolding — the exact cue fragments are reused verbatim as
# evidence_in_ticket so cue-grounding is guaranteed by construction.
# ---------------------------------------------------------------------------

_FEATURES = [
    "Remote Start",
    "Remote Climate",
    "Remote Lock/Unlock",
    "the Touring package",
    "Vehicle Finder",
    "the connected services subscription",
]
_TIME = ["3 days ago", "a couple of days ago", "yesterday", "last week", "a few days ago"]

#: Cue sentence variants per RC. Each entry is a (complaint_sentence, evidence_fragment) pair;
#: the evidence fragment is built from words that appear in the sentence, so cue-grounding holds
#: by construction. Element [0] is the PRIMARY cue (always injected) — it makes
#: ``oracle.detect_rcs`` return this RC, so RC↔cue match is guaranteed.
_RC_CUES: dict[str, list[tuple[str, str]]] = {
    "TCU_OFFLINE": [
        (
            "My car has been parked in an underground garage all week.",
            "parked in underground garage all week",
        ),
        ("When I tap it in the app it just spins and then times out.", "the app spins and times out"),
        ("The car has no signal where it is parked.", "car has no signal"),
        ("The car hasn't been driven in days.", "car not driven in days"),
        ("The subscription itself shows active though.", "subscription shows active"),
    ],
    "ENTITLEMENT_CACHE_STALE": [
        (
            "I can see it active on the website but the app shows it as off.",
            "active on the website but the app shows off",
        ),
        ("It worked fine yesterday and then suddenly stopped.", "worked fine yesterday then stopped"),
        ("It's intermittent and flickers on and off.", "intermittent and flickers"),
        (
            "Logging out and back in sometimes makes it show up briefly.",
            "logging out and back in sometimes",
        ),
    ],
    "ELIGIBILITY_RULE_CONFLICT": [
        (
            "The app keeps prompting me to Subscribe even though I already paid.",
            "app prompting subscribe even though paid",
        ),
        ("I have a CR-V Touring here in US-West.", "cr v touring in us west"),
        ("It has been like this since I paid, every single time.", "since paid every single time"),
        ("Why does it still ask me to buy when I already paid?", "ask me to buy already paid"),
    ],
}

#: Minimum cues to inject (so weak/strong cases vary the confidence).
_RC_OPENERS = {
    "TCU_OFFLINE": "I bought {feature} {time} and it still won't work.",
    "ENTITLEMENT_CACHE_STALE": "I subscribed to {feature} {time} and now it's acting up.",
    "ELIGIBILITY_RULE_CONFLICT": "I paid for {feature} {time} but I can't use it.",
}

#: The alternative RC the <think> trace down-weights, with the distinguishing reason.
_DIFFERENTIAL_ALT: dict[str, tuple[str, str]] = {
    "TCU_OFFLINE": (
        "ENTITLEMENT_CACHE_STALE",
        "cache issues show an inconsistent active state, not a car-side timeout",
    ),
    "ENTITLEMENT_CACHE_STALE": (
        "TCU_OFFLINE",
        "a TCU offline issue shows a car-side timeout, not an app showing the feature missing",
    ),
    "ELIGIBILITY_RULE_CONFLICT": (
        "ENTITLEMENT_CACHE_STALE",
        "cache stale would still show the entitlement active somewhere; here nothing is active",
    ),
}

#: What the <think> says it cannot see (so it stays calibrated, no fabricated telemetry).
_CANNOT_SEE: dict[str, str] = {
    "TCU_OFFLINE": "the TCU last_seen status",
    "ENTITLEMENT_CACHE_STALE": "the cache invalidation status",
    "ELIGIBILITY_RULE_CONFLICT": "the eligibility decision log",
}

#: to_confirm per RC (what a human should check — no telemetry values).
_TO_CONFIRM: dict[str, list[str]] = {
    "TCU_OFFLINE": ["TCU last_seen timestamp", "entitlement record is active"],
    "ENTITLEMENT_CACHE_STALE": [
        "entitlement record is active for the VIN",
        "CCS cache TTL or last invalidation",
    ],
    "ELIGIBILITY_RULE_CONFLICT": [
        "eligibility decision for the VIN",
        "whether an entitlement record exists",
    ],
}


@dataclass
class Case:
    """A fully-grounded diagnostic case ready to render as an SFT/eval record."""

    leading: str
    complaint: str
    think: str
    resolution: dict[str, Any]
    evidence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Artifacts (§3.2 artifacts block)
# ---------------------------------------------------------------------------

_MERMAID: dict[str, str] = {
    "TCU_OFFLINE": (
        "sequenceDiagram\n"
        "  participant E as Entitlement\n  participant C as CCS\n  participant T as TCU\n"
        "  E->>C: entitlement active\n  C->>T: push activation\n"
        "  Note over T: TCU offline (no cellular) -> not delivered"
    ),
    "ENTITLEMENT_CACHE_STALE": (
        "sequenceDiagram\n"
        "  participant E as Entitlement\n  participant C as CCS Cache\n  participant A as App\n"
        "  E->>C: entitlement active\n  Note over C: cache stale (TTL/invalidation missed)\n"
        "  C-->>A: stale view -> feature shows off"
    ),
    "ELIGIBILITY_RULE_CONFLICT": (
        "sequenceDiagram\n"
        "  participant P as Honda Pay\n  participant E as Entitlement\n  participant A as App\n"
        "  P->>E: payment.succeeded webhook\n"
        "  Note over E: eligibility rejects region/trim/plan -> no entitlement created\n"
        "  A-->>A: keeps prompting Subscribe"
    ),
}


def _render_artifacts(rc: str, rb: dict[str, Any], evidence: list[str]) -> dict[str, str]:
    """Render the RCA / work-order / customer-email / mermaid artifacts for a concrete RC."""
    fix_steps_md = "\n".join(f"{i}. {s}" for i, s in enumerate(rb["fix_steps"], 1))
    evidence_md = "; ".join(evidence) if evidence else "see ticket"
    rca = (
        f"## Root Cause Analysis\n"
        f"**Leading cause:** {rc} ({rb['runbook_id']})\n\n"
        f"**Why:** {rb['why_technical']}\n\n"
        f"**Evidence from ticket:** {evidence_md}\n\n"
        f"**Owner:** {rb['owner_team']} · **Severity:** {rb['severity']} · "
        f"**Priority:** {rb['priority']}"
    )
    work_order = (
        f"## Work Order\n"
        f"- Owner: {rb['owner_team']}\n"
        f"- Support: {rb['support_contact']}\n"
        f"- Escalation: {rb['escalation']}\n"
        f"- ETA: {rb['eta_ttr']}\n\n"
        f"**Steps:**\n{fix_steps_md}"
    )
    email = _customer_email(rb)
    return {
        "rca_md": rca,
        "work_order_md": work_order,
        "customer_email": email,
        "diagram_mermaid": _MERMAID[rc],
    }


def _customer_email(rb: dict[str, Any]) -> str:
    """Compose the customer-facing email from the runbook (why_plain + communication tone)."""
    return (
        "Hi, thanks for reaching out. "
        f"{rb['why_plain']} "
        f"{rb['customer_communication'].split('.')[0]}. "
        "I'll follow up to confirm everything is working."
    )


# ---------------------------------------------------------------------------
# Resolution package
# ---------------------------------------------------------------------------


def build_resolution(
    rc: str, *, confidence: float, evidence: list[str], differential: list[dict[str, str]]
) -> dict[str, Any]:
    """Assemble the §3.2 resolution package for a concrete RC straight from the runbook gold."""
    rb = runbook_for(rc)
    return {
        "diagnosis": {
            "leading_root_cause": rc,
            "confidence": round(confidence, 2),
            "differential": differential,
            "evidence_in_ticket": evidence,
            "to_confirm": _TO_CONFIRM[rc],
        },
        "runbook_id": rb["runbook_id"],
        "why_plain": rb["why_plain"],
        "why_technical": rb["why_technical"],
        "owner_team": rb["owner_team"],
        "support_contact": rb["support_contact"],
        "escalation": rb["escalation"],
        "fix_steps": rb["fix_steps"],
        "eta_ttr": rb["eta_ttr"],
        "severity": rb["severity"],
        "priority": rb["priority"],
        "churn_risk": rb["churn_risk"],
        "compensation": {
            "offer": rb["compensation_policy"]["offer"],
            "proactive": rb["compensation_policy"]["when_proactive"],
            "note": rb["compensation_policy"]["note"],
        },
        "similar_incident": rb["similar_incident"],
        "artifacts": _render_artifacts(rc, rb, evidence),
    }


def _think_for(rc: str, evidence: list[str], confidence: float) -> str:
    """Build a calibrated <think> trace: read cues → leading hypothesis → down-weight alt → confirm."""
    alt_rc, alt_why = _DIFFERENTIAL_ALT[rc]
    cues_phrase = "; ".join(evidence)
    cannot = _CANNOT_SEE[rc]
    one_line = CUE_LIBRARY[rc]["one_line"]
    return (
        f"The cues in the ticket are: {cues_phrase}. {one_line} so my leading hypothesis is {rc}. "
        f"I also considered {alt_rc}, but {alt_why}, so I down-weight it. "
        f"I cannot see {cannot} from the ticket, so I keep confidence around {confidence:.2f} "
        f"and flag what a human should verify."
    )


def build_case(rng: random.Random, rc: str, *, n_cues: int | None = None) -> Case:
    """Synthesize one grounded complaint→resolution case for a concrete RC.

    Args:
        rng: Seeded RNG for reproducible variety.
        rc: One of :data:`slm_coach.ground_truth.ROOT_CAUSES`.
        n_cues: How many distinguishing cues to inject (defaults to 2-3 → varies confidence).

    Returns:
        A :class:`Case` whose resolution passes the oracle by construction.
    """
    if rc not in ROOT_CAUSES:
        raise ValueError(f"build_case expects a concrete RC, got {rc!r}")
    all_cues = _RC_CUES[rc]
    primary = all_cues[0]  # always injected → guarantees detect_rcs == {rc}
    rest = list(all_cues[1:])
    rng.shuffle(rest)
    k = n_cues if n_cues is not None else rng.randint(2, 3)
    chosen = [primary, *rest[: max(0, k - 1)]]
    rng.shuffle(chosen)
    cue_sentences = [c for c, _ in chosen]
    evidence = [e for _, e in chosen]

    opener = _RC_OPENERS[rc].format(feature=rng.choice(_FEATURES), time=rng.choice(_TIME))
    complaint = opener + " " + " ".join(cue_sentences)

    # More cues → higher (but still calibrated ≤0.85) confidence.
    confidence = min(0.82, 0.55 + 0.09 * k + rng.uniform(-0.03, 0.03))
    alt_rc, alt_why = _DIFFERENTIAL_ALT[rc]
    differential = [
        {"rc": rc, "likelihood": "high", "why": CUE_LIBRARY[rc]["one_line"]},
        {"rc": alt_rc, "likelihood": "low", "why": alt_why},
    ]
    think = _think_for(rc, evidence, confidence)
    resolution = build_resolution(
        rc, confidence=confidence, evidence=evidence, differential=differential
    )
    return Case(leading=rc, complaint=complaint, think=think, resolution=resolution, evidence=evidence)


# ---------------------------------------------------------------------------
# Abstention (§3 abstention / §5.5)
# ---------------------------------------------------------------------------

_ABSTAIN_ROUTE = "route to human (DSD L2 triage)"


def build_abstention(rng: random.Random, *, kind: str = "vague") -> Case:
    """Synthesize an abstention case (ambiguous or out-of-catalog) — no fabricated runbook.

    Args:
        rng: Seeded RNG.
        kind: ``"vague"`` (no distinguishing cue) or ``"out_of_catalog"`` (403/billing/crash/OTA).

    Returns:
        A :class:`Case` with ``leading = INSUFFICIENT_EVIDENCE`` that passes the oracle.
    """
    if kind == "out_of_catalog":
        complaint = rng.choice(OUT_OF_CATALOG_CUES)
        # Ground a short literal fragment (≤3 content tokens) so cue-grounding passes.
        evidence = [_short_fragment(complaint)]
    else:
        complaint = rng.choice(VAGUE_COMPLAINTS)
        evidence = []

    confidence = round(rng.uniform(0.25, 0.42), 2)
    # Two-branch differential, both low, with to_confirm prominent.
    a, b = rng.sample(ROOT_CAUSES, 2)
    differential = [
        {"rc": a, "likelihood": "low", "why": "no distinguishing cue in the complaint"},
        {"rc": b, "likelihood": "low", "why": "would need a confirming signal not present here"},
    ]
    to_confirm = [
        "which system shows the feature as active (web/app/vehicle)",
        "whether the customer sees any error code or a Subscribe prompt",
        "whether an entitlement record exists for the VIN",
    ]
    think = (
        "The complaint has no distinguishing cue: it does not say whether the feature shows "
        "active anywhere, whether the car is offline, or whether the app keeps asking to "
        "subscribe. With no single cue, I do not commit to a root cause; I keep confidence low "
        f"({confidence:.2f}) and route to a human after listing what to confirm."
    )
    resolution = {
        "diagnosis": {
            "leading_root_cause": ABSTAIN,
            "confidence": confidence,
            "differential": differential,
            "evidence_in_ticket": evidence,
            "to_confirm": to_confirm,
        },
        "runbook_id": None,
        "why_plain": (
            "We don't have enough information yet to tell what's going on, so we're routing this "
            "to a specialist who will check a few details."
        ),
        "why_technical": "Insufficient distinguishing cues to select a root cause; route to triage.",
        "owner_team": _ABSTAIN_ROUTE,
        "support_contact": _ABSTAIN_ROUTE,
        "escalation": _ABSTAIN_ROUTE,
        "fix_steps": ["route to human triage", "gather the to_confirm signals", "re-triage"],
        "eta_ttr": "TBD",
        "severity": None,
        "priority": None,
        "churn_risk": {"level": "medium", "why": "paid customer with an unresolved, unclear issue"},
        "compensation": None,
        "similar_incident": None,
        "artifacts": {
            "rca_md": "## Root Cause Analysis\nInsufficient evidence — routed to human triage.",
            "work_order_md": f"## Work Order\n- Action: {_ABSTAIN_ROUTE}",
            "customer_email": (
                "Hi, thanks for reaching out. I'm sorry for the trouble. I'm escalating this to a "
                "specialist who will look into it and get back to you shortly with next steps."
            ),
            "diagram_mermaid": "sequenceDiagram\n  participant A as Agent\n  participant H as Human triage\n  A->>H: insufficient evidence -> route",
        },
    }
    return Case(
        leading=ABSTAIN, complaint=complaint, think=think, resolution=resolution, evidence=evidence
    )


def _short_fragment(text: str) -> str:
    """Pick a short (≤3-token) literal fragment from a complaint for grounded abstention evidence."""
    lowered = text.lower()
    for marker in ("403", "logs me out", "logged out", "permission denied", "refund", "dispute",
                   "crashes", "crash", "ota", "update"):
        if marker in lowered:
            return marker
    words = re.sub(r"[^a-z0-9 ]+", " ", lowered).split()
    return " ".join(words[:3])


# ---------------------------------------------------------------------------
# Rendering to chat messages
# ---------------------------------------------------------------------------


def assistant_content(think: str, resolution: dict[str, Any]) -> str:
    """Render an assistant turn: ``<think>...</think>`` + the compact JSON resolution package."""
    payload = json.dumps(resolution, ensure_ascii=False)
    return f"<think>\n{think}\n</think>\n{payload}"


def sft_messages(case: Case) -> list[dict[str, str]]:
    """Render a complaint→resolution case into a system/user/assistant message list."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": case.complaint},
        {"role": "assistant", "content": assistant_content(case.think, case.resolution)},
    ]


def _extract_evidence(complaint: str, rc: str) -> list[str]:
    """Pick complaint sentences that carry a cue for ``rc`` (grounded evidence for the demo)."""
    from slm_coach.oracle import CUE_SIGNATURES, _matches, _norm

    sentences = re.split(r"(?<=[.!?])\s+", complaint.strip())
    sigs = CUE_SIGNATURES.get(rc, [])
    out: list[str] = []
    for sent in sentences:
        norm = _norm(sent)
        if any(_matches(sig, norm) for sig in sigs):
            out.append(sent.strip().rstrip(".!?"))
    return out[:3] or [complaint.strip().rstrip(".!?")[:80]]


def answer_for_complaint(complaint: str) -> Case:
    """Ground-truth answer for an ARBITRARY complaint (demo fallback when no model is loaded).

    Uses the oracle's cue detection to pick the RC (or abstain) and builds the resolution package
    straight from ground truth — so the demo runs offline and always stays faithful.
    """
    from slm_coach.oracle import detect_rcs

    detected = detect_rcs(complaint)
    if len(detected) == 1:
        rc = next(iter(detected))
        evidence = _extract_evidence(complaint, rc)
        confidence = min(0.82, 0.55 + 0.09 * max(1, len(evidence)))
        alt_rc, alt_why = _DIFFERENTIAL_ALT[rc]
        differential = [
            {"rc": rc, "likelihood": "high", "why": CUE_LIBRARY[rc]["one_line"]},
            {"rc": alt_rc, "likelihood": "low", "why": alt_why},
        ]
        think = _think_for(rc, evidence, confidence)
        resolution = build_resolution(
            rc, confidence=confidence, evidence=evidence, differential=differential
        )
        return Case(leading=rc, complaint=complaint, think=think, resolution=resolution, evidence=evidence)
    # No single distinguishing cue → abstain (reuse the abstention package shape).
    case = build_abstention(random.Random(0))
    case.complaint = complaint
    case.resolution["diagnosis"]["evidence_in_ticket"] = []
    return case
