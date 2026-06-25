r"""Five SFT groups (spec §5.1–5.5), all in ``messages`` chat format.

Phân bổ mục tiêu (§5.0), scale theo ``--limit``:

* Nhóm 1 — complaint→resolution (3 RC) — ca chính.
* Nhóm 2 — knowledge augmentation — nhồi runbook field, 2 chiều.
* Nhóm 3 — differential reasoning traces.
* Nhóm 4 — distractors (same surface, khác cue) — chống "luôn 1 RC".
* Nhóm 5 — abstention (mơ hồ / ngoài catalog).

Mọi record có resolution package đều qua oracle (rejection sampling) trước khi nhận.
"""

from __future__ import annotations

import random
from typing import Any

from slm_coach.datagen.core import SYSTEM_PROMPT, assistant_content, build_abstention, build_case, sft_messages
from slm_coach.datagen.records import sft_record
from slm_coach.ground_truth import (
    RC_TO_RUNBOOK,
    RC_TO_SLICE,
    ROOT_CAUSES,
    RUNBOOKS,
    SEVERITY_NOTES,
    incidents_for,
    median_ttr_min,
)
from slm_coach.oracle import check_resolution

# Target mixture (§5.0) — used to scale group sizes from a single --limit.
DEFAULT_MIX: dict[str, int] = {
    "complaint_resolution": 900,
    "knowledge": 600,
    "differential": 350,
    "distractor": 250,
    "abstention": 200,
}


def _accept(complaint: str, think: str, resolution: dict[str, Any]) -> bool:
    """Oracle gate — only keep samples that pass all five rules (§4)."""
    return check_resolution(complaint, think, resolution).ok


# ---------------------------------------------------------------------------
# Group 1 — complaint → resolution
# ---------------------------------------------------------------------------


def group_complaint_resolution(rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Synthesize ~balanced complaint→resolution records across the 3 RCs."""
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < n:
        rc = ROOT_CAUSES[i % len(ROOT_CAUSES)]
        i += 1
        case = build_case(rng, rc)
        if not _accept(case.complaint, case.think, case.resolution):
            continue
        out.append(
            sft_record(
                f"sft-cr-{len(out):05d}", sft_messages(case), slice_tag=RC_TO_SLICE[rc]
            )
        )
    return out


# ---------------------------------------------------------------------------
# Group 3 — differential reasoning traces (lighter cue, emphasise the trade-off)
# ---------------------------------------------------------------------------


def group_differential(rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Differential traces: read cue → leading hypothesis → down-weight alt → confidence + confirm."""
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < n:
        rc = ROOT_CAUSES[i % len(ROOT_CAUSES)]
        i += 1
        case = build_case(rng, rc, n_cues=2)  # fewer cues → genuine differential
        if not _accept(case.complaint, case.think, case.resolution):
            continue
        out.append(
            sft_record(f"sft-diff-{len(out):05d}", sft_messages(case), slice_tag="differential")
        )
    return out


# ---------------------------------------------------------------------------
# Group 4 — distractors (same surface "paid but unusable" → different RC by cue)
# ---------------------------------------------------------------------------


def group_distractors(rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Strictly cycle the 3 RCs so the RC distribution stays balanced (chống thiên lệch)."""
    out: list[dict[str, Any]] = []
    order = list(ROOT_CAUSES)
    i = 0
    while len(out) < n:
        rc = order[i % 3]
        i += 1
        case = build_case(rng, rc, n_cues=rng.choice([1, 2]) + 1)
        if not _accept(case.complaint, case.think, case.resolution):
            continue
        out.append(
            sft_record(f"sft-dist-{len(out):05d}", sft_messages(case), slice_tag="distractor")
        )
    return out


# ---------------------------------------------------------------------------
# Group 5 — abstention
# ---------------------------------------------------------------------------


def group_abstention(rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Ambiguous / out-of-catalog → differential + route to human, NO fabricated runbook."""
    out: list[dict[str, Any]] = []
    while len(out) < n:
        kind = "out_of_catalog" if len(out) % 2 == 0 else "vague"
        case = build_abstention(rng, kind=kind)
        if not _accept(case.complaint, case.think, case.resolution):
            continue
        out.append(
            sft_record(
                f"sft-abs-{len(out):05d}",
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": case.complaint},
                    {"role": "assistant", "content": assistant_content(case.think, case.resolution)},
                ],
                slice_tag="abstention",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Group 2 — knowledge augmentation (runbook fields, both directions)
# ---------------------------------------------------------------------------

_RC_HUMAN: dict[str, str] = {
    "TCU_OFFLINE": "a TCU offline issue",
    "ENTITLEMENT_CACHE_STALE": "an entitlement cache stale issue",
    "ELIGIBILITY_RULE_CONFLICT": "an eligibility rule conflict",
}

#: Question paraphrases per field — gives ≥8 variants/field across openers (spec §5.2).
_ASK = [
    "{q}",
    "Can you tell me: {ql}",
    "For our runbook, {ql}",
    "Quick question — {ql}",
]


def _bidir(rng: random.Random, q: str, a: str) -> list[tuple[str, str]]:
    """Return question paraphrases (forward) for one (question, answer) fact."""
    ql = q[0].lower() + q[1:]
    return [(tmpl.format(q=q, ql=ql), a) for tmpl in _ASK]


def _field_qa(rc: str) -> list[tuple[str, str]]:
    """Build the (question, answer) facts extracted from one RC's runbook (both directions)."""
    rb = RUNBOOKS[rc]
    human = _RC_HUMAN[rc]
    comp = rb["compensation_policy"]
    proactive = "proactively" if comp["when_proactive"] else "only when it recurs/persists (not proactively)"
    fix = "; ".join(f"{i}. {s}" for i, s in enumerate(rb["fix_steps"], 1))
    cues = "; ".join(rb["detection_cues"])
    confirm = "; ".join(rb["confirm_checks"])
    inc = incidents_for(rc)
    facts: list[tuple[str, str]] = [
        (
            f"What compensation should we offer for {human}?",
            f"{comp['offer'].capitalize()}, offered {proactive}. Escalate if {comp['escalate_if']}.",
        ),
        (
            f"What is the churn risk for {human} and why?",
            f"{rb['churn_risk']['level']} — {rb['churn_risk']['why']}.",
        ),
        (f"How long does {human} typically take to fix?", f"{rb['eta_ttr']}."),
        (
            f"Who owns and supports {rb['runbook_id']}?",
            f"Owner: {rb['owner_team']}. Support: {rb['support_contact']}. "
            f"Escalation: {rb['escalation']}.",
        ),
        (
            f"What severity and priority is {human}?",
            f"Severity {SEVERITY_NOTES[rc]}, priority {rb['priority']}.",
        ),
        (f"What are the resolution steps for {human}?", fix),
        (f"In plain words, why does {human} happen?", rb["why_plain"]),
        (f"Technically, what causes {human}?", rb["why_technical"]),
        (f"What cues in a complaint point to {human}?", f"Use this runbook when: {cues}."),
        (f"What should I confirm before acting on {human}?", f"Confirm: {confirm}."),
        (
            f"Which runbook covers {human} and what is its id?",
            f"{rb['runbook_id']} — {rb['title']}.",
        ),
        (
            f"How should I communicate with the customer for {human}?",
            rb["customer_communication"],
        ),
    ]
    if inc:
        facts.append(
            (
                f"Give me a similar past incident for {human}.",
                f"{inc[0]['id']}: {inc[0]['customer_complaint']} "
                f"(key cue: {inc[0]['key_cue']}; TTR ~{inc[0]['ttr_min']} min).",
            )
        )
        facts.append(
            (
                f"What is the median time-to-resolution seen for {human}?",
                f"Median ~{median_ttr_min(rc)} minutes across {len(inc)} past incidents.",
            )
        )
    # Reverse-direction facts (combat the reversal curse, §5.2).
    facts.append(
        (
            f"Which root cause maps to runbook {rb['runbook_id']}?",
            f"{rc} — {rb['one_line']}",
        )
    )
    facts.append(
        (
            f"Which runbook id should I open for {rc}?",
            f"{RC_TO_RUNBOOK[rc]} ({rb['title']}).",
        )
    )
    return facts


def group_knowledge(rng: random.Random, n: int) -> list[dict[str, Any]]:
    """Generate runbook-field Q&A across all 3 RCs (plain user/assistant, no system turn)."""
    pairs: list[tuple[str, str]] = []
    for rc in ROOT_CAUSES:
        for q, a in _field_qa(rc):
            pairs.extend(_bidir(rng, q, a))
    rng.shuffle(pairs)
    out: list[dict[str, Any]] = []
    i = 0
    while len(out) < n:
        q, a = pairs[i % len(pairs)]
        i += 1
        out.append(
            sft_record(
                f"sft-kn-{len(out):05d}",
                [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                slice_tag="knowledge",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_sft(
    seed: int, *, limit: int | None = None, source: str = "mixed"
) -> list[dict[str, Any]]:
    """Generate the full SFT set across the 5 groups, scaled by ``limit`` (None = full mix).

    Args:
        seed: RNG seed (spec quickstart uses 42).
        limit: Optional cap on TOTAL records; group sizes scale proportionally to :data:`DEFAULT_MIX`.
        source: Where the complaint/abstention LANGUAGE comes from —
            ``"authored"`` = only Claude-authored seeds (richest, smaller) + template knowledge Q&A;
            ``"template"`` = only the rule-based generator (largest, least diverse);
            ``"mixed"`` (default) = Claude-authored seeds as the rich core, template top-up for volume.

    Returns:
        A shuffled list of SFT records (every resolution sample passed the oracle).
    """
    rng = random.Random(seed)
    total_default = sum(DEFAULT_MIX.values())
    if limit is None:
        sizes = dict(DEFAULT_MIX)
    else:
        scale = limit / total_default
        sizes = {k: max(1, round(v * scale)) for k, v in DEFAULT_MIX.items()}

    authored: list[dict[str, Any]] = []
    if source in ("authored", "mixed"):
        from slm_coach.datagen.authored import generate_authored

        authored = generate_authored()

    if source == "authored":
        records = list(authored)
        records += group_knowledge(rng, sizes["knowledge"])  # runbook-field Q&A (factual)
        rng.shuffle(records)
        return records

    # Count how many of each group the authored seeds already cover (RC slices → complaint_resolution).
    n_auth_cr = sum(1 for r in authored if r["mode"] in ("tcu_offline", "cache_stale", "eligibility"))
    n_auth_abs = sum(1 for r in authored if r["mode"] == "abstention")

    records: list[dict[str, Any]] = list(authored)
    records += group_complaint_resolution(rng, max(0, sizes["complaint_resolution"] - n_auth_cr))
    records += group_knowledge(rng, sizes["knowledge"])
    records += group_differential(rng, sizes["differential"])
    records += group_distractors(rng, sizes["distractor"])
    records += group_abstention(rng, max(0, sizes["abstention"] - n_auth_abs))
    rng.shuffle(records)
    return records
