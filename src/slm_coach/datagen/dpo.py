r"""Six DPO pair types (spec §5.6) — chống đúng các lỗi nguy hiểm.

Mỗi cặp ``{prompt, chosen, rejected}``: ``chosen`` là câu trả lời tốt (qua oracle), ``rejected``
cố tình mắc ĐÚNG một lỗi để DPO đẩy model tránh nó. ``rejected`` KHÔNG cần qua oracle (nó là mẫu
xấu cố ý). 6 loại:

1. ``cue_dropped`` — đọc đúng cue (garage→TCU) vs bỏ cue → phán CACHE.
2. ``fabricated_telemetry`` — calibrated vs BỊA telemetry ("webhook delivered at T+28s, record not
   found"). **Cặp quan trọng nhất.**
3. ``overconfident`` — confidence 0.7 + differential vs 0.97 phán chắc nịch.
4. ``missing_fields`` — đủ business field vs bỏ churn_risk + compensation.
5. ``forced_guess`` — abstain trên ca mơ hồ vs ép một RC cụ thể.
6. ``overpromise`` — bồi thường đúng policy vs hứa "hoàn tiền + 1 năm free".
"""

from __future__ import annotations

import copy
import random
from typing import Any

from slm_coach.datagen.core import (
    SYSTEM_PROMPT,
    assistant_content,
    build_abstention,
    build_case,
    build_resolution,
)
from slm_coach.datagen.records import pref_record
from slm_coach.ground_truth import RC_TO_SLICE, ROOT_CAUSES

DPO_TYPES: tuple[str, ...] = (
    "cue_dropped",
    "fabricated_telemetry",
    "overconfident",
    "missing_fields",
    "forced_guess",
    "overpromise",
)
PER_TYPE_DEFAULT = 100


def _prompt(complaint: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": complaint}]


def _wrong_rc(rc: str, rng: random.Random) -> str:
    return rng.choice([r for r in ROOT_CAUSES if r != rc])


Pair = tuple[list[dict[str, str]], str, str, str]  # (prompt, chosen, rejected, slice_tag)


def _pair_cue_dropped(rng: random.Random) -> Pair:
    case = build_case(rng, "TCU_OFFLINE")
    wrong = "ENTITLEMENT_CACHE_STALE"
    rej_res = build_resolution(
        wrong,
        confidence=0.74,
        evidence=["the app is showing the feature off"],
        differential=[{"rc": wrong, "likelihood": "high", "why": "assumed a cache glitch"}],
    )
    rej_think = (
        "The app shows the feature off, so this is probably just a stale cache; I'll go with "
        "cache stale and skip the rest of the ticket."
    )
    return (
        _prompt(case.complaint),
        assistant_content(case.think, case.resolution),
        assistant_content(rej_think, rej_res),
        RC_TO_SLICE["TCU_OFFLINE"],
    )


def _pair_fabricated(rng: random.Random) -> Pair:
    rc = rng.choice(ROOT_CAUSES)
    case = build_case(rng, rc)
    rej_res = copy.deepcopy(case.resolution)
    rej_res["diagnosis"]["confidence"] = 0.95
    rej_think = (
        "The webhook was delivered at T+28s and the entitlement record was not found, and the TCU "
        f"last_seen was 14:32, so I am certain this is {rc}."
    )
    return (
        _prompt(case.complaint),
        assistant_content(case.think, case.resolution),
        assistant_content(rej_think, rej_res),
        RC_TO_SLICE[rc],
    )


def _pair_overconfident(rng: random.Random) -> Pair:
    rc = rng.choice(ROOT_CAUSES)
    case = build_case(rng, rc)
    rej_res = copy.deepcopy(case.resolution)
    rej_res["diagnosis"]["confidence"] = 0.97
    rej_res["diagnosis"]["differential"] = [
        {"rc": rc, "likelihood": "certain", "why": "obviously this"}
    ]
    rej_think = f"It's clearly {rc}. No need to consider alternatives. Confidence 0.97."
    return (
        _prompt(case.complaint),
        assistant_content(case.think, case.resolution),
        assistant_content(rej_think, rej_res),
        RC_TO_SLICE[rc],
    )


def _pair_missing_fields(rng: random.Random) -> Pair:
    rc = rng.choice(ROOT_CAUSES)
    case = build_case(rng, rc)
    rej_res = copy.deepcopy(case.resolution)
    rej_res.pop("churn_risk", None)
    rej_res.pop("compensation", None)
    return (
        _prompt(case.complaint),
        assistant_content(case.think, case.resolution),
        assistant_content(case.think, rej_res),
        RC_TO_SLICE[rc],
    )


def _pair_forced_guess(rng: random.Random) -> Pair:
    case = build_abstention(rng, kind=rng.choice(["vague", "out_of_catalog"]))
    rc = rng.choice(ROOT_CAUSES)
    rej_res = build_resolution(
        rc,
        confidence=0.88,
        evidence=[],
        differential=[{"rc": rc, "likelihood": "high", "why": "guessed without a cue"}],
    )
    rej_think = f"There's not much detail, but I'll just guess {rc} and move on."
    return (
        _prompt(case.complaint),
        assistant_content(case.think, case.resolution),
        assistant_content(rej_think, rej_res),
        "abstention",
    )


def _pair_overpromise(rng: random.Random) -> Pair:
    case = build_case(rng, "ELIGIBILITY_RULE_CONFLICT")
    rej_res = copy.deepcopy(case.resolution)
    rej_res["compensation"] = {
        "offer": "full refund plus one year of free service",
        "proactive": True,
        "note": "promised the customer a full refund and a year free",
    }
    rej_res["artifacts"]["customer_email"] = (
        "Hi, so sorry! To make it up to you we'll give you a full refund AND one year of the "
        "service completely free. Consider it done!"
    )
    return (
        _prompt(case.complaint),
        assistant_content(case.think, case.resolution),
        assistant_content(case.think, rej_res),
        RC_TO_SLICE["ELIGIBILITY_RULE_CONFLICT"],
    )


_BUILDERS = {
    "cue_dropped": _pair_cue_dropped,
    "fabricated_telemetry": _pair_fabricated,
    "overconfident": _pair_overconfident,
    "missing_fields": _pair_missing_fields,
    "forced_guess": _pair_forced_guess,
    "overpromise": _pair_overpromise,
}


def generate_dpo(seed: int, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Generate DPO preference pairs, balanced across the 6 types (§5.6).

    Args:
        seed: RNG seed (spec quickstart uses 42).
        limit: Optional cap on TOTAL pairs; split evenly across the 6 types.

    Returns:
        A shuffled list of preference records.
    """
    rng = random.Random(seed)
    per_type = PER_TYPE_DEFAULT if limit is None else max(1, round(limit / len(DPO_TYPES)))
    records: list[dict[str, Any]] = []
    for dtype in DPO_TYPES:
        builder = _BUILDERS[dtype]
        for _ in range(per_type):
            prompt, chosen, rejected, slice_tag = builder(rng)
            records.append(
                pref_record(
                    f"dpo-{dtype}-{len(records):05d}",
                    prompt,
                    chosen,
                    rejected,
                    slice_tag=slice_tag,
                    bad_type=dtype,
                )
            )
    rng.shuffle(records)
    return records
