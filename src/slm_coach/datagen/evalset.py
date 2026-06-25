r"""Evaluation sets (spec §7): ``eval.jsonl`` (seed 999) + ``eval_hard.jsonl`` (20 hand-written).

Gold record giữ ``leading_root_cause`` mong đợi + ``reference`` (assistant content gold) để chấm
RC-accuracy, runbook-completeness, calibration. eval_hard là lời than thật, lộn xộn, cue ẩn —
seed khác hẳn data train, báo cáo riêng.
"""

from __future__ import annotations

import random
from typing import Any

from slm_coach.datagen.core import assistant_content, build_abstention, build_case
from slm_coach.datagen.records import gold_record
from slm_coach.ground_truth import ABSTAIN, RC_TO_SLICE, ROOT_CAUSES


def generate_eval(seed: int = 999, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Generate a balanced eval set (3 RC + abstention) with a held-out seed (default 999)."""
    rng = random.Random(seed)
    n = limit or 180
    out: list[dict[str, Any]] = []
    # 3 RC + abstention, roughly even.
    plan = [*ROOT_CAUSES, ABSTAIN]
    i = 0
    while len(out) < n:
        label = plan[i % len(plan)]
        i += 1
        if label == ABSTAIN:
            case = build_abstention(rng, kind="vague" if i % 2 else "out_of_catalog")
        else:
            case = build_case(rng, label)
        ref = assistant_content(case.think, case.resolution)
        out.append(
            gold_record(
                f"eval-{len(out):05d}",
                case.complaint,
                ref,
                slice_tag=RC_TO_SLICE[label],
                leading_root_cause=label,
            )
        )
    return out


# ---------------------------------------------------------------------------
# eval_hard — 20 hand-written messy complaints with hidden cues
# ---------------------------------------------------------------------------

#: (complaint, expected_leading_root_cause). Messy, real-sounding, cue buried in noise.
EVAL_HARD: list[tuple[str, str]] = [
    (
        "ok so im super frustrated. paid for remote start last fri, works on the website i checked, "
        "but my phone app just wont show it. tried reinstalling. weird right",
        "ENTITLEMENT_CACHE_STALE",
    ),
    (
        "remote climate spins forever then says timed out. fwiw the car lives in our condo's level "
        "B3 parking, barely any bars down there. subscription is paid up",
        "TCU_OFFLINE",
    ),
    (
        "i bought the touring tier for my crv (its a 2025, im out in seattle) and the app STILL "
        "nags me to subscribe every time i open it. i already paid you guys??",
        "ELIGIBILITY_RULE_CONFLICT",
    ),
    (
        "nothing works. i paid. please just fix it. i dont know what else to tell you",
        "INSUFFICIENT_EVIDENCE",
    ),
    (
        "keep getting kicked out, says error 403 whenever i tap the remote features. logged in fine "
        "otherwise",
        "INSUFFICIENT_EVIDENCE",
    ),
    (
        "had it working tuesday, gone wednesday, back thursday after i logged out and in. its on and "
        "off, drives me nuts",
        "ENTITLEMENT_CACHE_STALE",
    ),
    (
        "car's been sitting in the airport garage since i flew out a week ago. now remote lock wont "
        "respond at all, just hangs. it says active in the app though",
        "TCU_OFFLINE",
    ),
    (
        "so i have a civic, canada, and i went for the touring plan. money came out of my account. "
        "app acts like i never subscribed. been days",
        "ELIGIBILITY_RULE_CONFLICT",
    ),
    (
        "my subscription is broken somehow, can you look into it? account number is on file",
        "INSUFFICIENT_EVIDENCE",
    ),
    (
        "the feature shows up green/active on the web portal but the hondalink app says i need to "
        "buy it. confusing. paid 2 days ago",
        "ENTITLEMENT_CACHE_STALE",
    ),
    (
        "vehicle finder times out every single time. truck hasnt moved from the underground lot all "
        "week. signal is probably trash there",
        "TCU_OFFLINE",
    ),
    (
        "i want a refund honestly, this whole thing is a mess and ive been charged twice maybe",
        "INSUFFICIENT_EVIDENCE",
    ),
    (
        "remote start = paid, confirmed on my receipt. app keeps asking me to subscribe again. its a "
        "premium plan on a limited trim if that matters",
        "ELIGIBILITY_RULE_CONFLICT",
    ),
    (
        "app crashes the second i open the remote tab. cant even get to the feature",
        "INSUFFICIENT_EVIDENCE",
    ),
    (
        "everything looked fine then the climate control feature just vanished from the app. comes "
        "back sometimes. account looks normal to me",
        "ENTITLEMENT_CACHE_STALE",
    ),
    (
        "parked in a basement garage for my whole work trip, ~8 days. commands just spin. subscription "
        "active per the app",
        "TCU_OFFLINE",
    ),
    (
        "idk man it just doesnt do anything when i press the buttons",
        "INSUFFICIENT_EVIDENCE",
    ),
    (
        "paid for elite on my cr-v out west (oregon) wait no i think i picked touring. anyway it wont "
        "activate, keeps prompting purchase",
        "ELIGIBILITY_RULE_CONFLICT",
    ),
    (
        "the web dashboard clearly shows my plan active. the app on my pixel does not. logging in "
        "again helps for like a minute",
        "ENTITLEMENT_CACHE_STALE",
    ),
    (
        "remote engine start has been timing out for days. car is garaged underground at work and i "
        "havent driven it. shows active though so i'm confused",
        "TCU_OFFLINE",
    ),
]


def generate_eval_hard() -> list[dict[str, Any]]:
    """Materialize the 20 hand-written hard cases as gold records (reference left empty)."""
    out: list[dict[str, Any]] = []
    for complaint, label in EVAL_HARD:
        slice_tag = RC_TO_SLICE.get(label, "abstention")
        out.append(
            gold_record(
                f"hard-{len(out):05d}",
                complaint,
                "",  # hand-written: no gold reference text, RC label is the ground truth
                slice_tag=slice_tag,
                leading_root_cause=label,
            )
        )
    return out
