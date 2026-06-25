"""Tests for the Honda Entitlement Resolver: ground truth, oracle, datagen, eval, RAG."""

from __future__ import annotations

import random

from slm_coach.datagen.core import answer_for_complaint, assistant_content, build_abstention, build_case
from slm_coach.datagen.dpo import generate_dpo
from slm_coach.datagen.evalset import EVAL_HARD, generate_eval
from slm_coach.datagen.sft import generate_sft
from slm_coach.eval.honda import aggregate, score_sample
from slm_coach.eval.rag import RagBaseline
from slm_coach.ground_truth import ABSTAIN, ROOT_CAUSES, RUNBOOKS, render_runbook
from slm_coach.oracle import (
    check_resolution,
    detect_rcs,
    find_fabricated_telemetry,
    is_grounded,
    parse_output,
)

# --- ground truth -----------------------------------------------------------------------------


def test_runbooks_cover_three_rcs():
    assert set(RUNBOOKS) == set(ROOT_CAUSES)
    for rc in ROOT_CAUSES:
        doc = render_runbook(rc)
        assert RUNBOOKS[rc]["runbook_id"] in doc
        assert "Resolution steps" in doc and "Compensation" not in doc[:50]


# --- oracle -----------------------------------------------------------------------------------


def test_oracle_flags_fabricated_telemetry():
    assert find_fabricated_telemetry("webhook delivered at T+28s")
    assert find_fabricated_telemetry("the record was not found")
    assert not find_fabricated_telemetry("I cannot see the TCU last_seen from the ticket")


def test_oracle_grounding():
    complaint = "My car is parked in an underground garage and it times out."
    assert is_grounded("parked in underground garage", complaint)
    assert not is_grounded("the webhook was delivered at T+28s", complaint)


def test_detect_rcs_single_cue():
    assert detect_rcs("parked in an underground garage, remote start times out") == {"TCU_OFFLINE"}
    # "subscribed" must NOT trigger eligibility on its own.
    assert "ELIGIBILITY_RULE_CONFLICT" not in detect_rcs("I subscribed to remote climate")


# --- datagen passes the oracle by construction ------------------------------------------------


def test_generated_cases_pass_oracle():
    rng = random.Random(1)
    for _ in range(50):
        for rc in ROOT_CAUSES:
            c = build_case(rng, rc)
            assert check_resolution(c.complaint, c.think, c.resolution).ok
    for kind in ("vague", "out_of_catalog"):
        for _ in range(20):
            c = build_abstention(rng, kind=kind)
            assert check_resolution(c.complaint, c.think, c.resolution).ok


def test_generate_sft_balanced_and_valid():
    records = generate_sft(42, limit=140)
    modes = {r["mode"] for r in records}
    assert {"tcu_offline", "cache_stale", "eligibility", "abstention", "knowledge"} <= modes
    # every complaint→resolution record's assistant turn parses + passes the oracle
    for r in records:
        msgs = r["messages"]
        if msgs[0]["role"] == "system":
            think, res = parse_output(msgs[-1]["content"])
            assert res is not None
            assert check_resolution(msgs[1]["content"], think, res).ok


def test_generate_dpo_six_types():
    records = generate_dpo(42, limit=60)
    assert {r["bad_type"] for r in records} == {
        "cue_dropped", "fabricated_telemetry", "overconfident",
        "missing_fields", "forced_guess", "overpromise",
    }
    # chosen passes the oracle (rejected need not).
    for r in records:
        comp = r["prompt"][1]["content"]
        think, res = parse_output(r["chosen"][0]["content"])
        assert res is not None and check_resolution(comp, think, res).ok


# --- eval metrics -----------------------------------------------------------------------------


def test_eval_perfect_replay_scores_high():
    ev = generate_eval(999, limit=40)
    cases = [
        {"id": r["id"], "slice": r["mode"], "complaint": r["messages"][0]["content"],
         "gold_rc": r["leading_root_cause"], "reference": r["reference"]}
        for r in ev
    ]
    samples = [score_sample(c, c["reference"]) for c in cases]
    rep = aggregate(samples)
    assert rep.rc_accuracy_clear == 1.0
    assert rep.no_fabrication_rate == 1.0
    assert rep.abstention_hallucination == 0.0
    assert rep.cue_faithfulness >= 0.95


def test_eval_detects_fabrication_and_wrong_rc():
    bad = (
        "<think>webhook delivered at T+28s, record not found.</think>"
        '{"diagnosis":{"leading_root_cause":"TCU_OFFLINE","confidence":0.97,'
        '"differential":[],"evidence_in_ticket":["webhook delivered at T+28s"],"to_confirm":[]},'
        '"runbook_id":"RB-TCU-04","owner_team":"x","support_contact":"x","escalation":"x",'
        '"fix_steps":["a"],"eta_ttr":"x","severity":"S4","priority":"P3",'
        '"churn_risk":{"level":"low"},"compensation":{"offer":"x"},'
        '"artifacts":{"rca_md":"","work_order_md":"","customer_email":"","diagram_mermaid":""}}'
    )
    case = {"id": "c", "slice": "cache_stale", "complaint": "app shows active on web not app",
            "gold_rc": "ENTITLEMENT_CACHE_STALE", "reference": ""}
    s = score_sample(case, bad)
    assert not s.correct
    assert not s.no_fabricated_telemetry
    assert not s.artifacts_valid


# --- RAG baseline foil ------------------------------------------------------------------------


def test_rag_never_abstains_so_misses_ambiguous():
    rag = RagBaseline()
    pred = rag.predict("It just doesn't work. I paid and nothing happens.")
    # RAG copies the nearest ticket → it can never return INSUFFICIENT_EVIDENCE.
    assert pred["leading_root_cause"] in ROOT_CAUSES
    assert pred["leading_root_cause"] != ABSTAIN


def test_demo_fallback_matches_cues():
    c = answer_for_complaint(
        "Parked in an underground garage all week, remote start spins and times out, shows active."
    )
    assert c.leading == "TCU_OFFLINE"
    think, res = parse_output(assistant_content(c.think, c.resolution))
    assert check_resolution(c.complaint, think, res).ok
