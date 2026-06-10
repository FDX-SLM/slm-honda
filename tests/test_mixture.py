"""Tests for single/multi-turn mixing and curriculum staging."""

from __future__ import annotations

from slm_coach.data.mixture import (
    build_curriculum,
    is_multi_turn,
    mix_single_multi,
    split_single_multi,
)
from slm_coach.data.schema import parse_record


def _sft(rec_id: str, *, multi: bool) -> object:
    messages = [{"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"}]
    if multi:
        messages += [{"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"}]
    return parse_record(
        {
            "id": rec_id,
            "data_type": "sft",
            "conversation_type": "multi_turn" if multi else "single",
            "mode": "comparison",
            "persona": "P01",
            "messages": messages,
            "lang": "vi",
            "version": "v1",
            "audit_status": "approved",
        }
    )


def _reasoning(rec_id: str) -> object:
    return parse_record(
        {
            "id": rec_id,
            "data_type": "reasoning",
            "mode": "objection_handling",
            "persona": "P02",
            "situation": "S",
            "reasoning": ["a"],
            "response": "R",
            "lang": "vi",
            "version": "v1",
            "audit_status": "approved",
        }
    )


def test_split_single_multi():
    records = [_sft("m1", multi=True), _sft("s1", multi=False), _sft("m2", multi=True)]
    single, multi = split_single_multi(records)
    assert {r.id for r in single} == {"s1"}
    assert {r.id for r in multi} == {"m1", "m2"}


def test_mix_single_multi_hits_ratio():
    records = [_sft(f"m{i}", multi=True) for i in range(6)]
    records += [_sft(f"s{i}", multi=False) for i in range(6)]

    mixed = mix_single_multi(records, multi_turn=0.66, single=0.34, seed=1)

    n_multi = sum(1 for r in mixed if is_multi_turn(r))
    n_single = len(mixed) - n_multi
    assert len(mixed) == 9
    assert n_multi == 6
    assert n_single == 3


def test_mix_is_deterministic_with_seed():
    records = [_sft(f"m{i}", multi=True) for i in range(6)]
    records += [_sft(f"s{i}", multi=False) for i in range(6)]
    first = [r.id for r in mix_single_multi(records, multi_turn=0.66, single=0.34, seed=7)]
    second = [r.id for r in mix_single_multi(records, multi_turn=0.66, single=0.34, seed=7)]
    assert first == second


def test_build_curriculum_orders_and_selects():
    sft_records = [_sft(f"m{i}", multi=True) for i in range(4)]
    sft_records += [_sft(f"s{i}", multi=False) for i in range(4)]
    records_by_type = {"sft": sft_records, "reasoning": [_reasoning("r1"), _reasoning("r2")]}

    specs = [
        {"name": "broad", "include": ["sft"], "mix": {"multi_turn": 0.66, "single": 0.34}},
        {"name": "reasoning", "include": ["sft", "reasoning"], "reasoning_thinking": True},
    ]
    stages = build_curriculum(specs, records_by_type, seed=3)

    assert [s.name for s in stages] == ["broad", "reasoning"]
    assert all(r.data_type == "sft" for r in stages[0].records)
    assert stages[0].reasoning_thinking is False
    assert stages[1].reasoning_thinking is True
    assert any(r.data_type == "reasoning" for r in stages[1].records)
