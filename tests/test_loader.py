"""Tests for reading, validating, filtering, and splitting the data directory."""

from __future__ import annotations

import json

from slm_coach.data.loader import load_records, load_split, validate_data_dir


def _sft(rec_id: str, audit: str) -> dict:
    return {
        "id": rec_id,
        "data_type": "sft",
        "conversation_type": "single",
        "mode": "comparison",
        "persona": "P01",
        "messages": [
            {"role": "user", "content": "iPhone nào tốt?"},
            {"role": "assistant", "content": "Tùy nhu cầu của anh/chị."},
        ],
        "lang": "vi",
        "version": "v1",
        "audit_status": audit,
    }


def _write_sft_dir(tmp_path):
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    lines = [
        json.dumps(_sft("a1", "approved")),
        json.dumps(_sft("a2", "approved")),
        json.dumps(_sft("p1", "pending")),  # valid but filtered out
        '{"id": "bad", "data_type": "sft"}',  # invalid (missing required fields)
    ]
    (sft_dir / "part.jsonl").write_text("\n".join(lines), encoding="utf-8")


def test_load_split_filters_audit_status(tmp_path):
    _write_sft_dir(tmp_path)
    split = load_split(tmp_path, "sft", keep_audit_status=["approved"])

    assert len(split.records) == 2
    assert {r.id for r in split.records} == {"a1", "a2"}
    assert split.filtered_out == 1  # the pending record
    assert split.stats.valid == 3
    assert split.stats.invalid == 1


def test_load_records_splits_by_type(tmp_path):
    _write_sft_dir(tmp_path)
    loaded = load_records(tmp_path, data_types=["sft", "reasoning"], keep_audit_status=["approved"])

    assert loaded.count("sft") == 2
    assert loaded.count("reasoning") == 0  # directory absent -> empty split
    assert loaded.total == 2


def test_load_split_without_filter_keeps_all_valid(tmp_path):
    _write_sft_dir(tmp_path)
    split = load_split(tmp_path, "sft", keep_audit_status=None)
    assert len(split.records) == 3  # approved + pending, invalid excluded
    assert split.filtered_out == 0


def test_validate_data_dir(tmp_path):
    _write_sft_dir(tmp_path)
    stats = validate_data_dir(tmp_path, data_types=["sft"])
    assert stats["sft"].valid == 3
    assert stats["sft"].invalid == 1
    assert stats["sft"].by_mode["comparison"] == 3
