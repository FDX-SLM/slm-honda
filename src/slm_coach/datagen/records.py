"""Record builders: wrap synthesized content in the canonical JSONL schema (English data).

Mỗi record mang metadata chung (``id, data_type, mode, persona, lang, version, audit_status``).
``mode`` ở đây là **slice tag** (tcu_offline / cache_stale / eligibility / abstention / knowledge /
differential / distractor) — metadata để per-slice eval + holdout, KHÔNG bao giờ vào sequence train.
"""

from __future__ import annotations

from typing import Any

PERSONA = "internal_agent"
LANG = "en"
VERSION = "poc6-v1"
AUDIT = "approved"


def sft_record(rid: str, messages: list[dict[str, str]], *, slice_tag: str) -> dict[str, Any]:
    """Build a single-turn SFT record from rendered messages."""
    return {
        "id": rid,
        "data_type": "sft",
        "conversation_type": "single",
        "mode": slice_tag,
        "persona": PERSONA,
        "lang": LANG,
        "version": VERSION,
        "audit_status": AUDIT,
        "messages": messages,
    }


def pref_record(
    rid: str,
    prompt: list[dict[str, str]],
    chosen_text: str,
    rejected_text: str,
    *,
    slice_tag: str,
    bad_type: str,
) -> dict[str, Any]:
    """Build a DPO preference record (explicit prompt + chosen/rejected assistant turns)."""
    return {
        "id": rid,
        "data_type": "preference",
        "mode": slice_tag,
        "persona": PERSONA,
        "bad_type": bad_type,
        "lang": LANG,
        "version": VERSION,
        "audit_status": AUDIT,
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": chosen_text}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
    }


def gold_record(
    rid: str,
    complaint: str,
    reference: str,
    *,
    slice_tag: str,
    leading_root_cause: str,
) -> dict[str, Any]:
    """Build a gold/eval record (prompt + reference + the expected leading root cause)."""
    return {
        "id": rid,
        "mode": slice_tag,
        "leading_root_cause": leading_root_cause,
        "messages": [{"role": "user", "content": complaint}],
        "reference": reference,
        "persona": PERSONA,
    }
