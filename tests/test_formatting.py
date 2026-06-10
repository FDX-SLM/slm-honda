"""Tests for canonical -> TRL formatting and multi-turn masking."""

from __future__ import annotations

from slm_coach.data.formatting import (
    fold_reasoning,
    iter_assistant_spans,
    preference_to_example,
    reasoning_to_messages,
    record_to_messages,
    render_chatml,
    sft_to_messages,
    to_preference_dataset,
    to_sft_dataset,
)
from slm_coach.data.schema import parse_record


def _multi_turn_sft() -> object:
    return parse_record(
        {
            "id": "s1",
            "data_type": "sft",
            "conversation_type": "multi_turn",
            "mode": "comparison",
            "persona": "P01",
            "messages": [
                {"role": "system", "content": "SYS"},
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
            ],
            "lang": "vi",
            "version": "v1",
            "audit_status": "approved",
        }
    )


def _reasoning() -> object:
    return parse_record(
        {
            "id": "r1",
            "data_type": "reasoning",
            "mode": "objection_handling",
            "persona": "P02",
            "situation": "Khách chê đắt.",
            "reasoning": ["step-A", "step-B"],
            "response": "Câu trả lời cuối.",
            "why": "SECRET_AUDIT_ONLY",
            "lang": "vi",
            "version": "v1",
            "audit_status": "approved",
        }
    )


def test_multiturn_masking_marks_every_assistant_turn():
    messages = sft_to_messages(_multi_turn_sft())
    text = render_chatml(messages)
    spans = iter_assistant_spans(messages)

    # One trainable span per assistant turn (not just the last).
    assert len(spans) == 2
    assert [text[s:e] for s, e in spans] == ["A1", "A2"]

    # User/system content is excluded from the trainable spans.
    for needle in ("SYS", "Q1", "Q2"):
        start = text.index(needle)
        assert not any(s <= start < e for s, e in spans)


def test_reasoning_thinking_fold_and_why_excluded():
    with_think = reasoning_to_messages(_reasoning(), thinking=True)
    assistant = with_think[-1]["content"]
    assert assistant.startswith("<think>\nstep-A\nstep-B\n</think>\n")
    assert assistant.endswith("Câu trả lời cuối.")
    assert "SECRET_AUDIT_ONLY" not in assistant  # `why` is never trained

    without_think = reasoning_to_messages(_reasoning(), thinking=False)
    assert without_think[-1]["content"] == "Câu trả lời cuối."
    assert "<think>" not in without_think[-1]["content"]


def test_fold_reasoning_helper():
    assert fold_reasoning(["a", "b"], "resp", thinking=False) == "resp"
    assert fold_reasoning(["a", "b"], "resp", thinking=True) == "<think>\na\nb\n</think>\nresp"


def test_messages_strip_metadata():
    messages = record_to_messages(_multi_turn_sft())
    for message in messages:
        assert set(message.keys()) == {"role", "content"}  # no mode/persona leakage


def test_to_sft_dataset_columns():
    ds = to_sft_dataset([_multi_turn_sft()])
    assert ds.column_names == ["messages", "text"]
    assert len(ds) == 1
    assert "<|im_start|>" in ds[0]["text"]


def test_to_preference_dataset():
    record = parse_record(
        {
            "id": "p1",
            "data_type": "preference",
            "mode": "purchase_intent",
            "persona": "P03",
            "prompt": [{"role": "user", "content": "Giảm giá không?"}],
            "chosen": [{"role": "assistant", "content": "Ưu đãi trả góp 0%."}],
            "rejected": [{"role": "assistant", "content": "Mua ngay đi!"}],
            "lang": "vi",
            "version": "v1",
            "audit_status": "approved",
        }
    )
    example = preference_to_example(record)
    assert example["chosen"] == "Ưu đãi trả góp 0%."
    assert example["rejected"] == "Mua ngay đi!"
    assert example["prompt"] == [{"role": "user", "content": "Giảm giá không?"}]

    ds = to_preference_dataset([record])
    assert set(ds.column_names) == {"prompt", "chosen", "rejected"}
