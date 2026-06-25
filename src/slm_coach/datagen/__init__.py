"""Synthetic data generation for the Honda Entitlement Resolver PoC (spec §5).

Đảo ngược "consume-only" của bản sale: ở PoC6, repo **tự sinh** data từ
:mod:`slm_coach.ground_truth`, và mọi mẫu phải qua :mod:`slm_coach.oracle` (rejection sampling)
trước khi ghi. Tất cả nội dung train viết tiếng Anh, định dạng ``messages`` trung lập (model-agnostic).

Submodules:
* :mod:`slm_coach.datagen.core` — complaint synthesis, ``<think>`` trace, resolution package, artifacts.
* :mod:`slm_coach.datagen.sft` — 5 nhóm SFT (§5.1–5.5).
* :mod:`slm_coach.datagen.dpo` — 6 loại cặp DPO (§5.6).
* :mod:`slm_coach.datagen.evalset` — eval (seed 999) + eval_hard.
"""

from __future__ import annotations

from slm_coach.datagen.core import (
    SYSTEM_PROMPT,
    assistant_content,
    build_abstention,
    build_case,
    build_resolution,
)

__all__ = [
    "SYSTEM_PROMPT",
    "assistant_content",
    "build_abstention",
    "build_case",
    "build_resolution",
]
