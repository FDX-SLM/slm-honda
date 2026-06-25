"""Model registry — map the 4 base models to their per-model knobs (spec §6.1/§6.4).

Triết lý model-agnostic (§6.2): **data dùng chung 100%, KHÔNG hardcode template** — mỗi model tự
render chat template qua ``tokenizer.apply_chat_template``. Registry chỉ giữ vài khác biệt per-model
(hf_id, dtype, sampling khuyến nghị, có ``<think>`` native không, ghi chú). Train/eval nhận
``--base`` rồi tra registry để áp các knob này; phần còn lại vẫn config-driven.

KHÔNG thêm Mistral-Small-3 (cố tình ngoài scope — chỉ 4 model dưới).

HF id chỉ là chỉ dẫn — **kiểm tra lại đúng repo trên Hugging Face trước khi chạy** vì tên có thể đổi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Per-model knobs that differ across the 4 base models (everything else is config-driven)."""

    key: str  # short alias (e.g. "qwen")
    hf_id: str  # Hugging Face repo id
    params: str  # parameter count label
    dtype: str  # training/inference compute dtype
    think_native: bool  # does the chat template understand <think> natively?
    sampling: dict[str, Any] = field(default_factory=dict)  # recommended decode (eval/infer)
    notes: str = ""


#: The four in-scope base models (§6.1). Sampling defaults follow §6.4: Qwen/Gemma/Granite use
#: greedy/default decode; only Phi-4-reasoning *requires* temperature=0.8/top_p=0.95/top_k=50 — the
#: plain microsoft/phi-4 here uses default, with the note recorded for the reasoning variant.
_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="qwen",
        hf_id="Qwen/Qwen3.5-9B",
        params="9B",
        dtype="bfloat16",
        think_native=True,
        sampling={"temperature": 0.7, "top_p": 0.9},
        notes="ChatML-like (<|im_start|>…<|im_end|>); <think> native; verify GGUF convert (new arch).",
    ),
    ModelSpec(
        key="gemma",
        hf_id="google/gemma-4-12b-it",
        params="12B",
        dtype="bfloat16",
        think_native=False,
        sampling={"temperature": 0.7, "top_p": 0.9},
        notes="Gemma template (<start_of_turn>…<end_of_turn>); <think> taught via data (literal).",
    ),
    ModelSpec(
        key="phi",
        hf_id="microsoft/phi-4",
        params="14B",
        dtype="bfloat16",
        think_native=True,
        sampling={"temperature": 0.7, "top_p": 0.9},
        notes="ChatML (<|im_start|>role<|im_sep|>); the phi-4-reasoning variant REQUIRES "
        "temperature=0.8, top_p=0.95, top_k=50.",
    ),
    ModelSpec(
        key="granite",
        hf_id="ibm-granite/granite-4.1-8b-instruct",
        params="8B",
        dtype="bfloat16",
        think_native=False,
        sampling={"temperature": 0.7, "top_p": 0.9},
        notes="Granite template (<|start_of_role|>…<|end_of_role|>); <think> taught via data.",
    ),
)

#: Lookup by short alias and by hf_id (both lowercased).
REGISTRY: dict[str, ModelSpec] = {}
for _spec in _SPECS:
    REGISTRY[_spec.key] = _spec
    REGISTRY[_spec.hf_id.lower()] = _spec


def list_models() -> list[ModelSpec]:
    """Return the four base-model specs in registry order."""
    return list(_SPECS)


def resolve(name: str) -> ModelSpec:
    """Resolve a base-model name (alias or hf_id) to its :class:`ModelSpec`.

    Args:
        name: Short alias (``"qwen"``), full hf_id, or a unique substring of one.

    Returns:
        The matching :class:`ModelSpec`.

    Raises:
        KeyError: If no spec matches (with the list of valid keys).
    """
    key = name.strip().lower()
    if key in REGISTRY:
        return REGISTRY[key]
    matches = [s for s in _SPECS if key in s.hf_id.lower() or key in s.key]
    if len(matches) == 1:
        return matches[0]
    valid = ", ".join(s.key for s in _SPECS)
    raise KeyError(f"unknown base model {name!r}; valid: {valid} (or a full HF id)")


def apply_to_config(cfg: Any, name: str) -> ModelSpec:
    """Override a loaded SFT/Align config in place with a base model's knobs.

    Sets ``model_name`` and ``model.dtype`` from the registry (template stays auto via
    ``apply_chat_template``; quant/seq-len remain config-driven). Returns the resolved spec.
    """
    spec = resolve(name)
    cfg.model_name = spec.hf_id
    cfg.model.dtype = spec.dtype
    return spec


def apply_sampling(gen_cfg: Any, name: str) -> ModelSpec:
    """Override a GenerationConfig's sampling with a base model's recommended decode."""
    spec = resolve(name)
    if "temperature" in spec.sampling:
        gen_cfg.temperature = spec.sampling["temperature"]
    if "top_p" in spec.sampling:
        gen_cfg.top_p = spec.sampling["top_p"]
    return spec
