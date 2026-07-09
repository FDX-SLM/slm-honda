"""SLM service — sinh output closed-book cho một lời than.

Thứ tự ưu tiên:
  1) vLLM server (OpenAI-compatible) nếu chạy → NHANH (~3x, resident). Không nạp model HF (nhường GPU).
  2) HF transformers (base + adapter DPO) nếu không có vLLM và có GPU.
  3) Ground truth (answer_for_complaint) nếu thiếu GPU/adapter — vẫn đúng schema, demo không vỡ.

vLLM chạy ở venv cô lập /workspace/vllm-venv (build cu129 khớp driver). Backend chỉ gọi HTTP tới nó.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from threading import Lock

import httpx

from slm_coach.datagen.core import answer_for_complaint, assistant_content
from slm_coach.ground_truth import SYSTEM_PROMPT

BASE = os.environ.get("HONDA_BASE", "Qwen/Qwen3.5-9B")
ADAPTER = os.environ.get("HONDA_ADAPTER", "checkpoints/sft/best")
FOUR_BIT = os.environ.get("HONDA_4BIT", "true").strip().lower() not in ("false", "0", "no")
MAX_NEW_TOKENS = int(os.environ.get("HONDA_MAX_NEW_TOKENS", "1700"))

# vLLM OpenAI-compatible server (~3x). Tự dùng nếu server 8800 đang chạy, không thì rớt về HF.
# vLLM serve checkpoint ĐÃ MERGE FULL (/workspace/honda-merged-full) qua path multimodal — xem run_vllm.sh.
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8800/v1").strip()
VLLM_MODEL = os.environ.get("VLLM_MODEL", "dpo_qwen").strip() or "dpo_qwen"
VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT_S", "180"))

_LOCK = Lock()
_LOADED: tuple[object, object] | None = None
_LOAD_TRIED = False
_LOAD_ERROR: str | None = None


# --------------------------------------------------------------------------- #
# vLLM path
# --------------------------------------------------------------------------- #
def _vllm_up() -> bool:
    """VLLM server có sẵn sàng (đã nạp model) không? Kiểm nhanh /models."""
    if not VLLM_URL:
        return False
    try:
        r = httpx.get(f"{VLLM_URL}/models", timeout=2.0)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _run_vllm(complaint: str) -> tuple[str, int]:
    """Gọi vLLM /chat/completions (áp LoRA dpo_qwen); vLLM tự render chat template."""
    t0 = time.perf_counter()
    r = httpx.post(
        f"{VLLM_URL}/chat/completions",
        json={
            "model": VLLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": complaint},
            ],
            "temperature": 0,
            "max_tokens": MAX_NEW_TOKENS,
        },
        timeout=VLLM_TIMEOUT,
    )
    r.raise_for_status()
    text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    return text, int((time.perf_counter() - t0) * 1000)


# --------------------------------------------------------------------------- #
# HF transformers path (fallback khi không có vLLM)
# --------------------------------------------------------------------------- #
def _load() -> tuple[object, object] | None:
    """Nạp (lười, 1 lần) base + adapter HF. None nếu thiếu GPU/torch/adapter → ground truth."""
    global _LOADED, _LOAD_TRIED, _LOAD_ERROR
    if _LOAD_TRIED:
        return _LOADED
    with _LOCK:
        if _LOAD_TRIED:
            return _LOADED
        _LOAD_TRIED = True
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            if FOUR_BIT:
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                base = AutoModelForCausalLM.from_pretrained(
                    BASE, quantization_config=bnb, device_map="auto"
                )
            else:
                base = AutoModelForCausalLM.from_pretrained(
                    BASE, dtype=torch.bfloat16, device_map="auto"
                )
            adapter = Path(ADAPTER)
            if not adapter.exists():
                _LOAD_ERROR = f"adapter '{ADAPTER}' không tồn tại → dùng ground truth"
                _LOADED = None
                return None
            model = PeftModel.from_pretrained(base, str(adapter)).eval()
            tok = AutoTokenizer.from_pretrained(str(adapter))
            _LOADED = (model, tok)
        except Exception as exc:  # noqa: BLE001 - thiếu GPU/torch → ground truth
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
            _LOADED = None
        return _LOADED


def _run_hf(complaint: str) -> tuple[str, int]:
    import torch

    loaded = _load()
    if loaded is None:
        t0 = time.perf_counter()
        case = answer_for_complaint(complaint)
        text = assistant_content(case.think, case.resolution)
        return text, int((time.perf_counter() - t0) * 1000)
    model, tok = loaded
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": complaint},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    text = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return text, int((time.perf_counter() - t0) * 1000)


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def status() -> dict[str, object]:
    """Trạng thái cho endpoint health (vllm / slm-hf / ground-truth)."""
    if _vllm_up():
        return {
            "modelLoaded": True,
            "base": BASE,
            "adapter": ADAPTER,
            "mode": "vllm",
            "backend": VLLM_URL,
            "loadError": None,
        }
    loaded = _load()
    return {
        "modelLoaded": loaded is not None,
        "base": BASE,
        "adapter": ADAPTER,
        "loadError": _LOAD_ERROR,
        "mode": "slm-hf" if loaded is not None else "ground-truth",
        "backend": "transformers",
    }


def run_slm(complaint: str) -> tuple[str, int]:
    """Sinh output thô (``<think>`` + JSON). vLLM nếu có, else HF, else ground truth."""
    if _vllm_up():
        try:
            return _run_vllm(complaint)
        except Exception:  # noqa: BLE001 - vLLM lỗi giữa chừng → rớt về HF/ground-truth
            pass
    return _run_hf(complaint)
