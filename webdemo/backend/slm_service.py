"""SLM service — sinh output closed-book cho một lời than.

Thứ tự ưu tiên:
  1) vLLM server (OpenAI-compatible) nếu chạy → NHANH (~3x, resident). Không nạp model HF (nhường GPU).
  2) HF transformers (base + adapter DPO) nếu không có vLLM và có GPU.
  3) Ground truth (answer_for_complaint) nếu thiếu GPU/adapter — vẫn đúng schema, demo không vỡ.

vLLM chạy ở venv cô lập /workspace/vllm-venv (build cu129 khớp driver). Backend chỉ gọi HTTP tới nó.
"""

from __future__ import annotations

import json
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
# Chống degeneration lặp: ca ELIGIBILITY_RULE_CONFLICT dễ kẹt vòng lặp customer_self_service
# (đẻ tier vô hạn "…credit the month" → JSON không đóng → parse lỗi). 1.05 phá vòng lặp mà KHÔNG
# đổi root cause; 1.1 quá mạnh (làm sai RC các ca khác). Áp cho cả 3 đường sinh (vllm/stream/hf).
REPETITION_PENALTY = float(os.environ.get("HONDA_REPETITION_PENALTY", "1.05"))

# `artifacts` (rca_md/work_order_md/customer_email/diagram_mermaid) là key CUỐI & NẶNG nhất của JSON.
# CHỈ ca hướng dẫn khách (TCU) mới không cần nó → chặn sinh từ đúng chỗ này để nhanh hơn ~1.4x.
# Các ca nội bộ (cache/eligibility/…) VẪN cần artifacts nên sinh đầy đủ. Quyết định per-case ở caller.
# Env chỉ là "van tổng": HONDA_SKIP_ARTIFACTS=false → không bao giờ cắt (kể cả TCU).
_ALLOW_SKIP = os.environ.get("HONDA_SKIP_ARTIFACTS", "true").strip().lower() not in (
    "false",
    "0",
    "no",
)
# Chuỗi stop: model dừng NGAY trước khi mở khối artifacts (vLLM/transformers loại chuỗi này khỏi output).
# An toàn vì diagnosis đã sinh xong TRƯỚC điểm này — không đổi phân loại root cause.
# (Đã thử prompt model bỏ `customer_self_service` cho ca nội bộ để nhanh hơn, NHƯNG nó làm model
#  fine-tuned phân loại sai / rớt artifacts → BỎ, ca nội bộ sinh đầy đủ.)
_ARTIFACTS_STOP = '"artifacts"'

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
# JSON repair (đóng lại object bị cắt ở stop "artifacts")
# --------------------------------------------------------------------------- #
def _close_truncated_json(text: str) -> str:
    """Đóng object/array còn mở khi generation dừng sớm ở ``"artifacts"``.

    Đi qua phần sau ``</think>``, đếm ngoặc (bỏ qua nội dung trong chuỗi). Nếu đã cân bằng thì
    trả nguyên văn; nếu còn ngoặc mở thì bỏ dấu phẩy thừa cuối rồi thêm ngoặc đóng tương ứng.
    """
    start = text.find("{")
    if start == -1:
        return text
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in text[start:]:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if not stack or in_str:  # đã cân bằng, hoặc bị cắt giữa chuỗi (không tự sửa) → để nguyên
        return text
    body = text.rstrip()
    if body.endswith(","):
        body = body[:-1].rstrip()
    closers = {"{": "}", "[": "]"}
    return body + "".join(closers[c] for c in reversed(stack))


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


def _run_vllm(complaint: str, skip_artifacts: bool) -> tuple[str, int, int | None]:
    """Gọi vLLM /chat/completions (áp LoRA dpo_qwen); vLLM tự render chat template.

    Returns:
        (text, latency_ms, gen_tokens) — gen_tokens lấy từ ``usage.completion_tokens`` để tính token/s.
    """
    t0 = time.perf_counter()
    payload: dict[str, object] = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": complaint},
        ],
        "temperature": 0,
        "max_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
    }
    if skip_artifacts:
        payload["stop"] = [_ARTIFACTS_STOP]  # dừng trước khối artifacts (vLLM loại chuỗi stop)
    r = httpx.post(f"{VLLM_URL}/chat/completions", json=payload, timeout=VLLM_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    text = (data["choices"][0]["message"]["content"] or "").strip()
    gen_tokens = (data.get("usage") or {}).get("completion_tokens")
    if skip_artifacts:
        text = _close_truncated_json(text)
    return text, int((time.perf_counter() - t0) * 1000), gen_tokens


def _stream_vllm(complaint: str, skip_artifacts: bool):
    """Stream vLLM: yield ``("token", delta)`` từng đoạn, cuối cùng ``("final", text, ms, gen_tokens)``.

    ``text`` cuối là bản đã đóng JSON (nếu skip_artifacts) để parse; các token stream ra là raw model.
    """
    t0 = time.perf_counter()
    payload: dict[str, object] = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": complaint},
        ],
        "temperature": 0,
        "max_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": REPETITION_PENALTY,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if skip_artifacts:
        payload["stop"] = [_ARTIFACTS_STOP]
    parts: list[str] = []
    gen_tokens: int | None = None
    with httpx.stream(
        "POST", f"{VLLM_URL}/chat/completions", json=payload, timeout=VLLM_TIMEOUT
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            try:
                j = json.loads(data)
            except json.JSONDecodeError:
                continue
            if j.get("usage"):
                gen_tokens = j["usage"].get("completion_tokens")
            choices = j.get("choices") or []
            if choices:
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    parts.append(delta)
                    yield ("token", delta)
    text = "".join(parts).strip()
    if skip_artifacts:
        text = _close_truncated_json(text)
    yield ("final", text, int((time.perf_counter() - t0) * 1000), gen_tokens)


def stream_slm(complaint: str, skip_artifacts: bool = False):
    """Sinh streaming: dùng vLLM nếu có, else chạy blocking (HF/ground-truth) rồi phát cả cục.

    Yield: ``("token", str)`` nhiều lần, rồi ``("final", text, latency_ms, gen_tokens)`` một lần.
    """
    skip = skip_artifacts and _ALLOW_SKIP
    if _vllm_up():
        yield from _stream_vllm(complaint, skip)
        return
    text, ms, gen_tokens = _run_hf(complaint, skip)  # fallback: không stream token thật
    yield ("token", text)
    yield ("final", text, ms, gen_tokens)


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


def _run_hf(complaint: str, skip_artifacts: bool) -> tuple[str, int, int | None]:
    import torch

    loaded = _load()
    if loaded is None:  # ground-truth fallback: không sinh token thật → gen_tokens=None
        t0 = time.perf_counter()
        case = answer_for_complaint(complaint)
        text = assistant_content(case.think, case.resolution)
        return text, int((time.perf_counter() - t0) * 1000), None
    model, tok = loaded
    t0 = time.perf_counter()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": complaint},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    gen_kwargs: dict[str, object] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "repetition_penalty": REPETITION_PENALTY,
        "pad_token_id": tok.pad_token_id or tok.eos_token_id,
    }
    if skip_artifacts:  # transformers >=4.38: dừng khi gặp chuỗi (cần truyền tokenizer)
        gen_kwargs["stop_strings"] = [_ARTIFACTS_STOP]
        gen_kwargs["tokenizer"] = tok
    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    gen_tokens = int(out.shape[1] - enc["input_ids"].shape[1])  # số token mới sinh
    text = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    if skip_artifacts:
        # transformers GIỮ chuỗi stop trong output → cắt bỏ rồi đóng JSON.
        idx = text.rfind(_ARTIFACTS_STOP)
        if idx != -1:
            text = text[:idx]
        text = _close_truncated_json(text)
    return text, int((time.perf_counter() - t0) * 1000), gen_tokens


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


def run_slm(complaint: str, skip_artifacts: bool = False) -> tuple[str, int, int | None]:
    """Sinh output thô (``<think>`` + JSON). vLLM nếu có, else HF, else ground truth.

    Args:
        complaint: lời than của khách (thô).
        skip_artifacts: True chỉ với ca hướng dẫn khách (TCU) — dừng trước khối ``artifacts`` cho
            nhanh. Các ca nội bộ để False để sinh đầy đủ RCA/work-order/email/mermaid.

    Returns:
        (text, latency_ms, gen_tokens) — gen_tokens=None nếu không đo được (ground-truth fallback).
    """
    skip = skip_artifacts and _ALLOW_SKIP  # env HONDA_SKIP_ARTIFACTS=false là van tổng tắt hẳn
    if _vllm_up():
        try:
            return _run_vllm(complaint, skip)
        except Exception:  # noqa: BLE001 - vLLM lỗi giữa chừng → rớt về HF/ground-truth
            pass
    return _run_hf(complaint, skip)
