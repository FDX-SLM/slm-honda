"""Gemini proxy (server-side) — LLM tổng quát làm baseline đối chứng với SLM chuyên biệt.

Bảo mật (§9): GOOGLE_API_KEY chỉ đọc từ env ở server, không xuống client. Có timeout, xử lý mọi lỗi
(thiếu key / 429 quota / HTTP / JSON hỏng / rỗng). Chưa cấu hình key → mock có nhãn rõ ràng.

Dùng model NHỎ (mặc định gemini-2.5-flash-lite) vì key free hết quota nhanh. Nhận CÙNG input chuẩn
hoá với SLM (không nhận RC ẩn) để so sánh công bằng.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from cases import mock_gemini

API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite").strip() or "gemini-2.5-flash-lite"
BASE_URL = os.environ.get(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
).strip()
TIMEOUT_S = float(os.environ.get("GEMINI_TIMEOUT_S", "20"))

_SYSTEM = (
    "You are a general-purpose support assistant for a connected-vehicle entitlement system. "
    "Given a customer complaint and system signals, diagnose the most likely root cause and next "
    "steps. Respond ONLY with a compact JSON object using these keys: rootCause (string), "
    "confidence (0..1 number), summary (string), evidence (string[]), customerSteps (string[]), "
    "affectedSystem (string|null), owner (string|null), runbook (string|null), similarIncident "
    "(string|null), nextActions (string[]). Do not invent identifiers you are not sure about; use "
    "null when unknown."
)


def _normalized_shell(latency_ms: int) -> dict[str, Any]:
    return {
        "model": "gemini",
        "status": "success",
        "outputChannel": "internal",
        "rootCause": None,
        "confidence": None,
        "summary": "",
        "evidence": [],
        "customerSteps": [],
        "affectedSystem": None,
        "owner": None,
        "escalation": None,
        "runbook": None,
        "similarIncident": None,
        "nextActions": [],
        "missingEvidence": [],
        "latencyMs": latency_ms,
        "isMock": False,
        "errorMessage": None,
    }


def _user_prompt(complaint: str, signals: list[dict[str, Any]], requested: str) -> str:
    sig = "\n".join(f"- {s.get('name')}: {s.get('value')} [{s.get('status')}]" for s in signals)
    return (
        f"Requested output: {requested}\n\nCustomer complaint:\n{complaint}\n\n"
        f"System signals:\n{sig or '- (none)'}\n\nReturn the JSON object now."
    )


async def run_gemini(
    complaint: str,
    signals: list[dict[str, Any]],
    requested_output: str,
    output_channel: str,
    case_id: str | None,
) -> dict[str, Any]:
    """Gọi Gemini (nếu có key) → NormalizedDiagnosis; lỗi/thiếu key/quota → mock có nhãn."""
    if not API_KEY:
        mock = mock_gemini(case_id or "abstain")
        mock["outputChannel"] = output_channel
        mock["errorMessage"] = "Mock response · GOOGLE_API_KEY not configured"
        return mock

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            resp = await client.post(
                f"{BASE_URL}/models/{MODEL}:generateContent",
                params={"key": API_KEY},
                json={
                    "systemInstruction": {"parts": [{"text": _SYSTEM}]},
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": _user_prompt(complaint, signals, requested_output)}],
                        }
                    ],
                    "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
                },
            )
        if resp.status_code == 429:
            raise RuntimeError("quota/rate-limit (429)")
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        content = "".join(p.get("text", "") for p in parts).strip()
        if not content:
            raise ValueError("empty output")
        parsed = json.loads(content)
    except Exception as exc:  # noqa: BLE001 - mọi lỗi → giữ SLM hiển thị, báo lỗi non-blocking
        latency_ms = int((time.perf_counter() - t0) * 1000)
        out = _normalized_shell(latency_ms)
        out["outputChannel"] = output_channel
        out["status"] = "error"
        out["errorMessage"] = f"Gemini lỗi: {str(exc)[:60]}"
        return out

    latency_ms = int((time.perf_counter() - t0) * 1000)
    out = _normalized_shell(latency_ms)
    out["outputChannel"] = output_channel
    out["rootCause"] = parsed.get("rootCause")
    conf = parsed.get("confidence")
    out["confidence"] = float(conf) if isinstance(conf, (int, float)) else None
    out["summary"] = str(parsed.get("summary") or "")
    out["evidence"] = list(parsed.get("evidence") or [])
    out["customerSteps"] = list(parsed.get("customerSteps") or [])
    out["affectedSystem"] = parsed.get("affectedSystem")
    out["owner"] = parsed.get("owner")
    out["runbook"] = parsed.get("runbook")
    out["similarIncident"] = parsed.get("similarIncident")
    out["nextActions"] = list(parsed.get("nextActions") or [])
    return out
