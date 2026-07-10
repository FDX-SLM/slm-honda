"""Chuẩn hoá output SLM (đã parse) → NormalizedDiagnosis (schema chung §10 cho cả SLM & DeepSeek).

SLM sinh schema đã train (diagnosis/runbook_id/customer_self_service/artifacts...). Ở đây map sang
schema phẳng để frontend render đồng nhất 2 cột. RC ABSTAIN → status="abstain" (vẫn là kết quả hợp lệ,
không phải lỗi). Affected system suy từ map theo RC (SLM JSON không có field riêng).
"""

from __future__ import annotations

from typing import Any

ABSTAIN_RC = "INSUFFICIENT_EVIDENCE"

# Hệ thống bị ảnh hưởng theo RC (bám SYSTEMS §1.1 trong ground_truth).
AFFECTED_SYSTEM: dict[str, str] = {
    "TCU_OFFLINE": "TCU / Vehicle (cellular link)",
    "ENTITLEMENT_CACHE_STALE": "Entitlement Service → CCS Portal cache",
    "ELIGIBILITY_RULE_CONFLICT": "Entitlement Service — eligibility rules",
}


def _empty(model: str) -> dict[str, Any]:
    """Khung NormalizedDiagnosis rỗng."""
    return {
        "model": model,
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
        "severity": None,
        "priority": None,
        "artifacts": None,
        "latencyMs": None,
        "isMock": False,
        "errorMessage": None,
    }


def normalize_slm(
    think: str,
    res: dict[str, Any] | None,
    latency_ms: int,
    output_channel: str,
) -> dict[str, Any]:
    """Map resolution package của SLM → NormalizedDiagnosis.

    Args:
        think: khối ``<think>`` (không hiển thị cho khách).
        res: dict resolution đã parse (None nếu parse fail → status error).
        latency_ms: thời gian generate (ms).
        output_channel: "customer" | "internal" (từ fixture — nguồn sự thật routing).
    """
    out = _empty("slm")
    out["latencyMs"] = latency_ms
    out["outputChannel"] = output_channel
    if res is None:
        out["status"] = "error"
        out["errorMessage"] = "Không parse được resolution package từ output của model."
        return out

    diag = res.get("diagnosis", {}) or {}
    rc = diag.get("leading_root_cause")
    out["rootCause"] = rc
    out["confidence"] = diag.get("confidence")
    out["evidence"] = list(diag.get("evidence_in_ticket", []) or [])
    out["missingEvidence"] = list(diag.get("to_confirm", []) or [])

    is_abstain = rc == ABSTAIN_RC or rc is None
    out["status"] = "abstain" if is_abstain else "success"

    # Khách: lời giải thích đời thường + các bước tự xử lý.
    out["summary"] = res.get("why_plain") or res.get("why_technical") or ""
    out["customerSteps"] = [
        s.get("action", "") for s in (res.get("customer_self_service") or []) if s.get("action")
    ]

    # Nội bộ: định tuyến + hành động.
    out["affectedSystem"] = AFFECTED_SYSTEM.get(rc or "")
    out["owner"] = res.get("owner_team")
    out["escalation"] = res.get("escalation")
    out["runbook"] = res.get("runbook_id")
    out["similarIncident"] = res.get("similar_incident")
    out["nextActions"] = list(res.get("fix_steps", []) or [])
    out["severity"] = res.get("severity")
    out["priority"] = res.get("priority")
    # Artifacts (RCA / work-order / email / mermaid) — chỉ lấy các trường chuỗi để UI in ra.
    arts = res.get("artifacts") or {}
    out["artifacts"] = (
        {
            "rcaMd": arts.get("rca_md"),
            "workOrderMd": arts.get("work_order_md"),
            "customerEmail": arts.get("customer_email"),
            "diagramMermaid": arts.get("diagram_mermaid"),
        }
        if arts
        else None
    )

    if is_abstain:
        # Abstain: tóm tắt thành thông điệp "chưa đủ bằng chứng" + danh sách cần kiểm.
        out["summary"] = out["summary"] or (
            "Chưa xác định được root cause đáng tin từ bằng chứng hiện có."
        )
        out["owner"] = out["owner"] or None
    return out
