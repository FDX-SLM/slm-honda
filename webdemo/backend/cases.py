"""Demo fixtures cho web demo Honda Entitlement Resolver (v3).

Mỗi ca gồm: lời than gốc (EN), tín hiệu hệ thống, bối cảnh xe, kênh output (customer/internal)
và một mock DeepSeek (LLM tổng quát) để demo chạy được khi CHƯA có DEEPSEEK_API_KEY.

Kênh output là "demo control data" — nguồn sự thật để UI tự chuyển đúng tab, KHÔNG để một câu trả
lời DeepSeek bất định quyết định tab. RC ẩn chỉ dùng cho routing/fixture, không gửi vào prompt model.
"""

from __future__ import annotations

from typing import Any

# --- 4 ca demo (thứ tự = thứ tự nút bấm). id khớp spec §11. ---
CASES: list[dict[str, Any]] = [
    {
        "id": "tcu-offline",
        "label": "TCU offline",
        "complaint": (
            "I bought Remote Start 3 days ago and it still won't work. My car has been parked in "
            "my building's underground garage all week. When I tap remote start in the app it just "
            "spins and then times out. The subscription itself shows active though."
        ),
        "expectedRootCause": "TCU_OFFLINE",
        "outputChannel": "customer",
        "customer": {
            "name": "Daniel Osei",
            "email": "daniel.osei@gmail.com",
            "phone": "+1 415-555-0142",
        },
        "vehicleContext": {
            "model": "2025 CR-V Touring",
            "region": "US-West",
            "vin": "1HGRM4H50JL000000",
        },
        "systemSignals": [
            {"name": "Payment", "value": "succeeded", "status": "ok"},
            {"name": "Subscription", "value": "active", "status": "ok"},
            {"name": "Entitlement record", "value": "present", "status": "ok"},
            {
                "name": "TCU last check-in",
                "value": "stale (garage, no cellular)",
                "status": "error",
            },
            {"name": "App remote start", "value": "spins then times out", "status": "warning"},
        ],
    },
    {
        "id": "cache-stale",
        "label": "Cache stale",
        "complaint": (
            "It works on the website but not in the app. It was fine yesterday and then suddenly "
            "stopped this morning. Logging out helps for a minute, then it's back to missing — it "
            "keeps flickering on and off. My account looks normal."
        ),
        "expectedRootCause": "ENTITLEMENT_CACHE_STALE",
        "outputChannel": "internal",
        "customer": {
            "name": "Priya Nair",
            "email": "priya.nair@gmail.com",
            "phone": "+1 617-555-0188",
        },
        "vehicleContext": {
            "model": "2024 Accord Touring",
            "region": "US-East",
            "vin": "1HGCV1F3XLA000000",
        },
        "systemSignals": [
            {"name": "Payment", "value": "succeeded", "status": "ok"},
            {"name": "Subscription", "value": "active", "status": "ok"},
            {"name": "Entitlement record", "value": "present", "status": "ok"},
            {"name": "Web portal", "value": "feature active", "status": "ok"},
            {"name": "App (CCS cache)", "value": "feature shows off", "status": "error"},
            {"name": "TCU", "value": "online", "status": "ok"},
        ],
    },
    {
        "id": "eligibility",
        "label": "Eligibility",
        "complaint": (
            "I paid for the Touring plan on my CR-V (2025, I'm in Seattle so US-West) and the app "
            "STILL prompts me to subscribe every single time. I already paid you guys. This is "
            "persistent, not a glitch."
        ),
        "expectedRootCause": "ELIGIBILITY_RULE_CONFLICT",
        "outputChannel": "internal",
        "customer": {
            "name": "Marcus Reeves",
            "email": "marcus.reeves@gmail.com",
            "phone": "+1 206-555-0119",
        },
        "vehicleContext": {
            "model": "2025 CR-V Touring",
            "region": "US-West",
            "vin": "2HKRW2H50MH000000",
        },
        "systemSignals": [
            {"name": "Payment", "value": "valid", "status": "ok"},
            {"name": "Subscription", "value": "active", "status": "ok"},
            {"name": "Connectivity / TCU", "value": "healthy", "status": "ok"},
            {
                "name": "Eligibility rule",
                "value": "feature not allowed for trim/region",
                "status": "error",
            },
        ],
    },
    {
        "id": "abstain",
        "label": "Insufficient evidence",
        "complaint": "It just doesn't work. I paid for it and nothing happens.",
        "expectedRootCause": None,
        "outputChannel": "internal",
        "customer": {
            "name": "Alex Morgan",
            "email": "alex.morgan@gmail.com",
            "phone": "+1 512-555-0170",
        },
        "vehicleContext": {
            "model": "2025 Honda (unspecified)",
            "region": "Unknown",
            "vin": "Unknown",
        },
        "systemSignals": [
            {"name": "Payment", "value": "unknown", "status": "unknown"},
            {"name": "Subscription", "value": "unknown", "status": "unknown"},
            {"name": "Entitlement record", "value": "unknown", "status": "unknown"},
            {"name": "TCU last check-in", "value": "unknown", "status": "unknown"},
        ],
    },
]

CASES_BY_ID: dict[str, dict[str, Any]] = {c["id"]: c for c in CASES}


def public_cases() -> list[dict[str, Any]]:
    """Các ca gửi cho frontend (kèm outputChannel để tự chuyển tab; không kèm mock nội bộ)."""
    keys = (
        "id",
        "label",
        "complaint",
        "expectedRootCause",
        "outputChannel",
        "customer",
        "vehicleContext",
        "systemSignals",
    )
    return [{k: c[k] for k in keys} for c in CASES]


# --- Mock Gemini (LLM tổng quát): hợp lý nhưng KÉM grounding hơn SLM — không có runbook/incident
#     ID chuẩn, owner chung chung. Dùng khi chưa cấu hình GOOGLE_API_KEY. ---
def mock_gemini(case_id: str) -> dict[str, Any]:
    """Trả NormalizedDiagnosis mock cho Gemini (LLM tổng quát) theo từng ca (isMock=True)."""
    base = {
        "model": "gemini",
        "isMock": True,
        "errorMessage": None,
        "latencyMs": 900,
        "affectedSystem": None,
        "owner": None,
        "escalation": None,
        "runbook": None,
        "similarIncident": None,
        "missingEvidence": [],
        "customerSteps": [],
        "evidence": [],
        "nextActions": [],
    }
    if case_id == "tcu-offline":
        base.update(
            {
                "status": "success",
                "outputChannel": "customer",
                "rootCause": "Connectivity issue",
                "confidence": 0.6,
                "summary": (
                    "The vehicle likely has weak cellular signal in the garage, so Remote Start "
                    "cannot reach it. Try again from an open area."
                ),
                "customerSteps": [
                    "Move the car outside to get better signal.",
                    "Restart the app and try Remote Start again.",
                    "If it still fails, contact Honda support.",
                ],
            }
        )
    elif case_id == "cache-stale":
        base.update(
            {
                "status": "success",
                "outputChannel": "internal",
                "rootCause": "Sync/caching problem",
                "confidence": 0.55,
                "summary": "Web shows the feature but the app does not, which points to a data sync delay.",
                "evidence": [
                    "active on website",
                    "app shows off",
                    "intermittent",
                    "logout/login helps",
                ],
                "affectedSystem": "App backend (unspecified)",
                "owner": "Unknown",
                "runbook": None,
                "similarIncident": None,
                "nextActions": [
                    "Clear the app cache",
                    "Reinstall the app",
                    "Wait for it to sync",
                    "Contact support if it persists",
                ],
            }
        )
    elif case_id == "eligibility":
        base.update(
            {
                "status": "success",
                "outputChannel": "internal",
                "rootCause": "Subscription not applied",
                "confidence": 0.5,
                "summary": "The purchase may not have been applied to the account; re-check the order.",
                "evidence": ["paid for Touring", "app keeps prompting to subscribe"],
                "affectedSystem": "Billing / subscription (unspecified)",
                "owner": "Unknown",
                "runbook": None,
                "similarIncident": None,
                "nextActions": [
                    "Verify the payment went through",
                    "Ask the customer to re-purchase",
                    "Restart the app",
                    "Escalate to billing if unresolved",
                ],
            }
        )
    else:  # abstain
        base.update(
            {
                "status": "success",
                "outputChannel": "internal",
                "rootCause": "General activation failure",
                "confidence": 0.4,
                "summary": "Something in the activation flow failed; try the usual troubleshooting steps.",
                "evidence": ["paid for it", "nothing happens"],
                "nextActions": [
                    "Restart the app",
                    "Log out and back in",
                    "Reinstall",
                    "Contact support",
                ],
            }
        )
    return base
