"""FastAPI backend cho web demo Honda Entitlement Resolver (v3).

Endpoints:
    GET  /api/health            trạng thái model (slm thật hay ground-truth)
    GET  /api/cases             4 ca demo (kèm outputChannel để UI tự chuyển tab)
    POST /api/diagnose/slm      chạy SLM chuyên biệt → NormalizedDiagnosis
    POST /api/diagnose/gemini   proxy Gemini (LLM tổng quát) → NormalizedDiagnosis

Frontend gọi 2 endpoint diagnose SONG SONG (Promise.allSettled) nên một model lỗi không chặn model kia.
SLM endpoint là ``def`` (sync) → FastAPI chạy trong threadpool, không block event loop của DeepSeek.
Phục vụ luôn frontend đã build (../frontend/dist) trên cùng cổng → 1 process, không cần CORS.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# nạp .env (HONDA_*, DEEPSEEK_*) TRƯỚC khi import service đọc env.
_ROOT = Path(__file__).resolve().parents[2]  # .../slm-honda
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # cho phép import cases/normalize/... phẳng

import slm_service  # noqa: E402
from cases import CASES_BY_ID, public_cases  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from gemini_service import run_gemini  # noqa: E402
from normalize import normalize_slm  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from slm_coach.oracle import parse_output  # noqa: E402

app = FastAPI(title="Honda Entitlement Resolver — Web Demo")

_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


class DiagnoseRequest(BaseModel):
    complaint: str
    caseId: str | None = None
    requestedOutput: str = "internal_diagnosis"


def _case_channel(case_id: str | None) -> str:
    """Kênh output = nguồn sự thật routing (demo). Không biết ca → internal (an toàn nhất)."""
    case = CASES_BY_ID.get(case_id or "")
    return case["outputChannel"] if case else "internal"


@app.get("/api/health")
def health() -> dict[str, Any]:
    return slm_service.status()


@app.get("/api/cases")
def cases() -> list[dict[str, Any]]:
    return public_cases()


@app.post("/api/diagnose/slm")
def diagnose_slm(req: DiagnoseRequest) -> dict[str, Any]:
    """Chạy SLM closed-book trên lời than → NormalizedDiagnosis (chạy trong threadpool)."""
    text, latency_ms = slm_service.run_slm(req.complaint)
    # chat template chèn sẵn '<think>' vào prompt nên output bắt đầu thẳng bằng reasoning; thêm lại
    # thẻ mở để split_think lấy được phần suy luận.
    if "</think>" in text and "<think>" not in text:
        text = "<think>\n" + text
    think, res = parse_output(text)
    return normalize_slm(think, res, latency_ms, _case_channel(req.caseId))


@app.post("/api/diagnose/gemini")
async def diagnose_gemini(req: DiagnoseRequest) -> dict[str, Any]:
    """Proxy Gemini (LLM tổng quát) trên CÙNG input (không gửi RC ẩn) → NormalizedDiagnosis."""
    case = CASES_BY_ID.get(req.caseId or "")
    signals = case["systemSignals"] if case else []
    return await run_gemini(
        complaint=req.complaint,
        signals=signals,
        requested_output=req.requestedOutput,
        output_channel=_case_channel(req.caseId),
        case_id=req.caseId,
    )


# --- Phục vụ frontend đã build (nếu có). Đặt CUỐI để không nuốt /api. ---
if _DIST.exists():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_DIST / "index.html")

    @app.get("/{path:path}")
    def spa(path: str) -> FileResponse:
        target = _DIST / path
        if target.is_file():
            return FileResponse(target)
        return FileResponse(_DIST / "index.html")  # SPA fallback
