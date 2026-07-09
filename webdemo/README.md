# Web demo — Honda Entitlement Resolver (v3)

React UI (Vite + TypeScript) + backend FastAPI, chạy trên **một cổng**. Bấm **Diagnose** → chạy
**SLM chuyên biệt** (closed-book, GPU) và **Gemini** (LLM tổng quát) **song song** trên cùng một
complaint, chuẩn hoá về một schema và hiển thị cạnh nhau. Kết quả tự route theo độ phức tạp của ca:

| Ca | Root cause | Tab hiển thị |
|---|---|---|
| TCU offline | `TCU_OFFLINE` | **Hướng dẫn khách hàng** — chỉ bước tự xử lý an toàn |
| Cache stale | `ENTITLEMENT_CACHE_STALE` | **Chẩn đoán nội bộ** — owner/runbook/evidence |
| Eligibility | `ELIGIBILITY_RULE_CONFLICT` | **Chẩn đoán nội bộ** |
| Insufficient evidence | abstain | **Chẩn đoán nội bộ** — hỏi thêm bằng chứng, không đoán |

## Chạy

```bash
./webdemo/run.sh            # build frontend (lần đầu) + backend ở cổng 8600
# mở http://localhost:8600
```

Cần `.env` ở gốc repo (đã có `HONDA_BASE`, `HONDA_ADAPTER`, `HONDA_4BIT`, `HONDA_MAX_NEW_TOKENS`).
Không GPU/adapter → backend tự chạy **ground-truth** (vẫn đúng schema) để demo không vỡ.

## Tăng tốc bằng vLLM (~3x) — ĐÃ CHẠY

HF transformers chạy Qwen3.5-9B (hybrid linear-attention) ~18 tok/s (mỗi ca ~40–80s). vLLM đạt
**~24s/ca (~3x)** và cho **đúng RC** như HF (confidence trùng khớp).

Bật:
```bash
./webdemo/run_vllm.sh    # vLLM serve ở 127.0.0.1:8800 (load ~3 phút, giữ resident) — chạy TRƯỚC
./webdemo/run.sh         # backend tự phát hiện vLLM (VLLM_URL) và gọi qua đó; vLLM tắt → tự rớt về HF
```

Cách làm (Qwen3.5 = hybrid linear-attention + multimodal + M-RoPE + MTP, rất mới):
1. vLLM ở venv cô lập `/workspace/vllm-venv` — bản `vllm==0.24.0+cu129` (khớp driver CUDA 12.9;
   4090 không có forward-compat nên KHÔNG dùng được wheel cu130 mặc định).
2. vLLM KHÔNG áp được LoRA lên lớp linear-attention, và path **text-only** của vLLM 0.24 chưa hoàn
   chỉnh (thiếu `get_mamba_state_copy_func`). Nên: **merge adapter vào base + giữ nguyên vision/MTP +
   config multimodal** → `/workspace/honda-merged-full` → vLLM serve qua path **multimodal**
   (`Qwen3_5ForConditionalGeneration`) đã chạy được. Không dùng `--enable-lora`.

vLLM giữ ~40GB VRAM (util 0.85); khi bật vLLM thì backend KHÔNG nạp model HF (nhường GPU). Muốn tắt
vLLM: đặt `VLLM_URL=` rỗng (hoặc không chạy run_vllm.sh) → backend dùng HF.

## Gemini (LLM tổng quát đối chứng)

Baseline LLM tổng quát để đối chứng với SLM (không phải RAG, không phải judge chấm điểm). Key đọc ở
**server** (`.env`, `GOOGLE_API_KEY`), không bao giờ xuống client. Chưa cấu hình key → **mock có nhãn**.
Dùng model **nhỏ** (`gemini-2.5-flash-lite`) vì key free hết quota nhanh; lỗi 429/quota → panel Gemini
báo lỗi non-blocking, SLM vẫn hiển thị.

```env
GOOGLE_API_KEY=...            # đã có sẵn (dùng chung với eval)
GEMINI_MODEL=gemini-2.5-flash-lite
```

## Cấu trúc

```
webdemo/
  backend/   server.py (FastAPI) · slm_service · gemini_service · normalize · cases
  frontend/  React + Vite + TS (src/App.tsx, styles.css, api.ts, types.ts)
```

## Phát triển frontend

```bash
cd webdemo/frontend && npm run dev     # Vite HMR ở :5173, proxy /api → 127.0.0.1:8600
```
