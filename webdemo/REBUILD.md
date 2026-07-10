# REBUILD — dựng lại web demo trên instance Vast.ai MỚI

Hướng dẫn dựng lại demo UI (SLM closed-book, vLLM INT4, streaming) sau khi **destroy instance + volume cũ**
và thuê máy mới. Mọi thứ reproducible — **KHÔNG phải train lại**.

---

## 0. Cái gì mất vs cái gì còn

| Mất khi destroy volume | Khôi phục từ đâu |
|---|---|
| `.venv`, `/workspace/vllm-venv`, `node_modules`, `webdemo/frontend/dist/` | tạo lại bằng `setup.sh` |
| `/workspace/honda-adapters` (adapter `dpo_qwen`) | **HF Hub** `tatdat01/honda-entitlement-resolver-slm` (private) |
| `/workspace/honda-merged-full` (~18GB, base+adapter đã merge) | script `assemble_merged.py` merge lại |
| base `Qwen/Qwen3.5-9B` | tải lại từ HF |
| `.env` | copy từ `.env.example`, điền lại |

### ⚠️ Điểm sống-còn duy nhất: adapter trên HF
Cả demo phụ thuộc repo HF private **`tatdat01/honda-entitlement-resolver-slm`**. Trước khi destroy hãy chắc:
- Repo đó vẫn còn trên HF (ĐỪNG xoá).
- `HF_TOKEN` có quyền đọc nó.

Nếu repo HF mất → mới phải train lại từ đầu (gen → SFT → DPO). Còn adapter còn thì chỉ cần các bước dưới.

---

## 1. Yêu cầu instance mới
- **GPU: nên thuê lại RTX 4090** (driver CUDA 12.9). vLLM wheel đang ghim `0.24.0+cu129 + torch cu128`
  cho 4090. GPU khác hẳn (vd Blackwell) phải đổi version wheel/torch.
- Disk `/workspace` trống ≥ ~45GB (base 19GB + merge 18GB + venvs).
- `HF_TOKEN` đọc được adapter private.

---

## 2. Các bước chạy

```bash
# 2.1 Clone code + đúng nhánh demo
git clone https://github.com/FDX-SLM/slm-honda.git /workspace/slm-honda
cd /workspace/slm-honda
git checkout webdemo-slm-vs-gemini        # nhánh demo (KHÔNG phải main)

# 2.2 Tạo .env — cho bản SLM-only chỉ CẦN HF_TOKEN
cp .env.example .env
nano .env
#   HF_TOKEN=hf_...          <-- BẮT BUỘC (clone adapter private)
#   (các key OpenAI/Gemini KHÔNG cần cho bản SLM-only)
#   VLLM_QUANT=bitsandbytes  <-- đã có sẵn (INT4 nf4 ~2.2x; để rỗng nếu muốn bf16)

# 2.3 Setup một lệnh (idempotent, chạy lại nhiều lần an toàn)
./webdemo/setup.sh
#   Làm: uv sync → clone adapter từ HF → tạo vLLM venv (vllm cu129 + torch cu128)
#        → cài bitsandbytes (INT4) → tải base 19GB + merge full → build frontend
#   Lần đầu ~vài phút (tải 19GB base).

# 2.4 Chạy demo — 2 tiến trình
./webdemo/run_vllm.sh     # cửa sổ 1: vLLM INT4 tại :8800 — CHỜ ~3 phút tới "Application startup complete"
./webdemo/run.sh          # cửa sổ 2: backend tại :8600
```

Mở demo: **http://localhost:8600**
- Từ máy bạn: SSH forward `ssh -p <VAST_TCP_PORT_22> -L 8600:127.0.0.1:8600 root@<PUBLIC_IP>`
- Hoặc map một open port của Vast (chưa dùng) sang 8600.

---

## 3. Kiểm tra nhanh (sau khi chạy)

```bash
curl -s http://localhost:8600/api/health
# mong đợi: {"modelLoaded":true, "mode":"vllm", "adapter":".../dpo_qwen", ...}
```

Trên UI, chọn 1 ca → **Diagnose**:
- Full text stream chạy xuống → xong dựng bản cấu trúc + block "Raw SLM output".
- Latency + **tok/s** hiện cạnh nhau (~90–125 tok/s với INT4).
- Ca **TCU offline** → tab *Customer guidance* (các bước tự sửa, không có artifacts).
- Ca nội bộ (Cache stale / Eligibility) → tab *Internal diagnosis* + **RCA / Work order / mermaid** (markdown), Severity/Priority.

---

## 4. Sự thật kỹ thuật (để khỏi hiểu nhầm)
- **Không chạy `run_vllm.sh`** → backend tự rớt về HF transformers (vẫn đúng nhưng chậm hơn, path này chưa
  test kỹ với arch này). Luôn chạy vLLM cho demo.
- **INT4 (bitsandbytes nf4)** chỉ tăng tốc **decode 1 request** (~57 → ~125 tok/s) vì decode bị chặn băng
  thông VRAM; INT4 đọc ~1/4 số byte weight. Không đổi phân loại root cause (đã verify cả 4 ca).
- **TCU** bỏ khối `artifacts` (dừng trước `"artifacts"`) cho nhanh — an toàn, không đổi phân loại. Ca nội bộ
  giữ đầy đủ artifacts. (KHÔNG prompt cắt `customer_self_service` vì làm model phân loại sai — đã thử, hỏng.)
- **AWQ/GPTQ offline không dùng được** vì arch `qwen3_5` (transformers 5.x) chưa được các tool đó hỗ trợ;
  bitsandbytes in-flight dùng chính loader của vLLM nên chạy.

---

## 5. Xử lý sự cố
- `setup.sh` báo thiếu `HF_TOKEN` → điền vào `.env` rồi chạy lại.
- Clone adapter fail 401/403 → `HF_TOKEN` không có quyền đọc repo private → xin lại quyền/tạo token mới.
- vLLM crash `driver too old` / `unsupported PTX` → GPU/driver không khớp wheel cu129 → thuê lại 4090, hoặc
  đổi wheel+torch theo GPU (xem `vast-capabilities | jq '.hardware.gpu.cuda'`).
- vLLM `bitsandbytes ... not found` → chạy `uv pip install --python /workspace/vllm-venv/bin/python bitsandbytes`
  (setup.sh đã tự làm; chỉ cần khi gắn lại volume cũ có vllm-venv nhưng thiếu bnb).
- Port 8600/8800 bận → `PORT=8700 ./webdemo/run.sh`, hoặc `VLLM_URL` trỏ cổng khác.
