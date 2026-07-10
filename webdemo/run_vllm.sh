#!/bin/bash
# Chạy vLLM server tăng tốc SLM (~3x): phục vụ model ĐÃ MERGE (Qwen3.5-9B + adapter dpo_qwen).
#   ./webdemo/run_vllm.sh
# Server lên ở 127.0.0.1:8800 (OpenAI-compatible). Backend demo tự gọi qua VLLM_URL trong .env.
# Lưu ý: vLLM ở venv cô lập /workspace/vllm-venv (build cu129, khớp driver 12.9). Load ~3 phút một lần.
# Phải phục vụ checkpoint MERGE FULL (adapter merge vào text + giữ vision + config multimodal): vLLM
# không áp được LoRA lên lớp linear-attention của qwen3_5, và path text-only chưa hoàn chỉnh — nên
# serve qua path MULTIMODAL (Qwen3_5ForConditionalGeneration) đã chạy được, KHÔNG dùng --enable-lora.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
set -a; [ -f .env ] && . ./.env; set +a
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_LOGGING_LEVEL=INFO
MODEL="${VLLM_MODEL_PATH:-/workspace/honda-merged-full}"
VENV="${VLLM_VENV:-/workspace/vllm-venv}"

if [ ! -x "$VENV/bin/vllm" ]; then
  echo "[vllm] không thấy $VENV/bin/vllm — cài vLLM cu129 vào venv cô lập trước." >&2
  exit 1
fi
# INT4 (bitsandbytes nf4, in-flight): decode ~2.2x nhanh hơn bf16 (~125 vs ~57 tok/s) — 1 request đơn
# lẻ bị chặn bởi băng thông VRAM, INT4 đọc ~1/4 số byte weight nên nhanh hơn. Không cần calibrate,
# dùng chính loader qwen3_5 của vLLM (autoawq/gptq build cho transformers 4.x, không hiểu arch 5.x).
# Tắt INT4 (quay lại bf16): đặt VLLM_QUANT="" trong .env.
QUANT="${VLLM_QUANT:-bitsandbytes}"
QUANT_ARGS=()
if [ -n "$QUANT" ]; then QUANT_ARGS=(--quantization "$QUANT" --load-format "$QUANT"); fi
echo "[vllm] serve $MODEL → http://127.0.0.1:8800 (served-name ${VLLM_MODEL:-dpo_qwen}, quant=${QUANT:-none})"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "${VLLM_MODEL:-dpo_qwen}" \
  "${QUANT_ARGS[@]}" \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 --port 8800
