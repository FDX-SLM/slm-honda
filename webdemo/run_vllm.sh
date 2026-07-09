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
echo "[vllm] serve $MODEL → http://127.0.0.1:8800 (served-name ${VLLM_MODEL:-dpo_qwen})"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name "${VLLM_MODEL:-dpo_qwen}" \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 --port 8800
