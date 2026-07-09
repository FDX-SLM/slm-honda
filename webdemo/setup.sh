#!/bin/bash
# ============================================================================
# SETUP MỘT LỆNH cho web demo Honda (React + FastAPI + vLLM ~3x).
#   ./webdemo/setup.sh
# Idempotent: bước nào đã xong thì bỏ qua. An toàn chạy lại nhiều lần.
#
# Yêu cầu duy nhất TRƯỚC khi chạy: .env ở gốc repo có HF_TOKEN (đọc adapter private)
# và DEEPSEEK_API_KEY (cột DeepSeek). Nếu /workspace là volume và bạn gắn lại volume cũ,
# hầu hết đã có sẵn → script chạy vèo cái là xong.
#
# Sau khi setup xong, chạy demo bằng 2 lệnh:
#   ./webdemo/run_vllm.sh     # vLLM server :8800 (chờ ~3 phút load)
#   ./webdemo/run.sh          # backend :8600 → mở http://localhost:8600
# ============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

log() { echo -e "\n\033[1;36m[setup]\033[0m $*"; }
have() { command -v "$1" >/dev/null 2>&1; }

VLLM_VENV="/workspace/vllm-venv"
ADAPTERS="/workspace/honda-adapters"
MERGED="/workspace/honda-merged-full"
ADAPTER_REPO="tatdat01/honda-entitlement-resolver-slm"
VLLM_WHEEL="https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"

# --- 0. .env + token ---
if [ ! -f .env ]; then
  log "THIẾU .env — tạo từ mẫu rồi điền HF_TOKEN + DEEPSEEK_API_KEY:"
  echo "    cp .env.example .env && nano .env"
  exit 1
fi
set -a; . ./.env; set +a
: "${HF_TOKEN:?Thiếu HF_TOKEN trong .env (cần để tải adapter private)}"

# --- 1. Python env cho backend + merge (torch cu128 + fastapi) ---
log "1/6 Cài Python deps (uv sync --extra train --extra webdemo)…"
uv sync --extra train --extra webdemo

# --- 2. Adapter (clone từ HF, private) ---
if [ -f "$ADAPTERS/dpo_qwen/adapter_config.json" ]; then
  log "2/6 Adapter đã có ($ADAPTERS) — bỏ qua."
else
  log "2/6 Clone adapter từ HF ($ADAPTER_REPO)…"
  GIT_LFS_SKIP_SMUDGE=0 git clone "https://user:${HF_TOKEN}@huggingface.co/${ADAPTER_REPO}" "$ADAPTERS"
fi

# --- 3. vLLM venv cô lập (wheel cu129 khớp driver 12.9 + torch cu128) ---
if [ -x "$VLLM_VENV/bin/vllm" ] && "$VLLM_VENV/bin/python" -c "import vllm,torch;assert 'cu128' in torch.__version__" 2>/dev/null; then
  log "3/6 vLLM venv đã có ($VLLM_VENV) — bỏ qua."
else
  log "3/6 Tạo vLLM venv + cài vllm==0.24.0+cu129 (khớp driver CUDA 12.9) + torch cu128…"
  uv venv "$VLLM_VENV" --python 3.12
  uv pip install --python "$VLLM_VENV/bin/python" "$VLLM_WHEEL" --extra-index-url https://download.pytorch.org/whl/cu128
  # ép torch-family sang cu128 (4090 driver 12.9, KHÔNG forward-compat → cu130 sẽ fail 'driver too old')
  uv pip install --python "$VLLM_VENV/bin/python" --index-url https://download.pytorch.org/whl/cu128 \
    "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
fi

# --- 4. Merge full checkpoint (tải base 19GB + merge) ---
if [ -f "$MERGED/model.safetensors.index.json" ]; then
  log "4/6 Checkpoint merge đã có ($MERGED) — bỏ qua."
else
  log "4/6 Lắp checkpoint merge full (tải base Qwen3.5-9B ~19GB + merge, vài phút)…"
  uv run python webdemo/scripts/assemble_merged.py
fi

# --- 5. Frontend build ---
log "5/6 Build frontend (React/Vite)…"
. /opt/nvm/nvm.sh 2>/dev/null || true
( cd webdemo/frontend && npm install && npm run build )

# --- 6. Xong ---
log "6/6 XONG. Chạy demo:"
cat <<'EOF'

  ./webdemo/run_vllm.sh     # cửa sổ 1: vLLM server (chờ ~3 phút tới 'Application startup complete')
  ./webdemo/run.sh          # cửa sổ 2: backend → mở http://localhost:8600

  (không chạy run_vllm.sh thì backend tự dùng HF transformers — chậm hơn nhưng vẫn đúng)
EOF
