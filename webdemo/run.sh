#!/bin/bash
# Chạy web demo Honda Entitlement Resolver (React UI + FastAPI backend, 1 cổng).
#   ./webdemo/run.sh            # build frontend nếu chưa có, rồi chạy backend ở 127.0.0.1:8600
#   PORT=8700 ./webdemo/run.sh  # đổi cổng
# Backend phục vụ luôn frontend đã build + API /api/* → mở http://localhost:<PORT>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8600}"

# build frontend nếu chưa có dist
if [ ! -f webdemo/frontend/dist/index.html ]; then
  echo "[webdemo] build frontend…"
  . /opt/nvm/nvm.sh 2>/dev/null || true
  ( cd webdemo/frontend && npm install && npm run build )
fi

echo "[webdemo] backend → http://127.0.0.1:${PORT}  (mở localhost:${PORT} trên máy bạn)"
exec env PYTHONPATH="$ROOT/src:$ROOT/webdemo/backend" \
  uv run uvicorn server:app --app-dir "$ROOT/webdemo/backend" --host 0.0.0.0 --port "$PORT"
