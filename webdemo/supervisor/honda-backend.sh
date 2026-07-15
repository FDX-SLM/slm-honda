#!/bin/bash
# Supervisor wrapper: backend FastAPI + UI (:8600). Expose qua Caddy với label "Honda Demo"
# (self-skip nếu gỡ mục đó khỏi /etc/portal.yaml). Backend tự phát hiện vLLM per-request nên
# không cần chờ service vllm sẵn sàng.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "Honda Demo"

exec /workspace/slm-honda/webdemo/run.sh
