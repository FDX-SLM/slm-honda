#!/bin/bash
# Supervisor wrapper: vLLM INT4 server cho Honda demo (nội bộ 127.0.0.1:8800).
# Backend gọi qua VLLM_URL — KHÔNG expose qua Caddy nên KHÔNG source exit_portal.
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/environment.sh"

exec /workspace/slm-honda/webdemo/run_vllm.sh
