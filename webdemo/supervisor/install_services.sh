#!/bin/bash
# ============================================================================
# Cài vLLM + backend thành SERVICE supervisor → chạy 24/7, sống độc lập VSCode,
# tự restart khi crash, tự lên khi boot. Idempotent (chạy lại an toàn).
#
#   ./webdemo/supervisor/install_services.sh
#
# Cần: đã chạy ./webdemo/setup.sh xong (có merged model + vllm-venv + frontend dist).
# Sau khi cài, vLLM cần ~3 phút load lần đầu. Kiểm tra: supervisorctl status
# Mở public URL (Caddy + token): http://$PUBLIC_IPADDR:$VAST_TCP_PORT_10100/?token=$OPEN_BUTTON_TOKEN
# ============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[svc] copy wrapper → /opt/supervisor-scripts/"
cp "$HERE/vllm.sh" "$HERE/honda-backend.sh" /opt/supervisor-scripts/
chmod +x /opt/supervisor-scripts/vllm.sh /opt/supervisor-scripts/honda-backend.sh

echo "[svc] copy conf → /etc/supervisor/conf.d/"
cp "$HERE/vllm.conf" "$HERE/honda-backend.conf" /etc/supervisor/conf.d/

echo "[svc] expose backend qua Caddy (portal.yaml: Honda Demo, external 10100 → internal 8600)"
python3 - <<'PY'
import yaml
p = "/etc/portal.yaml"
d = yaml.safe_load(open(p)) or {}
d.setdefault("applications", {})
d["applications"]["Honda Demo"] = {
    "hostname": "localhost", "external_port": 10100, "internal_port": 8600,
    "open_path": "/", "name": "Honda Demo",
}
yaml.safe_dump(d, open(p, "w"), sort_keys=False)
print("  portal.yaml: Honda Demo OK")
PY

echo "[svc] nạp service + restart caddy"
supervisorctl reread
supervisorctl update
supervisorctl restart caddy

echo "[svc] XONG. supervisorctl status:"
supervisorctl status | grep -E 'vllm|honda-backend|caddy' || true
cat <<EOF

  vLLM đang load (~3 phút). Theo dõi: tail -f /var/log/portal/vllm.log
  Health:  curl -s localhost:8600/api/health   (đợi "mode":"vllm")
  Public:  http://\$PUBLIC_IPADDR:\$VAST_TCP_PORT_10100/?token=\$OPEN_BUTTON_TOKEN
  Gỡ public (chỉ nội bộ): xoá mục "Honda Demo" khỏi /etc/portal.yaml rồi supervisorctl restart caddy
EOF
