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

---

## 6. Sự cố đã gặp trên máy Vast MỚI (đã fix sẵn trong `setup.sh`, không phải làm tay)

Ba lỗi dưới đây từng làm `setup.sh` gãy trên một instance mới; nay đã được xử lý tự động trong script.
Ghi lại để biết vì sao có các dòng đó, và để chẩn đoán nếu tái diễn:

1. **`invalid peer certificate: UnknownIssuer`** khi `uv sync` / cài vLLM / `hf_hub_download`.
   Nguyên nhân: máy có **proxy chặn HTTPS (MITM)**; `curl`/`git` tin CA hệ thống nhưng `uv` (rustls) và
   `huggingface_hub` (certifi) thì không. Fix: `setup.sh` export `UV_SYSTEM_CERTS=true` +
   `REQUESTS_CA_BUNDLE`/`SSL_CERT_FILE`/`CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt`. Vô hại nếu
   máy không có proxy. (Kiểm tra thủ công: `curl -sI https://files.pythonhosted.org/` chạy được nhưng
   `uv` báo cert lỗi → đúng bệnh này.)

2. **`vllm==0.24.0+cu129 ... unsatisfiable` (flashinfer-python==0.6.12 not found)**.
   Nguyên nhân: `flashinfer-python` bản đó chỉ có trên **PyPI**, không có trên index `cu128`; uv mặc định
   chỉ xét index đầu tiên chứa gói. Fix: thêm `--index-strategy unsafe-best-match` vào lệnh cài vLLM.

3. **vLLM crash `Can't load image processor ... preprocessor_config.json`**.
   Nguyên nhân: vLLM serve qua path **multimodal** (`Qwen3_5ForConditionalGeneration`) cần
   `preprocessor_config.json` + `video_preprocessor_config.json`, nhưng `assemble_merged.py` tải base qua
   `AutoModelForCausalLM`/`AutoTokenizer` nên các file này **không** về snapshot → thiếu trong thư mục
   merge. Fix: `assemble_merged.py` nay tự `hf_hub_download` 2 file đó từ base repo vào `honda-merged-full`.
   (Máy đã merge từ trước mà thiếu file → chạy `assemble_merged.py` lại cũng chỉ bổ sung 2 file này, không
   merge lại toàn bộ.)

### Ghi chú cho agent (Claude Code) — tiết kiệm thời gian debug
- **Background Bash báo "Exit code 1" mà không có output** thường KHÔNG phải lỗi thật của dịch vụ: harness
  reap process-group khi lệnh foreground kết thúc, hoặc `pkill -f '<pattern>'` **tự khớp chính dòng lệnh
  shell của mình** và giết luôn shell đó. Trước khi kết luận vLLM/backend lỗi, hãy kiểm tra sự thật:
  `ss -tlnp | grep 8800/8600`, `nvidia-smi` (VRAM), và đọc file `.output` của task. Đừng dùng
  `pkill -f 'vllm-venv/bin/vllm'` — nó khớp chính lệnh đang chạy; hãy kill theo **PID lấy từ cổng**
  (`ss -tlnp | grep :8800 | grep -oP 'pid=\K[0-9]+'`).
- Dịch vụ chạy nền **không sống qua lần restart session** (teardown giết chúng). Sau khi quay lại, kiểm tra
  `:8800`/`:8600` và chạy lại `run_vllm.sh` (chờ ~3′) rồi `run.sh` nếu cần.

### Về hiển thị UI (không phải lỗi)
- Ca **abstain** (`INSUFFICIENT_EVIDENCE`) render card gọn (summary + "To confirm"), KHÔNG có
  owner/runbook/artifacts — đúng thiết kế: chưa có root cause thì không bịa resolution.
- Card nội bộ đôi khi thiếu owner/artifacts nếu model xuất `diagnosis` và `resolution` thành **2 JSON
  object rời** (hay gặp ở path HF sampling); `extract_json` chỉ đọc object đầu. Backend nay tự **gộp** các
  object rời (`_merge_split_resolution` trong `server.py`). Với vLLM (greedy, temp=0) output là 1 object nên
  không dính; luôn chạy vLLM cho demo.

---

## 7. Chạy 24/7 (supervisor) — tùy chọn, để demo sống độc lập VSCode

Mặc định `run_vllm.sh`/`run.sh` chạy theo phiên terminal/VSCode → đóng là chết. Muốn demo **luôn-on**
(tự restart khi crash, tự lên khi boot, không phụ thuộc VSCode) thì cài 2 service supervisor:

```bash
./webdemo/setup.sh                       # phải xong trước (merged model + vllm-venv + dist)
./webdemo/supervisor/install_services.sh # cài service vllm + honda-backend, expose backend qua Caddy
```

Việc script làm (idempotent): copy `webdemo/supervisor/{vllm,honda-backend}.sh` → `/opt/supervisor-scripts/`,
`*.conf` → `/etc/supervisor/conf.d/`, thêm mục **Honda Demo** (external 10100 → internal 8600) vào
`/etc/portal.yaml`, rồi `supervisorctl reread/update` + restart caddy. vLLM load ~3 phút lần đầu.

- Điều khiển: `supervisorctl status | restart vllm | restart honda-backend`. Log: `/var/log/portal/vllm.log`.
- **Public URL** (Caddy + token chung của instance): `http://$PUBLIC_IPADDR:$VAST_TCP_PORT_10100/?token=$OPEN_BUTTON_TOKEN`.
  Token = `echo $OPEN_BUTTON_TOKEN` (mở luôn Jupyter/Portal → chỉ đưa người tin tưởng; đang là HTTP, token đi chữ thô).
- Riêng tư hơn: bỏ public bằng cách xoá mục "Honda Demo" khỏi `/etc/portal.yaml` + restart caddy, rồi
  SSH-forward `ssh -p <VAST_TCP_PORT_22> -L 8600:127.0.0.1:8600 root@<PUBLIC_IP>` → mở `localhost:8600`.
