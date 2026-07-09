"""Streamlit demo cho Honda Entitlement Resolver (PoC6 §8) — closed-book, KHÔNG tool.

Chạy:  uv run streamlit run app.py   (hoặc:  python -m streamlit run app.py  với PYTHONPATH=src)

Hai tab:
  Customer ticket — khách tạo ticket khiếu nại ("đã trả tiền nhưng không dùng được") → SLM
     closed-book đọc cue → resolution package. UI theo 3 BƯỚC, tách RÕ phần gửi cho KHÁCH (email +
     hướng dẫn tự xử lý + lời giải thích đời thường) với phần HỖ TRỢ NỘI BỘ (chẩn đoán kỹ thuật,
     owner/severity/churn, RCA/work-order/sơ đồ). Mỗi section có icon note — hover để xem ý nghĩa.
  SLM vs RAG — split-screen đối chứng: RAG "copy nearest ticket" (sai RC trên ca cue-flip, không
     biết abstain) để làm nổi bật SLM đọc cue. RAG là BASELINE đối chứng, không phải tool của SLM.

Mô hình đọc DUY NHẤT lời than (closed-book): metadata ticket (tên/VIN/xe) chỉ trang trí thẻ ticket,
KHÔNG bao giờ đưa vào prompt. Không gọi tool, không truy vấn telemetry.

Env (không hardcode path máy cá nhân):
    HONDA_BASE      base model (mặc định Qwen/Qwen3.5-9B)
    HONDA_ADAPTER   thư mục adapter SFT/DPO (mặc định checkpoints/sft/best)
    HONDA_4BIT      nạp base 4-bit? "true"=QLoRA (mặc định true)
Nếu KHÔNG nạp được model (thiếu GPU/adapter) → chạy DEMO mode từ ground truth (vẫn trung thực).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from slm_coach.datagen.core import answer_for_complaint, assistant_content
from slm_coach.eval.rag import RagBaseline
from slm_coach.ground_truth import CHURN_RISK_LEVELS, PRIORITY, SEVERITY, SYSTEM_PROMPT
from slm_coach.oracle import parse_output

# Tự nạp .env (HONDA_BASE/HONDA_ADAPTER/HONDA_4BIT/HONDA_MAX_NEW_TOKENS...) để `uv run streamlit run
# app.py` trần vẫn trỏ đúng adapter — không cần export tay. Thiếu python-dotenv thì bỏ qua êm.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"))
except Exception:  # noqa: BLE001 - dotenv là tùy chọn; env vẫn có thể set từ shell
    pass

BASE = os.environ.get("HONDA_BASE", "Qwen/Qwen3.5-9B")
ADAPTER = os.environ.get("HONDA_ADAPTER", "checkpoints/sft/best")
FOUR_BIT = os.environ.get("HONDA_4BIT", "true").strip().lower() not in ("false", "0", "no")
# Output đầy đủ (<think> + JSON + artifacts) ~1550 token; mặc định phải đủ rộng kẻo JSON bị cắt cụt.
MAX_NEW_TOKENS = int(os.environ.get("HONDA_MAX_NEW_TOKENS", "1700"))

ABSTAIN = "INSUFFICIENT_EVIDENCE"

#: Lời than mẫu (mỗi RC một ca + một ca mơ hồ để demo abstain). Nút bấm sẽ nạp vào ô soạn ticket.
SAMPLES: dict[str, str] = {
    "TCU offline": (
        "I bought Remote Start 3 days ago and it still won't work. My car has been parked in my "
        "building's underground garage all week. When I tap remote start in the app it just spins "
        "and then times out. The subscription itself shows active though."
    ),
    "Cache stale": (
        "I subscribed to Remote Climate a couple of days ago. I can see it active on the website "
        "but the app shows it as off. It worked fine yesterday and then suddenly stopped — it's "
        "intermittent, logging out and back in sometimes helps."
    ),
    "Eligibility": (
        "I paid for the Touring package yesterday but the app keeps prompting me to Subscribe even "
        "though I already paid. I have a CR-V Touring here in US-West."
    ),
    "Mơ hồ → abstain": "It just doesn't work. I paid for it and nothing happens.",
}

# Chú thích section (hover vào icon note để xem). Gom một chỗ cho gọn.
HELP_TICKET = (
    "Khách mô tả vấn đề bằng lời thường. Model CHỈ đọc nội dung khiếu nại (closed-book); "
    "tên/email/điện thoại/xe/VIN chỉ để dựng thẻ ticket cho CS liên hệ lại, không đưa vào model."
)
HELP_DIAGNOSE = (
    "Model đọc cue trong lời than → suy ra root cause + gói xử lý. Closed-book: không gọi tool, "
    "không truy vấn telemetry, không bịa timestamp."
)
HELP_CUSTOMER = (
    "Phần KHÁCH nhận được: lời giải thích đời thường + email phản hồi + hướng dẫn tự xử lý. "
    "Không lộ thuật ngữ hay quy trình kỹ thuật nội bộ."
)
HELP_INTERNAL = (
    "Chỉ nhân viên Honda thấy: chẩn đoán kỹ thuật, định tuyến (owner/severity), rủi ro rời bỏ, "
    "và artifacts (RCA / work-order / sơ đồ). KHÔNG gửi cho khách."
)
HELP_RAW = "Nguyên văn model trả: khối <think> + JSON resolution package mà UI phía trên parse ra."


# ---------------------------------------------------------------------------
# Model loading + closed-book generation (KHÔNG tool, KHÔNG telemetry)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model() -> tuple[object, object] | None:
    """Nạp base + adapter; trả None nếu thiếu GPU/torch/adapter (→ DEMO mode từ ground truth)."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if FOUR_BIT:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            base = AutoModelForCausalLM.from_pretrained(
                BASE, quantization_config=bnb, device_map="auto"
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                BASE, torch_dtype=torch.bfloat16, device_map="auto"
            )
        adapter = Path(ADAPTER)
        if adapter.exists():
            model = PeftModel.from_pretrained(base, str(adapter)).eval()
            tok = AutoTokenizer.from_pretrained(str(adapter))
        else:
            # Không thấy adapter → chạy BASE TRẦN: output sai schema (RC lạ, JSON bịa) và panel trống.
            # Báo RÕ để không âm thầm nhầm là "model hỏng". Sửa: trỏ HONDA_ADAPTER tới thư mục adapter.
            st.warning(
                f"⚠️ Không thấy adapter `{ADAPTER}` — đang chạy BASE `{BASE}` TRẦN (chưa fine-tune). "
                "Output sẽ sai schema và panel trống. Đặt HONDA_ADAPTER trỏ tới thư mục adapter thật."
            )
            model, tok = base.eval(), AutoTokenizer.from_pretrained(BASE)
        return model, tok
    except Exception as exc:  # noqa: BLE001 - any failure → DEMO mode
        st.warning(f"Model not loaded ({type(exc).__name__}); running DEMO mode from ground truth.")
        return None


def generate(complaint: str, loaded: tuple[object, object] | None) -> tuple[str, float]:
    """Return ``(assistant_text, latency_s)`` for a raw complaint — closed-book, no tools.

    Mô hình chỉ nhận ``SYSTEM_PROMPT`` + lời than (không metadata, không tool, không telemetry).
    Khi không nạp được model → dựng câu trả lời từ ground truth (vẫn trung thực, vẫn offline).
    """
    t0 = time.perf_counter()
    if loaded is None:
        case = answer_for_complaint(complaint)
        text = assistant_content(case.think, case.resolution)
        return text, time.perf_counter() - t0
    import torch

    model, tok = loaded
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": complaint},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    text = tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return text, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Tab 1 — Customer ticket (3 bước, section tách bạch, icon note hover)
# ---------------------------------------------------------------------------


def _ticket_ref(complaint: str) -> str:
    """Mã ticket ổn định (hash lời than) — không dùng thời gian để demo tái lập được."""
    return f"HON-{abs(hash(complaint)) % 100000:05d}"


def _esc(value: object) -> str:
    """Escape tối thiểu để chèn an toàn vào HTML (cho st.markdown unsafe_allow_html)."""
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _attr(value: object) -> str:
    """Escape một chuỗi để nhét an toàn vào HTML attribute (vd title='...')."""
    return _esc(value).replace("'", "&#39;").replace('"', "&quot;")


def _pill(text: object, kind: str = "gray") -> str:
    """Một badge bo tròn có màu (red/orange/amber/green/blue/gray)."""
    return f"<span class='pill pill-{kind}'>{_esc(text)}</span>"


#: Bản dịch tiếng Việt TRUNG THỰC với thang §2.2 (ground_truth SEVERITY/PRIORITY). Code S/P giữ y gốc;
#: nếu spec thêm cấp mới mà chưa dịch, tự rớt về chữ tiếng Anh gốc (không bịa).
_SEV_VI = {
    "S1": "diện rộng (nhiều khách / cả một khu vực)",
    "S2": "một tổ hợp / một nhóm khách",
    "S3": "một khách, tính năng không dùng được, không có cách lách tạm",
    "S4": "một khách, có cách lách tạm / do môi trường / tự khỏi",
}
_PRIO_VI = {
    "P1": "ngay lập tức",
    "P2": "trong ngày (khách đã trả tiền, không có cách lách)",
    "P3": "trong 1-2 ngày",
    "P4": "tồn đọng (backlog)",
}
HELP_SEVERITY = "Mức nghiêm trọng (theo §2.2) — " + " · ".join(
    f"{k}: {_SEV_VI.get(k, v)}" for k, v in SEVERITY.items()
)
HELP_PRIORITY = "Độ ưu tiên (theo §2.2) — " + " · ".join(
    f"{k}: {_PRIO_VI.get(k, v)}" for k, v in PRIORITY.items()
)
HELP_CHURN = (
    "Rủi ro khách rời bỏ dịch vụ (theo §2.2). Thang từ thấp đến cao: "
    + " < ".join(CHURN_RISK_LEVELS)
    + ". Mỗi runbook ghi lý do của mức ở trường churn_risk.why."
)
#: Một tooltip gộp cho cả khối Định tuyến (mức nghiêm trọng + ưu tiên + rủi ro rời bỏ).
HELP_ROUTING = HELP_SEVERITY + "\n\n" + HELP_PRIORITY + "\n\n" + HELP_CHURN


def _conf_kind(conf: float | None) -> str:
    """Màu badge confidence: cao→green, vừa→amber, thấp→gray."""
    if conf is None:
        return "gray"
    return "green" if conf >= 0.7 else ("amber" if conf >= 0.5 else "gray")


def _sev_kind(sev: object) -> str:
    """Màu badge severity (S1 gắt nhất → S4 nhẹ)."""
    return {"S1": "red", "S2": "orange", "S3": "amber", "S4": "blue"}.get(str(sev), "gray")


def _prio_kind(prio: object) -> str:
    """Màu badge priority (P1 cao nhất)."""
    return {"P1": "red", "P2": "orange", "P3": "amber"}.get(str(prio), "gray")


def _churn_kind(level: object) -> str:
    """Màu badge churn risk theo từ khoá mức độ."""
    low = str(level or "").lower()
    if "high" in low:
        return "red" if "medium-high" not in low else "orange"
    if "low" in low:
        return "green"
    if "medium" in low:
        return "amber"
    return "gray"


def _like_kind(like: object) -> str:
    """Màu badge likelihood của một nhánh differential."""
    return {"high": "orange", "medium": "amber", "low": "gray"}.get(str(like).lower(), "gray")


#: CSS tùy biến (inject 1 lần) cho look "console hỗ trợ": badge, card email, stepper, key-value.
_CSS = """
<style>
  .app-head { display:flex; align-items:baseline; gap:14px; flex-wrap:wrap;
              border-bottom:1px solid rgba(255,255,255,.08); padding-bottom:10px; margin-bottom:6px; }
  .app-title { font-size:1.5rem; font-weight:800; letter-spacing:-.01em; color:#eef1f5; }
  .app-title b { color:#e4002b; }
  .app-sub { font-size:.82rem; color:#8b929c; }

  .pill { display:inline-block; padding:1px 10px; border-radius:999px; font-size:.74rem;
          font-weight:600; letter-spacing:.02em; line-height:1.7; white-space:nowrap; }
  .pill-red    { background:rgba(255,93,93,.14);  color:#ff7a7a; border:1px solid rgba(255,93,93,.35); }
  .pill-orange { background:rgba(255,159,67,.14);  color:#ffb066; border:1px solid rgba(255,159,67,.35); }
  .pill-amber  { background:rgba(255,209,102,.14); color:#ffd166; border:1px solid rgba(255,209,102,.32); }
  .pill-green  { background:rgba(46,204,113,.14);  color:#5bd98a; border:1px solid rgba(46,204,113,.32); }
  .pill-blue   { background:rgba(86,156,255,.14);  color:#7db1ff; border:1px solid rgba(86,156,255,.32); }
  .pill-gray   { background:rgba(160,170,185,.12); color:#aeb6c2; border:1px solid rgba(160,170,185,.28); }

  .verdict { display:flex; align-items:center; gap:9px; flex-wrap:wrap; margin:.1rem 0 .5rem; }
  .verdict .lbl { font-size:.7rem; text-transform:uppercase; letter-spacing:.06em; color:#8b929c; }
  .verdict .rc  { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:1.02rem;
                  font-weight:700; color:#eef1f5; }

  .k { font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; color:#8b929c; margin:.5rem 0 .25rem; }
  .kv-row { display:flex; flex-wrap:wrap; gap:20px; margin:.2rem 0; }
  .kk { font-size:.66rem; text-transform:uppercase; letter-spacing:.05em; color:#8b929c; white-space:nowrap; }
  .vv { font-size:.9rem; color:#dfe3e8; font-weight:600; }
  .kv .kk { margin-bottom:3px; }
  .route-line { margin:.25rem 0; }
  .route-line .vv { font-weight:500; }
  .routes { display:flex; flex-wrap:wrap; gap:12px 30px; margin:.5rem 0; }
  .route .kk { margin-bottom:6px; }

  .chips { display:flex; flex-wrap:wrap; gap:6px; }
  .chip  { background:rgba(125,177,255,.10); border:1px solid rgba(125,177,255,.25); color:#cdd6e4;
           padding:2px 9px; border-radius:7px; font-size:.8rem; }
  .chip-todo { background:rgba(255,209,102,.08); border-color:rgba(255,209,102,.28); color:#d8cfb4; }
  .muted { color:#8b929c; font-size:.82rem; }

  .diff { display:flex; gap:8px; align-items:flex-start; margin:5px 0; font-size:.88rem; color:#cdd2da; }
  .diff code { color:#eef1f5; }

  .email-card { border:1px solid rgba(255,255,255,.10); border-radius:10px; overflow:hidden;
                background:rgba(255,255,255,.02); margin:.3rem 0 .2rem; }
  .email-head { background:rgba(228,0,43,.12); color:#ff8a8a; padding:7px 12px; font-size:.72rem;
                font-weight:700; letter-spacing:.04em; border-bottom:1px solid rgba(255,255,255,.08); }
  .email-body { padding:11px 13px; color:#dfe3e8; font-size:.9rem; line-height:1.55; white-space:pre-wrap; }

  .step { display:flex; gap:11px; padding:7px 2px; border-bottom:1px dashed rgba(255,255,255,.07); }
  .step:last-child { border-bottom:none; }
  .step-num { flex:0 0 24px; height:24px; border-radius:50%; background:rgba(125,177,255,.16);
              color:#9cc0ff; font-weight:700; font-size:.8rem; display:flex; align-items:center;
              justify-content:center; }
  .step-act  { color:#e6e9ee; font-size:.9rem; font-weight:600; }
  .step-meta { color:#8b929c; font-size:.78rem; margin-top:1px; }
  .lead { color:#cdd2da; font-size:.92rem; line-height:1.55; }
</style>
"""


#: mermaid.js UMD bundle đóng gói cục bộ (assets/vendor/) → render OFFLINE, không phụ thuộc CDN.
_MERMAID_JS_PATH = Path(__file__).parent / "assets" / "vendor" / "mermaid.min.js"


@st.cache_data(show_spinner=False)
def _mermaid_js() -> str | None:
    """Đọc (1 lần, cache) bundle mermaid.js cục bộ; None nếu thiếu file."""
    try:
        # Đề phòng chuỗi '</script>' trong bundle làm đóng sớm thẻ <script> khi nhúng inline.
        return _MERMAID_JS_PATH.read_text(encoding="utf-8").replace("</script", "<\\/script")
    except OSError:
        return None


def _sanitize_mermaid(code: str) -> str:
    """Đổi mũi tên ``->``/``-->`` NẰM TRONG text (sau dấu ':') thành ``→``.

    mermaid v11 coi ``->`` trong nội dung Note/message là lỗi cú pháp. Chỉ phần SAU dấu ':' đầu
    tiên mới là text; phần trước (vd ``P->>E``, ``Note over E``) là cú pháp connector — giữ nguyên.
    """
    lines: list[str] = []
    for line in code.splitlines():
        if ":" in line:
            head, tail = line.split(":", 1)
            tail = tail.replace("-->", "→").replace("->", "→")
            lines.append(f"{head}:{tail}")
        else:
            lines.append(line)
    return "\n".join(lines)


def _mermaid_theme() -> str:
    """Chọn theme mermaid hợp nền Streamlit (dark/light); mặc định dark để hợp nền tối."""
    try:
        base = st.get_option("theme.base")
    except Exception:  # noqa: BLE001
        base = None
    return "default" if base == "light" else "dark"


def render_mermaid(code: str, height: int = 340) -> None:
    """Vẽ sơ đồ mermaid thành SVG OFFLINE (Streamlit không render mermaid sẵn → nhúng qua iframe).

    Nhúng bundle mermaid.js cục bộ (không cần internet) + sanitize mũi tên trong text + theme hợp
    nền. Nếu thiếu bundle hoặc parse lỗi thì rớt về hiện mã nguồn, demo không vỡ.
    """
    if not code:
        st.caption("— (không có sơ đồ)")
        return
    lib = _mermaid_js()
    if lib is None:
        st.warning("Thiếu `assets/vendor/mermaid.min.js` — hiện mã nguồn thay thế.")
        st.code(code)
        return
    diagram = json.dumps(_sanitize_mermaid(code))  # literal JS string an toàn
    theme = _mermaid_theme()
    html = f"""
    <style>
      html, body {{ margin: 0; background: transparent; }}
      #mmd {{ display: flex; justify-content: center; font-family: system-ui; color: #9aa0a6; }}
      #mmd svg {{ max-width: 100%; height: auto; }}
    </style>
    <div id="mmd">đang chờ hiển thị…</div>
    <script>{lib}</script>
    <script>
      (function () {{
        var out = document.getElementById('mmd');
        var ns = window.__esbuild_esm_mermaid_nm && window.__esbuild_esm_mermaid_nm.mermaid;
        var M = window.mermaid || (ns && (ns.default || ns));
        if (!M || !M.initialize) {{
          out.innerHTML = '<div style="color:#c00">Không tìm thấy mermaid global trong bundle.</div>';
          return;
        }}
        M.initialize({{
          startOnLoad: false, theme: '{theme}', securityLevel: 'loose', suppressErrorRendering: true,
          themeVariables: {{ 'background': 'transparent', 'fontSize': '15px' }}
        }});
        var code = {diagram};
        var done = false;
        async function tryRender() {{
          // mermaid cần phần tử ĐANG hiển thị mới đo được kích thước. Tab ẩn → clientWidth 0 → chờ.
          if (done || !document.body || document.body.clientWidth === 0) return;
          done = true;
          try {{
            var res = await M.render('mmd_svg', code);
            out.innerHTML = res.svg;
            cleanup();
          }} catch (e) {{
            var msg = (e && (e.message || e)) + '';
            // "render tree" = phần tử vẫn chưa hiển thị đủ → cho thử lại; lỗi khác → dừng hẳn.
            if (msg.indexOf('render tree') >= 0) {{ done = false; return; }}
            cleanup();
            out.innerHTML =
              '<div style="color:#c00;font-family:system-ui">render lỗi: ' + msg +
              '</div><pre style="white-space:pre-wrap;color:#777">' + code + '</pre>';
          }}
        }}
        var ro = null, iv = null;
        function cleanup() {{
          if (ro) {{ ro.disconnect(); ro = null; }}
          if (iv) {{ clearInterval(iv); iv = null; }}
        }}
        try {{ ro = new ResizeObserver(function () {{ tryRender(); }}); ro.observe(document.body); }} catch (e) {{}}
        iv = setInterval(tryRender, 300);  // dự phòng khi ResizeObserver không bắn lúc tab hiện
        tryRender();
      }})();
    </script>
    """
    components.html(html, height=height, scrolling=True)


def render_customer_panel(res: dict[str, Any]) -> None:
    """Section GỬI CHO KHÁCH — lời giải thích đời thường + email (card) + ladder (stepper)."""
    with st.container(border=True):
        st.subheader("Gửi cho khách hàng", help=HELP_CUSTOMER)
        arts = res.get("artifacts", {}) or {}

        if res.get("why_plain"):
            st.markdown(f"<div class='lead'>{_esc(res['why_plain'])}</div>", unsafe_allow_html=True)

        if arts.get("customer_email"):
            st.markdown(
                "<div class='email-card'>"
                "<div class='email-head'>HONDA CONNECTED SERVICES · SUPPORT</div>"
                f"<div class='email-body'>{_esc(arts['customer_email'])}</div></div>",
                unsafe_allow_html=True,
            )

        ladder = res.get("customer_self_service") or []
        if ladder:
            st.markdown(
                "<div class='k'>Hướng dẫn khách tự xử lý (nhanh → khó)</div>",
                unsafe_allow_html=True,
            )
            rows = "".join(
                f"<div class='step'><div class='step-num'>{_esc(s.get('tier'))}</div>"
                f"<div><div class='step-act'>{_esc(s.get('action'))}</div>"
                f"<div class='step-meta'>~{_esc(s.get('expected_time', '—'))} · "
                f"kiểm chứng: {_esc(s.get('verify', '—'))}</div></div></div>"
                for s in ladder
            )
            st.markdown(rows, unsafe_allow_html=True)


def render_internal_panel(think: str, res: dict[str, Any]) -> None:
    """Section HỖ TRỢ NỘI BỘ — chẩn đoán, differential, định tuyến (badge), artifacts."""
    with st.container(border=True):
        st.subheader("Nội bộ Honda", help=HELP_INTERNAL)
        diag = res.get("diagnosis", {}) or {}

        if think:
            st.markdown("<div class='k'>Suy luận của model</div>", unsafe_allow_html=True)
            st.info(think)

        st.markdown("<div class='k'>Differential</div>", unsafe_allow_html=True)
        diff = "".join(
            f"<div class='diff'>{_pill(d.get('likelihood', '?'), _like_kind(d.get('likelihood')))}"
            f"<span><code>{_esc(d.get('rc'))}</code> — {_esc(d.get('why'))}</span></div>"
            for d in diag.get("differential", []) or []
        )
        st.markdown(diff or "<div class='muted'>—</div>", unsafe_allow_html=True)

        cols = st.columns(2)
        with cols[0]:
            st.markdown("<div class='k'>Cue đọc được</div>", unsafe_allow_html=True)
            ev = "".join(
                f"<span class='chip'>{_esc(e)}</span>"
                for e in diag.get("evidence_in_ticket", []) or []
            )
            st.markdown(
                f"<div class='chips'>{ev or '<span class=muted>—</span>'}</div>",
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown("<div class='k'>Cần xác minh</div>", unsafe_allow_html=True)
            tc = "".join(
                f"<span class='chip chip-todo'>{_esc(c)}</span>"
                for c in diag.get("to_confirm", []) or []
            )
            st.markdown(
                f"<div class='chips'>{tc or '<span class=muted>—</span>'}</div>",
                unsafe_allow_html=True,
            )

        # Định tuyến — một dấu ? duy nhất ở tiêu đề (native help), hover ra cả 3 thang §2.2.
        churn = res.get("churn_risk") or {}
        st.markdown("<span class='k'>Định tuyến</span>", unsafe_allow_html=True, help=HELP_ROUTING)
        sev = _pill(res.get("severity") or "—", _sev_kind(res.get("severity")))
        prio = _pill(res.get("priority") or "—", _prio_kind(res.get("priority")))
        churn_pill = _pill(churn.get("level", "—"), _churn_kind(churn.get("level")))
        st.markdown(
            f"<div class='route-line'><span class='kk'>Nhóm phụ trách</span> &nbsp;"
            f"<span class='vv'>{_esc(res.get('owner_team', '—'))}</span></div>"
            "<div class='routes'>"
            f"<div class='route'><div class='kk'>Nghiêm trọng</div>{sev}</div>"
            f"<div class='route'><div class='kk'>Ưu tiên</div>{prio}</div>"
            f"<div class='route'><div class='kk'>Rủi ro rời bỏ</div>{churn_pill}</div>"
            "</div>"
            f"<div class='route-line'><span class='kk'>Thời gian xử lý dự kiến</span> &nbsp;"
            f"<span class='vv'>{_esc(res.get('eta_ttr', '—'))}</span></div>",
            unsafe_allow_html=True,
        )

        arts = res.get("artifacts", {}) or {}
        tabs = st.tabs(["Phân tích nguyên nhân", "Phiếu xử lý", "Sơ đồ"])
        tabs[0].markdown(arts.get("rca_md", "—"))
        tabs[1].markdown(arts.get("work_order_md", "—"))
        with tabs[2]:
            render_mermaid(arts.get("diagram_mermaid", ""))
            with st.expander("Mã nguồn sơ đồ"):
                st.code(arts.get("diagram_mermaid", "—"))


def render_ticket_tab(loaded: tuple[object, object] | None) -> None:
    """Tab 'Customer ticket': 3 bước — tạo ticket → SLM chẩn đoán → output thô."""
    # --- Bước 1 — khách tạo ticket ---
    with st.container(border=True):
        st.subheader("Bước 1 — Khách tạo ticket", help=HELP_TICKET)
        cols = st.columns(len(SAMPLES))
        for i, (label, sample) in enumerate(SAMPLES.items()):
            if cols[i].button(label, use_container_width=True, key=f"sample_{i}"):
                st.session_state["ticket_complaint"] = sample

        with st.form("ticket_form"):
            c1 = st.columns(3)
            customer = c1[0].text_input("Tên khách", value="Jordan P.")
            email = c1[1].text_input("Email khách", value="jordan.p@example.com")
            phone = c1[2].text_input("Số điện thoại", value="+1 415 555 0142")
            c2 = st.columns(2)
            vehicle = c2[0].text_input("Xe / trim", value="CR-V Touring")
            vin = c2[1].text_input("Số VIN", value="1HGRM4H50JL000000")
            complaint = st.text_area(
                "Nội dung khiếu nại:",
                value=st.session_state.get("ticket_complaint", SAMPLES["TCU offline"]),
                height=140,
            )
            submitted = st.form_submit_button("Gửi ticket cho SLM", type="primary")

    if not submitted:
        return
    if not complaint.strip():
        st.error("Nhập nội dung khiếu nại trước đã.")
        return

    with st.spinner("SLM đang đọc cue và chẩn đoán…"):
        text, latency = generate(complaint, loaded)
    think, res = parse_output(text)

    if res is None:
        st.error("Không parse được resolution package từ output của model.")
        st.code(text[:2000])
        return

    # --- Bước 2 — SLM chẩn đoán → tách khách / nội bộ ---
    with st.container(border=True):
        ref = _ticket_ref(complaint)
        lead = (res.get("diagnosis") or {}).get("leading_root_cause", "?")
        conf = (res.get("diagnosis") or {}).get("confidence")
        st.subheader("Bước 2 — SLM chẩn đoán", help=HELP_DIAGNOSE)
        verdict = _pill(
            f"confidence {conf:.2f}" if conf is not None else "confidence —", _conf_kind(conf)
        )
        if res.get("runbook_id"):
            verdict += _pill(res["runbook_id"], "blue")
        st.markdown(
            f"<div class='verdict'><span class='lbl'>Kết quả</span>"
            f"<span class='rc'>{_esc(lead)}</span>{verdict}</div>",
            unsafe_allow_html=True,
        )
        email_html = (
            f"<a href='mailto:{_attr(email)}' style='color:#7db1ff;text-decoration:none'>{_esc(email)}</a>"
            if email.strip()
            else "—"
        )
        phone_html = (
            f"<a href='tel:{_attr(phone)}' style='color:#7db1ff;text-decoration:none'>{_esc(phone)}</a>"
            if phone.strip()
            else "—"
        )
        st.markdown(
            "<div class='kv-row' style='margin:.1rem 0 .3rem'>"
            f"<div class='kv'><div class='kk'>Mã ticket</div><div class='vv'>{_esc(ref)}</div></div>"
            f"<div class='kv'><div class='kk'>Khách hàng</div><div class='vv'>{_esc(customer)}</div></div>"
            f"<div class='kv'><div class='kk'>Email khách</div><div class='vv'>{email_html}</div></div>"
            f"<div class='kv'><div class='kk'>Số điện thoại</div><div class='vv'>{phone_html}</div></div>"
            f"<div class='kv'><div class='kk'>Xe / VIN</div><div class='vv'>{_esc(vehicle)} · {_esc(vin)}</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"{latency:.2f}s · closed-book · liên hệ trên chỉ để CS gọi lại, không đưa vào model"
        )

        left, right = st.columns(2, gap="large")
        with left:
            render_customer_panel(res)
        with right:
            render_internal_panel(think, res)

    # --- Bước 3 — output thô ---
    with st.container(border=True):
        st.subheader("Bước 3 — Output thô của model", help=HELP_RAW)
        with st.expander("Xem <think> + JSON"):
            st.markdown("**`<think>`**")
            st.code(think or "—", language="text")
            st.markdown("**JSON resolution package**")
            st.code(json.dumps(res, ensure_ascii=False, indent=2), language="json")


# ---------------------------------------------------------------------------
# Tab 2 — SLM vs RAG (foil đối chứng — RAG là BASELINE, không phải tool của SLM)
# ---------------------------------------------------------------------------


def render_slm(text: str) -> str | None:
    """Render gọn panel SLM cho tab so sánh; trả về leading RC (để đối chiếu với RAG)."""
    think, res = parse_output(text)
    if res is None:
        st.error("Could not parse a resolution package from the model output.")
        st.code(text[:1500])
        return None
    diag = res.get("diagnosis", {})
    lead = diag.get("leading_root_cause", "?")
    conf = diag.get("confidence")
    st.markdown(f"**Reasoning → `{lead}`  ·  confidence {conf}**")
    if think:
        st.info(think)
    st.markdown("**Differential**")
    for d in diag.get("differential", []):
        st.markdown(f"- `{d.get('rc')}` ({d.get('likelihood')}): {d.get('why')}")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Evidence in ticket**")
        for e in diag.get("evidence_in_ticket", []) or ["—"]:
            st.markdown(f"- {e}")
    with cols[1]:
        st.markdown("**To confirm**")
        for c in diag.get("to_confirm", []) or ["—"]:
            st.markdown(f"- {c}")
    st.markdown(f"**Resolution — {res.get('runbook_id', '—')}**")
    b = st.columns(3)
    b[0].metric("Owner", res.get("owner_team", "—"))
    b[1].metric("Severity / Priority", f"{res.get('severity', '—')} / {res.get('priority', '—')}")
    churn = res.get("churn_risk") or {}
    b[2].metric("Churn risk", churn.get("level", "—"))
    return lead


def render_rag(complaint: str, slm_rc: str | None) -> None:
    """Render panel RAG baseline (copy nearest ticket) + cảnh báo khi bất đồng với SLM."""
    pred = RagBaseline().predict(complaint)
    rc = pred["leading_root_cause"]
    st.markdown(f"**RAG → `{rc}`  ·  sim {pred['similarity']}**")
    st.caption(f"Copied resolution from incident {pred['retrieved_incident']}")
    st.markdown(f"> _{pred['retrieved_complaint']}_")
    st.markdown(f"**Runbook:** {pred['runbook_id']}  ·  **Owner:** {pred['owner_team']}")
    if slm_rc and rc != slm_rc:
        st.error(
            f"RAG disagrees with the SLM (RAG={rc} vs SLM={slm_rc}). RAG matched a "
            "surface-similar ticket and missed the distinguishing cue."
        )


def render_compare_tab(loaded: tuple[object, object] | None) -> None:
    """Tab 'SLM vs RAG': split-screen đối chứng. RAG là baseline foil, không phải tool."""
    with st.container(border=True):
        st.subheader(
            "SLM vs RAG",
            help=(
                "RAG copy ticket gần nhất → sai RC trên ca cue-flip, không biết abstain. Đây là "
                "baseline ĐỐI CHỨNG để làm nổi bật SLM đọc cue — KHÔNG phải tool của SLM."
            ),
        )
        cols = st.columns(len(SAMPLES))
        for i, (label, sample) in enumerate(SAMPLES.items()):
            if cols[i].button(label, use_container_width=True, key=f"cmp_sample_{i}"):
                st.session_state["compare_complaint"] = sample

        complaint = st.text_area(
            "Customer complaint (raw — no error code, no logs):",
            value=st.session_state.get("compare_complaint", SAMPLES["TCU offline"]),
            height=120,
            key="compare_textarea",
        )
        diagnose = st.button("Diagnose", type="primary", key="compare_diagnose")

    if diagnose:
        text, latency = generate(complaint, loaded)
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                st.subheader("SLM (closed-book)", help=HELP_DIAGNOSE)
                slm_rc = render_slm(text)
                st.caption(f"{latency:.2f}s · offline · closed-book")
        with right:
            with st.container(border=True):
                st.subheader("RAG baseline (foil)", help="Baseline đối chứng — không phải tool.")
                render_rag(complaint, slm_rc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Honda Entitlement Resolver", page_icon="🚗", layout="wide")
st.markdown(_CSS, unsafe_allow_html=True)

loaded = load_model()
st.markdown(
    "<div class='app-head'>"
    "<div class='app-title'><b>Honda</b> Entitlement Resolver</div>"
    "</div>",
    unsafe_allow_html=True,
)

ticket_tab, compare_tab = st.tabs(["Customer ticket", "SLM vs RAG"])
with ticket_tab:
    render_ticket_tab(loaded)
with compare_tab:
    render_compare_tab(loaded)
