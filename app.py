"""Streamlit demo cho Honda Entitlement Resolver (PoC §8) — split-screen SLM vs RAG.

Chạy:  uv run streamlit run app.py

Nhập lời than thô của khách → SLM closed-book đọc cue → suy ra RC (differential + confidence +
to_confirm) → nhả runbook (business fields) + artifacts. Cột phải: RAG "copy nearest ticket" để
đối chứng (sai RC trên ca cue-flip, không biết abstain). Money-shot: SLM đọc cue đúng, RAG chép sai.

Env (không hardcode path máy cá nhân):
    HONDA_BASE      base model (mặc định Qwen/Qwen3.5-9B)
    HONDA_ADAPTER   thư mục adapter SFT/DPO (mặc định checkpoints/sft/best)
    HONDA_4BIT      nạp base 4-bit? "true"=QLoRA (mặc định true)
Nếu KHÔNG nạp được model (thiếu GPU/adapter) → chạy DEMO mode từ ground truth (vẫn trung thực).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from slm_coach.datagen.core import answer_for_complaint, assistant_content
from slm_coach.eval.rag import RagBaseline
from slm_coach.ground_truth import SYSTEM_PROMPT
from slm_coach.oracle import parse_output

BASE = os.environ.get("HONDA_BASE", "Qwen/Qwen3.5-9B")
ADAPTER = os.environ.get("HONDA_ADAPTER", "checkpoints/sft/best")
FOUR_BIT = os.environ.get("HONDA_4BIT", "true").strip().lower() not in ("false", "0", "no")
MAX_NEW_TOKENS = int(os.environ.get("HONDA_MAX_NEW_TOKENS", "640"))

SAMPLES = {
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
    "Ambiguous (abstain)": "It just doesn't work. I paid for it and nothing happens.",
}


@st.cache_resource
def load_model() -> tuple[object, object] | None:
    """Nạp base + adapter; trả None nếu thiếu GPU/torch/adapter (→ DEMO mode từ ground truth)."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if FOUR_BIT:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
        else:
            base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
        adapter = Path(ADAPTER)
        if adapter.exists():
            model = PeftModel.from_pretrained(base, str(adapter)).eval()
            tok = AutoTokenizer.from_pretrained(str(adapter))
        else:
            model, tok = base.eval(), AutoTokenizer.from_pretrained(BASE)
        return model, tok
    except Exception as exc:  # noqa: BLE001 - any failure → DEMO mode
        st.warning(f"Model not loaded ({type(exc).__name__}); running DEMO mode from ground truth.")
        return None


def generate(complaint: str, loaded: tuple[object, object] | None) -> tuple[str, float]:
    """Return (assistant_text, latency_s). Uses the model if loaded, else the ground-truth demo."""
    t0 = time.perf_counter()
    if loaded is None:
        case = answer_for_complaint(complaint)
        text = assistant_content(case.think, case.resolution)
        return text, time.perf_counter() - t0
    import torch

    model, tok = loaded
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": complaint}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return text, time.perf_counter() - t0


def render_slm(text: str) -> str | None:
    """Render the SLM panel (think + differential + evidence + runbook fields + artifacts)."""
    think, res = parse_output(text)
    if res is None:
        st.error("Could not parse a resolution package from the model output.")
        st.code(text[:1500])
        return None
    diag = res.get("diagnosis", {})
    lead = diag.get("leading_root_cause", "?")
    conf = diag.get("confidence")
    st.markdown(f"### 🧠 Reasoning → **{lead}**  ·  confidence **{conf}**")
    if think:
        st.info(think)
    st.markdown("**Differential**")
    for d in diag.get("differential", []):
        st.markdown(f"- `{d.get('rc')}` ({d.get('likelihood')}): {d.get('why')}")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Evidence in ticket** (cues the model read)")
        for e in diag.get("evidence_in_ticket", []) or ["—"]:
            st.markdown(f"- 🔎 {e}")
    with cols[1]:
        st.markdown("**To confirm** (human checks)")
        for c in diag.get("to_confirm", []) or ["—"]:
            st.markdown(f"- ☐ {c}")

    st.divider()
    st.markdown(f"### 📋 Resolution — {res.get('runbook_id', '—')}")
    b = st.columns(3)
    b[0].metric("Owner", res.get("owner_team", "—"))
    b[1].metric("Severity / Priority", f"{res.get('severity', '—')} / {res.get('priority', '—')}")
    churn = res.get("churn_risk") or {}
    b[2].metric("Churn risk", churn.get("level", "—"))
    st.markdown(f"**ETA:** {res.get('eta_ttr', '—')}  ·  **Escalation:** {res.get('escalation', '—')}")
    comp = res.get("compensation") or {}
    if comp:
        st.markdown(f"**Compensation:** {comp.get('offer', '—')} (proactive: {comp.get('proactive')})")
    arts = res.get("artifacts", {}) or {}
    tabs = st.tabs(["RCA", "Work order", "Customer email", "Diagram"])
    tabs[0].markdown(arts.get("rca_md", "—"))
    tabs[1].markdown(arts.get("work_order_md", "—"))
    tabs[2].markdown(arts.get("customer_email", "—"))
    tabs[3].code(arts.get("diagram_mermaid", "—"))
    return lead


def render_rag(complaint: str, slm_rc: str | None) -> None:
    """Render the RAG baseline panel (copy nearest ticket) + a disagreement callout."""
    pred = RagBaseline().predict(complaint)
    rc = pred["leading_root_cause"]
    st.markdown(f"### 🔁 RAG → **{rc}**  ·  sim **{pred['similarity']}**")
    st.caption(f"Copied resolution from incident {pred['retrieved_incident']}")
    st.markdown(f"> _{pred['retrieved_complaint']}_")
    st.markdown(f"**Runbook:** {pred['runbook_id']}  ·  **Owner:** {pred['owner_team']}")
    if slm_rc and rc != slm_rc:
        st.error(
            f"⚠️ RAG disagrees with the SLM (RAG={rc} vs SLM={slm_rc}). RAG matched a "
            "surface-similar ticket and missed the distinguishing cue."
        )


st.set_page_config(page_title="Honda Entitlement Resolver", page_icon="🚗", layout="wide")
st.title("🚗 Honda Entitlement Resolver — SLM vs RAG")
st.caption("Closed-book SLM reads the complaint's cues → root cause + runbook. Offline · <1.5s target.")

loaded = load_model()
mode = "DEMO (ground truth)" if loaded is None else f"SLM ({Path(ADAPTER).name})"
st.caption(f"Mode: **{mode}**  ·  Base: `{BASE}`")

cols = st.columns(len(SAMPLES))
for i, (label, sample) in enumerate(SAMPLES.items()):
    if cols[i].button(label, use_container_width=True):
        st.session_state["complaint"] = sample

complaint = st.text_area(
    "Customer complaint (raw — no error code, no logs):",
    value=st.session_state.get("complaint", SAMPLES["TCU offline"]),
    height=120,
)

if st.button("Diagnose", type="primary"):
    text, latency = generate(complaint, loaded)
    left, right = st.columns(2)
    with left:
        st.subheader("SLM (closed-book)")
        slm_rc = render_slm(text)
        st.caption(f"⚡ {latency:.2f}s · offline · closed-book")
    with right:
        st.subheader("RAG baseline (foil)")
        render_rag(complaint, slm_rc)

st.divider()
st.caption(
    "Honesty: the SLM reasons only from cues in the complaint — it never invents telemetry "
    "(no timestamps, no 'record found'); when there is no distinguishing cue it abstains and "
    "routes to a human. Prototype on synthetic data; productionize on real Honda data (PM2.0)."
)
