"""Lắp checkpoint FULL đã merge để vLLM serve qua path multimodal (đã hoàn chỉnh).

Merge adapter dpo_qwen vào weights TEXT của Qwen3.5-9B, GIỮ NGUYÊN vision + MTP + config multimodal
→ /workspace/honda-merged-full. Cần thiết vì (1) vLLM không áp được LoRA lên lớp linear-attention của
qwen3_5, (2) path text-only của vLLM 0.24 chưa hoàn chỉnh. Chạy: uv run python webdemo/scripts/assemble_merged.py
"""

from __future__ import annotations

import glob
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = os.environ.get("HONDA_BASE", "Qwen/Qwen3.5-9B")
ADP = os.environ.get("HONDA_ADAPTER", "/workspace/honda-adapters/dpo_qwen")
OUT = os.environ.get("HONDA_MERGED_FULL", "/workspace/honda-merged-full")


def _remap(k: str) -> str:
    """Key text của AutoModelForCausalLM (``model.*``) → layout multimodal (``model.language_model.*``)."""
    if k == "lm_head.weight":
        return k
    if k.startswith("model.") and not k.startswith("model.language_model."):
        return "model.language_model." + k[len("model.") :]
    return k


def main() -> None:
    if Path(OUT, "model.safetensors.index.json").exists():
        print(f"[assemble] {OUT} đã có — bỏ qua.")
        return
    t = time.time()
    print("[assemble] 1) merge adapter vào text (CPU)…", flush=True)
    text = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")
    text = PeftModel.from_pretrained(text, ADP).merge_and_unload()
    msd = {_remap(k): v for k, v in text.state_dict().items()}
    del text

    print(f"[assemble] 2) load FULL multimodal model (CPU)… ({time.time() - t:.0f}s)", flush=True)
    try:
        from transformers import AutoModelForImageTextToText as FullCls
    except Exception:  # noqa: BLE001
        from transformers import AutoModel as FullCls
    full = FullCls.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")

    missing, unexpected = full.load_state_dict(msd, strict=False)
    print(
        f"[assemble]    missing(giữ từ base)={len(missing)} | unexpected={len(unexpected)}",
        flush=True,
    )
    if len(unexpected) != 0:
        sys.exit(f"[assemble] LỖI: {len(unexpected)} key không khớp — merge sai layout.")

    print(f"[assemble] 3) save → {OUT} ({time.time() - t:.0f}s)…", flush=True)
    Path(OUT).mkdir(parents=True, exist_ok=True)
    full.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
    AutoTokenizer.from_pretrained(ADP).save_pretrained(OUT)
    snaps = glob.glob(
        str(
            Path(os.environ.get("HF_HOME", "/workspace/.hf_home"))
            / "hub/models--Qwen--Qwen3.5-9B/snapshots/*"
        )
    )
    if snaps:
        for f in glob.glob(f"{snaps[0]}/*.json") + glob.glob(f"{snaps[0]}/*.jinja"):
            n = os.path.basename(f)
            dst = Path(OUT) / n
            if not dst.exists() and n not in ("model.safetensors.index.json", "config.json"):
                shutil.copy(f, dst)

    # vLLM serve qua path MULTIMODAL cần processor config, nhưng base tải qua
    # AutoModel/AutoTokenizer KHÔNG kéo các file này về snapshot → tải trực tiếp từ hub.
    from huggingface_hub import hf_hub_download

    for n in ("preprocessor_config.json", "video_preprocessor_config.json"):
        dst = Path(OUT) / n
        if not dst.exists():
            try:
                shutil.copy(hf_hub_download(BASE, n, token=os.environ.get("HF_TOKEN")), dst)
            except Exception as e:  # noqa: BLE001
                print(f"[assemble] CẢNH BÁO: không tải được {n}: {e}", flush=True)
    print(f"[assemble] DONE {time.time() - t:.0f}s → {OUT}", flush=True)


if __name__ == "__main__":
    main()
