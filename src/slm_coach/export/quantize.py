"""Quantize the merged FP16 model: AWQ INT4 (autoawq) + GGUF Q4_K_M (llama.cpp).

This is the final deliverable of the pipeline. AWQ requires a GPU + the ``export`` extra; GGUF
conversion shells out to the llama.cpp converter (located via ``$LLAMA_CPP_DIR`` or ``PATH``).
Heavy/optional dependencies are imported lazily so this module imports cleanly without them.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from slm_coach.utils.deps import require
from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)


def load_calibration_texts(path: str | Path) -> list[str]:
    """Load domain calibration texts for AWQ from a JSONL/TXT file.

    Better AWQ quality comes from calibrating on in-domain (Vietnamese sales) text rather than
    autoawq's English default. Accepts: a ``{"text": ...}`` field, a ``messages`` list (contents
    joined), a ``reference`` field, a bare JSON string, or a plain text line.

    Args:
        path: Path to the calibration file.

    Returns:
        A list of calibration strings.
    """
    texts: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            texts.append(line)
            continue
        if isinstance(obj, str):
            texts.append(obj)
        elif isinstance(obj, dict) and obj.get("text"):
            texts.append(str(obj["text"]))
        elif isinstance(obj, dict) and isinstance(obj.get("messages"), list):
            texts.append("\n".join(m.get("content", "") for m in obj["messages"]))
        elif isinstance(obj, dict) and obj.get("reference"):
            texts.append(str(obj["reference"]))
    return [t for t in texts if t.strip()]


def quantize_awq(
    fp16_dir: str | Path,
    output_dir: str | Path,
    *,
    calib_data: Sequence[str] | None = None,
) -> Path:
    """Quantize an FP16 model to AWQ INT4 using ``autoawq``.

    Args:
        fp16_dir: Directory of the merged FP16 model.
        output_dir: Destination directory for the AWQ model.
        calib_data: Optional in-domain calibration texts (see :func:`load_calibration_texts`).
            When omitted, autoawq's default calibration set is used.

    Returns:
        Path to the AWQ output directory.
    """
    awq = require("awq", "export")
    transformers = require("transformers", "train")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = awq.AutoAWQForCausalLM.from_pretrained(str(fp16_dir))
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(fp16_dir))
    quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
    quantize_kwargs = {"calib_data": list(calib_data)} if calib_data else {}
    model.quantize(tokenizer, quant_config=quant_config, **quantize_kwargs)
    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info(
        "AWQ INT4 export complete",
        extra={"output": str(out), "calib": "domain" if calib_data else "default"},
    )
    return out


def _find_llama_cpp_script(name: str) -> Path | None:
    """Locate a llama.cpp helper script via ``$LLAMA_CPP_DIR`` or ``PATH``."""
    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidate = Path(env_dir) / name
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    return Path(found) if found else None


def quantize_gguf(fp16_dir: str | Path, output_dir: str | Path, *, quant: str = "Q4_K_M") -> Path:
    """Convert an FP16 model to GGUF and quantize (default ``Q4_K_M``) via llama.cpp.

    Requires the llama.cpp tools: ``convert_hf_to_gguf.py`` and the ``llama-quantize`` binary,
    located via ``$LLAMA_CPP_DIR`` or ``PATH``.

    Args:
        fp16_dir: Directory of the merged FP16 model.
        output_dir: Destination directory for the GGUF model.
        quant: GGUF quantization type.

    Returns:
        Path to the quantized GGUF file.

    Raises:
        RuntimeError: If the llama.cpp tools cannot be located.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    converter = _find_llama_cpp_script("convert_hf_to_gguf.py")
    quantizer = _find_llama_cpp_script("llama-quantize") or _find_llama_cpp_script(
        "llama-quantize.exe"
    )
    if converter is None or quantizer is None:
        raise RuntimeError(
            "llama.cpp tools not found. Set $LLAMA_CPP_DIR to your llama.cpp checkout "
            "(needs convert_hf_to_gguf.py and llama-quantize)."
        )

    f16_gguf = out / "model-f16.gguf"
    quantized = out / f"model-{quant.lower()}.gguf"
    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(fp16_dir),
            "--outfile",
            str(f16_gguf),
            "--outtype",
            "f16",
        ],
        check=True,
    )
    subprocess.run([str(quantizer), str(f16_gguf), str(quantized), quant], check=True)
    logger.info("GGUF export complete", extra={"output": str(quantized), "quant": quant})
    return quantized
