"""Merge a trained LoRA adapter into the base model -> FP16 ``safetensors``.

This is the first export step (the merged FP16 model feeds the quantizers in
:mod:`slm_coach.export.quantize`). Delegates to :func:`slm_coach.training.model.merge_to_fp16`,
which imports ``torch``/``peft`` lazily — so this module imports cleanly without a GPU.
"""

from __future__ import annotations

from pathlib import Path

from slm_coach.training.model import merge_to_fp16
from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)


def merge_lora_to_fp16(checkpoint: str | Path, output_dir: str | Path) -> Path:
    """Merge a LoRA/QLoRA adapter into the base and save FP16 ``safetensors``.

    Args:
        checkpoint: Path to the trained adapter checkpoint.
        output_dir: Destination directory for the merged FP16 model.

    Returns:
        Path to the merged FP16 model directory.
    """
    logger.info("Merging adapter to FP16", extra={"checkpoint": str(checkpoint)})
    return merge_to_fp16(checkpoint, output_dir)
