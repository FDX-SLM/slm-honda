"""Base-model loading, adapter attachment, and merging.

Loads the base SLM via Unsloth + Flash Attention 2 (when a GPU and Unsloth are present),
otherwise plain transformers; attaches a LoRA (T1) or 4-bit QLoRA (T2) adapter; and merges the
adapter into FP16 ``safetensors`` for export. Heavy GPU dependencies (``torch``, ``unsloth``,
``peft``) are imported lazily via :func:`slm_coach.utils.deps.require`, so this module imports
cleanly on a machine without them and raises an actionable error only when actually used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.utils.deps import is_installed, require
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_coach.config import LoRAConfig, ModelRuntimeConfig, QuantConfig

logger = get_logger(__name__)


@dataclass
class LoadedModel:
    """A loaded model + tokenizer, tagged with how it was loaded."""

    model: Any
    tokenizer: Any
    via_unsloth: bool


def has_gpu() -> bool:
    """Return whether a CUDA device is available (False if torch is absent)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def supports_assistant_mask(tokenizer: Any) -> bool:
    """Whether the tokenizer's chat template marks assistant tokens for masking.

    TRL's ``assistant_only_loss`` (train-on-responses-only) needs the chat template to contain
    ``{% generation %}`` markers. When absent, enabling it would crash — callers should disable
    masking with a warning instead.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    return "generation" in template


def precision_kwargs() -> dict[str, bool]:
    """Return ``bf16``/``fp16`` TrainingArguments flags suited to the current hardware.

    Trainers (TRL/transformers) otherwise default to bf16, which CPUs and pre-Ampere GPUs
    reject. On CPU both are ``False`` (fp32); on a bf16-capable GPU bf16 is used, else fp16.
    """
    try:
        import torch

        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return {"bf16": True, "fp16": False}
            return {"bf16": False, "fp16": True}
    except ImportError:
        pass
    return {"bf16": False, "fp16": False}


def should_use_unsloth(model_cfg: ModelRuntimeConfig) -> bool:
    """Whether to use Unsloth: requested in config, installed, and a GPU is present."""
    return bool(model_cfg.use_unsloth and has_gpu() and is_installed("unsloth"))


def _resolve_dtype(name: str) -> Any:
    """Map a dtype name (``"bfloat16"`` ...) to a ``torch`` dtype."""
    torch = require("torch", "train")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


def _target_modules(lora_cfg: LoRAConfig) -> Any:
    """Return the LoRA target-modules spec (a string like ``all-linear`` or a list)."""
    return lora_cfg.target_modules


def load_base_model(
    model_name: str,
    model_cfg: ModelRuntimeConfig,
    quant_cfg: QuantConfig,
) -> LoadedModel:
    """Load the base model and tokenizer (Unsloth + Flash Attention 2, or transformers).

    Args:
        model_name: HF model id or local path.
        model_cfg: Runtime options (sequence length, attention impl, dtype).
        quant_cfg: Quantization options (4-bit QLoRA for T2).

    Returns:
        A :class:`LoadedModel` (model, tokenizer, and the load path used).
    """
    if should_use_unsloth(model_cfg):
        unsloth = require("unsloth", "gpu")
        model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=model_cfg.max_seq_len,
            load_in_4bit=quant_cfg.load_in_4bit,
            dtype=None,
        )
        logger.info("Loaded base via Unsloth", extra={"model": model_name})
        return LoadedModel(model=model, tokenizer=tokenizer, via_unsloth=True)

    transformers = require("transformers", "train")
    require("torch", "train")  # ensure torch present for model weights
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    attn = model_cfg.attn_implementation
    if attn == "flash_attention_2" and not is_installed("flash_attn"):
        logger.warning("flash-attn not installed; falling back to sdpa attention")
        attn = "sdpa"
    kwargs: dict[str, Any] = {
        "dtype": _resolve_dtype(model_cfg.dtype),
        "attn_implementation": attn,
    }
    if quant_cfg.load_in_4bit:
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_resolve_dtype(model_cfg.dtype),
            bnb_4bit_use_double_quant=True,
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Loaded base via transformers", extra={"model": model_name})
    return LoadedModel(model=model, tokenizer=tokenizer, via_unsloth=False)


def attach_adapter(loaded: LoadedModel, lora_cfg: LoRAConfig, *, quantized: bool) -> Any:
    """Attach a fresh LoRA (T1) or QLoRA (T2) adapter to the loaded model.

    Args:
        loaded: The loaded base model.
        lora_cfg: LoRA hyperparameters.
        quantized: Whether the base was loaded in 4-bit (QLoRA).

    Returns:
        The PEFT-wrapped model (also stored back on ``loaded.model``).
    """
    if loaded.via_unsloth:
        unsloth = require("unsloth", "gpu")
        loaded.model = unsloth.FastLanguageModel.get_peft_model(
            loaded.model,
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=_target_modules(lora_cfg),
            bias=lora_cfg.bias,
            use_gradient_checkpointing="unsloth",
        )
        return loaded.model

    peft = require("peft", "train")
    model = loaded.model
    if quantized:
        model = peft.prepare_model_for_kbit_training(model)
    lora_config = peft.LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=_target_modules(lora_cfg),
        bias=lora_cfg.bias,
        task_type="CAUSAL_LM",
    )
    loaded.model = peft.get_peft_model(model, lora_config)
    return loaded.model


def prepare_peft_model(
    model_name: str,
    model_cfg: ModelRuntimeConfig,
    quant_cfg: QuantConfig,
    lora_cfg: LoRAConfig,
    *,
    existing_adapter: str | Path | None = None,
) -> LoadedModel:
    """Load the base model and attach a trainable adapter (fresh or continued).

    Args:
        model_name: Base model id or path.
        model_cfg: Runtime options.
        quant_cfg: Quantization options.
        lora_cfg: LoRA hyperparameters (used only for a fresh adapter).
        existing_adapter: If set, continue this adapter (curriculum chaining) instead of
            attaching a fresh one.

    Returns:
        A :class:`LoadedModel` whose ``model`` is PEFT-wrapped and trainable.
    """
    loaded = load_base_model(model_name, model_cfg, quant_cfg)
    if existing_adapter is not None:
        peft = require("peft", "train")
        loaded.model = peft.PeftModel.from_pretrained(
            loaded.model, str(existing_adapter), is_trainable=True
        )
        logger.info("Continued existing adapter", extra={"adapter": str(existing_adapter)})
    else:
        attach_adapter(loaded, lora_cfg, quantized=quant_cfg.load_in_4bit)
    return loaded


def save_checkpoint(model: Any, tokenizer: Any, path: str | Path) -> Path:
    """Save the adapter, tokenizer, and config to ``path``.

    Args:
        model: The PEFT model.
        tokenizer: The tokenizer.
        path: Destination directory.

    Returns:
        The destination directory path.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.info("Saved checkpoint", extra={"path": str(out)})
    return out


def merge_to_fp16(adapter_path: str | Path, output_path: str | Path) -> Path:
    """Merge a LoRA adapter into the base and save FP16 ``safetensors`` + tokenizer.

    Args:
        adapter_path: Directory containing the trained adapter.
        output_path: Destination directory for the merged FP16 model.

    Returns:
        The output directory path.
    """
    peft = require("peft", "train")
    torch = require("torch", "train")
    transformers = require("transformers", "train")

    model = peft.AutoPeftModelForCausalLM.from_pretrained(str(adapter_path), dtype=torch.float16)
    model = model.merge_and_unload()
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), safe_serialization=True)
    transformers.AutoTokenizer.from_pretrained(str(adapter_path)).save_pretrained(str(out))
    logger.info("Merged adapter to FP16", extra={"output": str(out)})
    return out
