"""Supervised fine-tuning via TRL ``SFTTrainer`` (train-on-responses-only + multi-turn masking).

Wraps TRL's ``SFTTrainer`` with complete, resumable checkpointing (best + last + ``meta.json``),
the eval-during-training callback, and early stopping. Multi-turn masking computes loss on
*every* assistant turn via TRL's ``assistant_only_loss`` over the chat template (the offline
equivalent is :func:`slm_coach.data.formatting.iter_assistant_spans`).

Heavy training deps (``torch``, ``trl``) are imported lazily so this module imports without a
GPU; the no-GPU ``dry_run`` path exercises data loading/formatting/mixing and the plan only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.data.formatting import to_sft_dataset
from slm_coach.data.loader import load_gold_cases, load_records
from slm_coach.data.mixture import mix_single_multi
from slm_coach.tracking import init_tracking
from slm_coach.training.callbacks import EvalDuringTraining, eval_metric_key, write_meta_json
from slm_coach.training.model import (
    precision_kwargs,
    prepare_peft_model,
    save_checkpoint,
    supports_assistant_mask,
)
from slm_coach.utils.deps import require
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_coach.config import SFTFileConfig

logger = get_logger(__name__)


def load_gold_subset(config: SFTFileConfig) -> list[dict[str, Any]]:
    """Load the eval-during-training gold subset as ``{prompt, reference, mode}`` dicts.

    Returns an empty list if the gold file is absent (the callback then no-ops).
    """
    edt = config.eval_during_training
    if not edt.enabled:
        return []
    gold_path = Path(edt.gold_subset)
    if not gold_path.is_file():
        logger.warning(
            "Gold subset missing; eval-during-training disabled", extra={"path": str(gold_path)}
        )
        return []
    cases = load_gold_cases(gold_path)[: edt.subset_size]
    return [
        {
            "prompt": c.prompt,
            "reference": c.reference,
            "mode": c.mode,
        }
        for c in cases
    ]


def build_generate_fn(model: Any, tokenizer: Any, max_new_tokens: int) -> Any:
    """Return a ``prompts -> answers`` closure backed by ``model.generate`` (lazy torch)."""

    def _generate(prompts: list[list[dict[str, str]]]) -> list[str]:
        torch = require("torch", "train")
        outputs: list[str] = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id
                )
            new_tokens = gen[0][enc["input_ids"].shape[1] :]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return outputs

    return _generate


def _build_sft_args(config: SFTFileConfig, output_dir: Path, *, assistant_only_loss: bool) -> Any:
    """Construct a TRL ``SFTConfig`` from the validated config (complete checkpointing)."""
    trl = require("trl", "train")
    sft = config.sft
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": sft.epochs,
        "per_device_train_batch_size": sft.batch_size,
        "gradient_accumulation_steps": sft.grad_accum,
        "learning_rate": sft.lr,
        "warmup_ratio": sft.warmup_ratio,
        "weight_decay": sft.weight_decay,
        "lr_scheduler_type": sft.lr_scheduler_type,
        "max_length": sft.max_seq_len,
        "logging_steps": sft.logging_steps,
        "save_strategy": "steps",
        "save_steps": sft.save_steps,
        "eval_strategy": "steps",
        "eval_steps": sft.eval_steps,
        "save_total_limit": sft.save_total_limit,
        "load_best_model_at_end": sft.load_best_model_at_end,
        "metric_for_best_model": sft.metric_for_best_model,
        "greater_is_better": sft.greater_is_better,
        "assistant_only_loss": assistant_only_loss,
        "packing": sft.packing,
        "seed": config.seed,
        "report_to": [],
        **precision_kwargs(),
    }
    if sft.max_steps is not None:
        kwargs["max_steps"] = sft.max_steps
    return trl.SFTConfig(**kwargs)


def _build_eval_judges(config: SFTFileConfig) -> list[Any] | None:
    """Build judges for eval-during-training when ``use_judge`` is on (else ``None``).

    Falls back to ``None`` (proxy metric) if judges can't be constructed (e.g. missing keys),
    so training never crashes on the eval path.
    """
    edt = config.eval_during_training
    if not edt.use_judge:
        return None
    try:
        from slm_coach.eval.judge import build_judges

        return build_judges(edt.judges, edt.judge_models)
    except Exception as exc:  # noqa: BLE001 - fall back to the proxy metric, never crash training
        logger.warning("Could not build judges; using proxy metric", extra={"error": str(exc)})
        return None


def run_sft_core(
    config: SFTFileConfig,
    train_records: list,
    gold_records: list[dict[str, Any]],
    output_dir: Path,
    *,
    init_from: str,
    existing_adapter: str | Path | None = None,
    reasoning_thinking: bool = False,
    resume: str | Path | None = None,
    stage_name: str | None = None,
) -> Path:
    """Train one SFT pass (shared by T1 and each curriculum stage) and save best + last.

    Args:
        config: Validated SFT config.
        train_records: Canonical records for this pass.
        gold_records: Eval-during-training subset.
        output_dir: Directory for this pass's checkpoints.
        init_from: Base model id/path to load.
        existing_adapter: Continue this adapter instead of attaching a fresh one (chaining).
        reasoning_thinking: Whether reasoning records fold a ``<think>`` block.
        resume: Optional checkpoint to resume from.
        stage_name: Optional curriculum stage name (for tracking/logging).

    Returns:
        Path to the ``best`` checkpoint for this pass.
    """
    trl = require("trl", "train")
    from transformers import EarlyStoppingCallback

    dataset = to_sft_dataset(train_records, reasoning_thinking=reasoning_thinking)
    loaded = prepare_peft_model(
        init_from, config.model, config.quant, config.lora, existing_adapter=existing_adapter
    )
    run_name = config.run_name + (f":{stage_name}" if stage_name else "")
    tracker = init_tracking(config, run_name=run_name)

    # Multi-turn masking needs a chat template with generation markers; degrade gracefully.
    assistant_only = config.sft.multiturn_masking
    if assistant_only and not supports_assistant_mask(loaded.tokenizer):
        logger.warning(
            "Chat template lacks generation markers; disabling assistant_only_loss "
            "(multi-turn masking). Provide a template with {% generation %} to enable it."
        )
        assistant_only = False

    eval_size = min(len(dataset), config.eval_during_training.subset_size) or 1
    eval_dataset = dataset.select(range(min(eval_size, len(dataset))))

    callbacks: list[Any] = []
    if gold_records:
        generate_fn = build_generate_fn(
            loaded.model, loaded.tokenizer, config.eval_during_training.max_new_tokens
        )
        callbacks.append(
            EvalDuringTraining(
                config.eval_during_training,
                gold_records=gold_records,
                generate_fn=generate_fn,
                judges=_build_eval_judges(config),
                weights=config.eval_during_training.rubric_weights or None,
                metric_name=config.sft.metric_for_best_model,
                tracker=tracker,
            )
        )
    callbacks.append(
        EarlyStoppingCallback(early_stopping_patience=config.sft.early_stopping_patience)
    )

    trainer = trl.SFTTrainer(
        model=loaded.model,
        args=_build_sft_args(config, output_dir, assistant_only_loss=assistant_only),
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=loaded.tokenizer,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    best_dir = output_dir / "best"
    last_dir = output_dir / "last"
    save_checkpoint(trainer.model, loaded.tokenizer, best_dir)
    save_checkpoint(trainer.model, loaded.tokenizer, last_dir)
    best_metric = getattr(trainer.state, "best_metric", None)
    metric_key = eval_metric_key(config.sft.metric_for_best_model)
    meta_metrics = {metric_key: best_metric} if best_metric is not None else {}
    write_meta_json(
        best_dir,
        config=config.model_dump(),
        seed=config.seed,
        data_version=config.data.lang,
        metrics=meta_metrics,
    )
    write_meta_json(
        last_dir, config=config.model_dump(), seed=config.seed, data_version=config.data.lang
    )
    _export_metrics(trainer, config, output_dir)
    tracker.close()
    logger.info("SFT pass complete", extra={"best": str(best_dir), "stage": stage_name})
    return best_dir


def _export_metrics(trainer: Any, config: SFTFileConfig, output_dir: Path) -> None:
    """Persist trainer state and write the metric tables + curve charts (never blocks a run)."""
    rep = config.reporting
    if not (rep.tables or rep.plots):
        return
    try:
        from slm_coach.reporting import export_training_artifacts

        trainer.save_state()  # writes the cumulative trainer_state.json to output_dir
        export_training_artifacts(output_dir, make_tables=rep.tables, make_plots=rep.plots)
    except Exception as exc:  # noqa: BLE001 - artifact generation must never fail a good run
        logger.warning("Could not export training metrics", extra={"error": str(exc)})


def run_sft_training(
    config: SFTFileConfig,
    *,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Run single-stage SFT (T1) and return the best checkpoint path.

    Args:
        config: Validated SFT config.
        resume: Optional checkpoint to resume from (optimizer state intact).
        dry_run: If True, load/format/mix data and log the plan without launching training.

    Returns:
        Path to the best checkpoint directory.
    """
    output_dir = Path(config.output_dir) / config.run_name
    data = load_records(config.data.dir, ("sft", "reasoning"), config.data.keep_audit_status)
    sft_records = data["sft"]
    reasoning_records = data["reasoning"]
    mixed = (
        mix_single_multi(
            sft_records,
            multi_turn=config.sft.mixture.multi_turn,
            single=config.sft.mixture.single,
            seed=config.seed,
        )
        if sft_records
        else []
    )
    train_records = list(mixed) + list(reasoning_records)
    gold_records = load_gold_subset(config)

    logger.info(
        "SFT plan",
        extra={
            "run_name": config.run_name,
            "n_sft": len(sft_records),
            "n_reasoning": len(reasoning_records),
            "n_train": len(train_records),
            "multiturn_masking": config.sft.multiturn_masking,
            "gold_subset": len(gold_records),
        },
    )
    if dry_run:
        logger.info(
            "Dry run: skipping model load + training", extra={"output": str(output_dir / "best")}
        )
        return output_dir / "best"

    return run_sft_core(
        config, train_records, gold_records, output_dir, init_from=config.model_name, resume=resume
    )
