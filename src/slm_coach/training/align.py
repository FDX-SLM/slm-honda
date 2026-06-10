"""Alignment (T3): DPO or ORPO via TRL, selected by config.

ORPO is monolithic (trains a fresh adapter on the base in a single stage). DPO continues the
SFT adapter and requires an SFT checkpoint as its starting point. The method is chosen from
``config.align.method`` — never hardcoded. Heavy deps are imported lazily; ``dry_run`` loads
the preference data and logs the plan without a GPU.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from slm_coach.data.formatting import to_preference_dataset
from slm_coach.data.loader import load_records
from slm_coach.tracking import init_tracking
from slm_coach.training.callbacks import write_meta_json
from slm_coach.training.model import precision_kwargs, prepare_peft_model, save_checkpoint
from slm_coach.utils.deps import require
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_coach.config import AlignFileConfig

logger = get_logger(__name__)


def run_alignment(
    config: AlignFileConfig,
    *,
    sft_checkpoint: str | Path | None = None,
    resume: str | Path | None = None,
    dry_run: bool = False,
) -> Path:
    """Run DPO or ORPO alignment and return the best checkpoint path.

    Args:
        config: Validated alignment config (``align.method`` selects DPO vs ORPO).
        sft_checkpoint: SFT starting point — required for DPO, ignored for ORPO. Overrides
            ``config.sft_checkpoint`` when provided.
        resume: Optional checkpoint to resume from.
        dry_run: If True, load preference data and log the plan without launching training.

    Returns:
        Path to the best aligned checkpoint directory.

    Raises:
        ValueError: If DPO is selected without an SFT checkpoint.
    """
    method = config.align.method
    start = sft_checkpoint or config.sft_checkpoint
    if method == "dpo" and not start:
        raise ValueError("DPO requires an SFT checkpoint as its starting point (--sft-checkpoint).")

    output_dir = Path(config.output_dir) / config.run_name
    data = load_records(config.data.dir, ("preference",), config.data.keep_audit_status)
    preferences = data["preference"]

    logger.info(
        "Alignment plan",
        extra={"method": method, "n_pref": len(preferences), "start": str(start)},
    )
    if dry_run:
        logger.info(
            "Dry run: skipping model load + training", extra={"output": str(output_dir / "best")}
        )
        return output_dir / "best"

    dataset = to_preference_dataset(preferences)
    return _run_align_core(config, dataset, output_dir, method=method, start=start, resume=resume)


def _run_align_core(
    config: AlignFileConfig,
    dataset: Any,
    output_dir: Path,
    *,
    method: str,
    start: str | Path | None,
    resume: str | Path | None,
) -> Path:
    """Build and run the DPO/ORPO trainer, then save best + last + ``meta.json``."""
    require("torch", "train")
    trl = require("trl", "train")
    from transformers import EarlyStoppingCallback

    # ORPO trains a fresh adapter on the base; DPO continues the SFT adapter.
    existing_adapter = str(start) if method == "dpo" else None
    loaded = prepare_peft_model(
        config.model_name,
        config.model,
        config.quant,
        config.lora,
        existing_adapter=existing_adapter,
    )
    tracker = init_tracking(config, run_name=config.run_name)

    common = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.align.epochs,
        "learning_rate": config.align.lr,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "beta": config.align.beta,
        "max_length": config.align.max_length,
        "max_prompt_length": config.align.max_prompt_length,
        "warmup_ratio": config.train.warmup_ratio,
        "logging_steps": config.train.logging_steps,
        "save_strategy": "steps",
        "save_steps": config.train.save_steps,
        "save_total_limit": config.train.save_total_limit,
        "seed": config.seed,
        "report_to": [],
        **precision_kwargs(),
    }
    if method == "orpo":
        args = trl.ORPOConfig(**common)
        trainer = trl.ORPOTrainer(
            model=loaded.model, args=args, train_dataset=dataset, processing_class=loaded.tokenizer
        )
    else:
        args = trl.DPOConfig(**common)
        trainer = trl.DPOTrainer(
            model=loaded.model,
            ref_model=None,  # PEFT: reference is the adapter-disabled base
            args=args,
            train_dataset=dataset,
            processing_class=loaded.tokenizer,
        )
    trainer.add_callback(
        EarlyStoppingCallback(early_stopping_patience=config.train.early_stopping_patience)
    )

    trainer.train(resume_from_checkpoint=str(resume) if resume else None)

    best_dir = output_dir / "best"
    last_dir = output_dir / "last"
    save_checkpoint(trainer.model, loaded.tokenizer, best_dir)
    save_checkpoint(trainer.model, loaded.tokenizer, last_dir)
    write_meta_json(
        best_dir, config=config.model_dump(), seed=config.seed, data_version=config.data.lang
    )
    write_meta_json(
        last_dir, config=config.model_dump(), seed=config.seed, data_version=config.data.lang
    )
    rep = config.reporting
    if rep.tables or rep.plots:
        try:
            from slm_coach.reporting import export_training_artifacts

            trainer.save_state()  # writes the cumulative trainer_state.json to output_dir
            export_training_artifacts(output_dir, make_tables=rep.tables, make_plots=rep.plots)
        except Exception as exc:  # noqa: BLE001 - artifacts must never fail a good run
            logger.warning("Could not export training metrics", extra={"error": str(exc)})
    tracker.close()
    logger.info("Alignment complete", extra={"method": method, "best": str(best_dir)})
    return best_dir
