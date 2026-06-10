"""Custom ``lm-eval-harness`` task for standardized cross-checkpoint benchmarking.

Builds a task config and registers it so checkpoints can be run inside lm-eval-harness. Heavy
``lm_eval`` imports are deferred so this module imports without the ``eval`` extra installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slm_coach.utils.deps import require
from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)

TASK_NAME = "slm_sales_coach_vi"


def build_task_config(gold_path: str | Path) -> dict[str, Any]:
    """Build the lm-eval-harness task configuration for the gold test set.

    The task is a ``generate_until`` task that reads the gold JSONL, prompts the model with the
    case prompt, and compares against the reference with exact-match (a coarse standardized
    signal; the rich 7-criteria scoring lives in the project's own evaluator).

    Args:
        gold_path: Path to the gold test set (JSONL).

    Returns:
        A task-config dict consumable by lm-eval-harness.
    """
    return {
        "task": TASK_NAME,
        "dataset_path": "json",
        "dataset_kwargs": {"data_files": {"test": str(gold_path)}},
        "test_split": "test",
        "output_type": "generate_until",
        "doc_to_text": "{{prompt}}",
        "doc_to_target": "{{reference}}",
        "generation_kwargs": {"until": ["<|im_end|>"], "max_gen_toks": 512, "do_sample": False},
        "metric_list": [{"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}],
        "metadata": {"version": 1.0},
    }


def write_task_yaml(gold_path: str | Path, out_dir: str | Path) -> Path:
    r"""Write a runnable lm-eval task YAML for the gold set.

    The written file can be passed to lm-eval directly, e.g.::

        uv run lm_eval --model hf --model_args pretrained=<ckpt> \\
            --include_path <out_dir> --tasks slm_sales_coach_vi

    Args:
        gold_path: Path to the gold test set (JSONL).
        out_dir: Directory to write ``<TASK_NAME>.yaml`` into.

    Returns:
        Path to the written YAML file.
    """
    import yaml

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = out / f"{TASK_NAME}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(build_task_config(gold_path), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info("Wrote lm-eval task YAML", extra={"path": str(yaml_path)})
    return yaml_path


def register_task(gold_path: str | Path, include_dir: str | Path = "outputs/lm_eval_tasks") -> str:
    """Write the task YAML and register it with lm-eval-harness via an include path.

    Args:
        gold_path: Path to the gold test set used by the task.
        include_dir: Directory to write the task YAML into and register from.

    Returns:
        The registered task name.
    """
    lm_eval_tasks = require("lm_eval.tasks", "eval")
    yaml_path = write_task_yaml(gold_path, include_dir)
    manager = lm_eval_tasks.TaskManager(include_path=str(yaml_path.parent))
    if TASK_NAME not in manager.all_tasks:
        logger.warning("Task not picked up by lm-eval TaskManager", extra={"task": TASK_NAME})
    logger.info("Registered lm-eval task", extra={"task": TASK_NAME, "yaml": str(yaml_path)})
    return TASK_NAME
