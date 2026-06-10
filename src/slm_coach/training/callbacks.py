"""Training callbacks: eval-during-training, checkpoint metadata, early stopping.

The :class:`EvalDuringTraining` callback runs on each evaluation (every ``eval_steps``) over a
fast gold-test subset, generates answers, scores them with a fast proxy rubric (or a judge when
enabled), injects ``eval_rubric_avg`` into the trainer metrics so it drives best-model
selection + early stopping, and logs a sample generation to Langfuse. :func:`write_meta_json`
records the per-checkpoint provenance (config, git commit, data version, seed, metrics).

``transformers.TrainerCallback`` imports without torch, so this module stays import-clean.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import TrainerCallback

from slm_coach.eval.rubric import CRITERIA, weighted_average
from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

    from slm_coach.config import EvalDuringTrainingConfig
    from slm_coach.tracking import Tracker

logger = get_logger(__name__)

#: Default metric key the trainer watches (``eval_`` + ``metric_for_best_model``).
EVAL_METRIC_KEY = "eval_rubric_avg"


def eval_metric_key(metric_name: str) -> str:
    """Return the trainer metric key for a configured ``metric_for_best_model`` name."""
    return metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"


def _git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def write_meta_json(
    checkpoint_dir: str | Path,
    *,
    config: dict[str, Any],
    seed: int,
    data_version: str,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Write the per-checkpoint ``meta.json`` (config, git commit, data version, seed, metrics).

    Args:
        checkpoint_dir: Directory of the checkpoint to annotate.
        config: Serialized config snapshot.
        seed: Global seed used for the run.
        data_version: Version string of the training data.
        metrics: Optional metrics captured at this checkpoint.

    Returns:
        Path to the written ``meta.json``.
    """
    out = Path(checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": config,
        "git_commit": _git_commit(),
        "data_version": data_version,
        "seed": seed,
        "metrics": metrics or {},
    }
    path = out / "meta.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote checkpoint meta.json", extra={"path": str(path)})
    return path


def _tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization for the proxy overlap metric."""
    return text.lower().split()


def overlap_f1(prediction: str, reference: str) -> float:
    """Token-overlap F1 in ``[0, 1]`` between a prediction and reference.

    A fast, dependency-free proxy for answer quality during training (the full 7-criteria
    rubric runs offline in evaluation). Returns 0.0 when either side is empty.
    """
    pred = _tokenize(prediction)
    ref = _tokenize(reference)
    if not pred or not ref:
        return 0.0
    pred_set, ref_set = set(pred), set(ref)
    overlap = len(pred_set & ref_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(ref_set)
    return 2 * precision * recall / (precision + recall)


def proxy_rubric_score(prediction: str, reference: str) -> float:
    """Map token-overlap F1 onto the 1-5 rubric scale used as the training metric."""
    return 1.0 + 4.0 * overlap_f1(prediction, reference)


def _prompt_to_text(prompt: list[dict[str, str]]) -> str:
    """Flatten a chat prompt into plain text for a judge."""
    return "\n".join(m.get("content", "") for m in prompt)


class EvalDuringTraining(TrainerCallback):
    """Evaluate a gold-test subset on each evaluation and inject the rubric metric.

    Scores answers either with the real GPT/Gemini judges (when ``judges`` is provided, i.e.
    ``use_judge`` is on) or with a fast token-overlap proxy. The averaged 1-5 score is written
    into the trainer metrics under ``eval_<metric_name>`` so it drives best-model selection and
    early stopping.

    Attributes:
        config: Eval-during-training settings.
        gold: The (truncated) gold subset, each item ``{"prompt", "reference", "mode"}``.
        generate_fn: Callable mapping a list of chat prompts to generated answer strings.
        judges: Optional judge backends; when present, real rubric scoring is used.
        tracker: Optional Langfuse tracker for sample generations.
        metric_key: The trainer metric key this callback writes.
    """

    def __init__(
        self,
        config: EvalDuringTrainingConfig,
        *,
        gold_records: Sequence[dict[str, Any]],
        generate_fn: Callable[[list[list[dict[str, str]]]], list[str]],
        judges: Sequence[Any] | None = None,
        weights: Mapping[str, float] | None = None,
        metric_name: str = "rubric_avg",
        tracker: Tracker | None = None,
    ) -> None:
        """Store the callback configuration and dependencies."""
        self.config = config
        self.gold = list(gold_records)[: config.subset_size]
        self.generate_fn = generate_fn
        self.judges = list(judges) if judges else []
        self.weights = dict(weights) if weights else None
        self.metric_key = eval_metric_key(metric_name)
        self.tracker = tracker
        self.history: list[tuple[int, float]] = []

    def _score_answer(self, prompt_text: str, answer: str, reference: str) -> float:
        """Return a 1-5 score via the judges (if any) or the token-overlap proxy."""
        if not self.judges:
            return proxy_rubric_score(answer, reference)
        per_criterion: dict[str, list[float]] = {c: [] for c in CRITERIA}
        for judge in self.judges:
            for criterion, value in (
                judge.score(prompt=prompt_text, answer=answer, criteria=CRITERIA).as_dict().items()
            ):
                per_criterion[criterion].append(value)
        means = {c: (sum(v) / len(v) if v else 0.0) for c, v in per_criterion.items()}
        return weighted_average(means, self.weights)

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate on the gold subset, score, and inject the rubric metric."""
        if not self.config.enabled or not self.gold:
            return control

        prompts = [g["prompt"] for g in self.gold]
        references = [g.get("reference", "") for g in self.gold]
        answers = self.generate_fn(prompts)

        scores = [
            self._score_answer(_prompt_to_text(p), a, r)
            for p, a, r in zip(prompts, answers, references, strict=False)
        ]
        avg = sum(scores) / len(scores) if scores else 0.0

        step = getattr(state, "global_step", 0)
        if metrics is not None:
            metrics[self.metric_key] = avg
        self.history.append((step, avg))

        if self.tracker is not None and answers:
            self.tracker.log_generation(
                name="eval_sample",
                prompt=str(prompts[0]),
                completion=answers[0],
                step=step,
            )
        logger.info(
            "Eval-during-training",
            extra={"step": step, "metric": self.metric_key, "value": round(avg, 4)},
        )
        return control
