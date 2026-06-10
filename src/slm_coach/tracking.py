"""Experiment tracking via Langfuse.

Langfuse captures **sample generations** during training/evaluation for qualitative review. It
degrades to a no-op when the optional ``tracking`` extra is not installed or the Langfuse keys are
absent, so training and evaluation run anywhere. Quantitative metrics (loss, eval rubric, per-mode
scores) are persisted by :mod:`slm_coach.reporting` as CSV tables + PNG charts — not here.

Secrets come from the environment only (never hardcoded): ``LANGFUSE_PUBLIC_KEY``,
``LANGFUSE_SECRET_KEY``, and ``LANGFUSE_HOST`` (defaults to ``https://cloud.langfuse.com``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from slm_coach.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_coach.config import BaseConfig

logger = get_logger(__name__)

_DEFAULT_HOST = "https://cloud.langfuse.com"


class Tracker:
    """Thin Langfuse facade with graceful degradation.

    Attributes:
        run_name: Optional run name (recorded on logged generations).
        langfuse_enabled: Whether Langfuse logging is active.
    """

    def __init__(self, *, langfuse: bool = False, run_name: str | None = None) -> None:
        """Initialize Langfuse if requested, installed, and configured via env keys.

        Args:
            langfuse: Whether to attempt Langfuse logging (needs the ``tracking`` extra + keys).
            run_name: Optional run name attached to logged generations.
        """
        self.run_name = run_name
        self.langfuse_enabled = False
        self._langfuse: Any = None
        if langfuse:
            self._init_langfuse()

    def _init_langfuse(self) -> None:
        """Construct a Langfuse client if installed and the public/secret keys are present."""
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            logger.info("Langfuse disabled (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set)")
            return
        try:
            from langfuse import Langfuse
        except ImportError:
            logger.warning("langfuse not installed; install the 'tracking' extra to enable it")
            return
        self._langfuse = Langfuse()
        self.langfuse_enabled = True
        logger.info(
            "Langfuse logging enabled",
            extra={"host": os.environ.get("LANGFUSE_HOST", _DEFAULT_HOST)},
        )

    def log_generation(self, *, name: str, prompt: str, completion: str, **metadata: Any) -> None:
        """Record a sample generation to Langfuse for qualitative tracking (no-op if disabled).

        Args:
            name: Trace name (e.g. ``"eval_sample"``).
            prompt: The input prompt text.
            completion: The model's generated answer.
            **metadata: Extra fields (e.g. ``step``) attached to the trace.
        """
        if not self.langfuse_enabled:
            return
        try:
            self._langfuse.trace(
                name=name,
                input=prompt,
                output=completion,
                metadata={**metadata, "run_name": self.run_name},
            )
        except Exception as exc:  # noqa: BLE001 - tracking must never crash a run
            logger.warning("Langfuse trace failed", extra={"error": str(exc)})

    def close(self) -> None:
        """Flush buffered Langfuse events (no-op if disabled)."""
        if self.langfuse_enabled:
            try:
                self._langfuse.flush()
            except Exception as exc:  # noqa: BLE001 - flushing must never crash a run
                logger.warning("Langfuse flush failed", extra={"error": str(exc)})

    def __enter__(self) -> Tracker:
        """Enter the tracker context."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close tracking on context exit."""
        self.close()


def init_tracking(config: BaseConfig, *, run_name: str | None = None) -> Tracker:
    """Build a :class:`Tracker` from a :class:`~slm_coach.config.BaseConfig`.

    Args:
        config: Any config inheriting :class:`~slm_coach.config.BaseConfig`.
        run_name: Optional run name override.

    Returns:
        A configured (possibly no-op) :class:`Tracker`.
    """
    return Tracker(langfuse=config.tracking.langfuse, run_name=run_name)
