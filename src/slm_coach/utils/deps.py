"""Helpers for lazily importing optional (GPU / eval / export) dependencies.

Heavy dependencies live in pyproject *extras* (``train``, ``gpu``, ``eval``, ``export``,
``tracking``) so the default install stays CPU-only. Code imports them lazily via
:func:`require`, which raises a clear, actionable :class:`ImportError` (never a bare one)
when the extra is not installed — so modules import cleanly without a GPU.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

#: Human-readable install hints per extra.
INSTALL_HINTS: dict[str, str] = {
    "train": "uv sync --extra train",
    "gpu": "uv sync --extra gpu   # Linux + CUDA only",
    "eval": "uv sync --extra eval",
    "export": "uv sync --extra export",
    "tracking": "uv sync --extra tracking",
}


def is_installed(module: str) -> bool:
    """Return whether ``module`` can be imported without importing it."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def require(module: str, extra: str) -> ModuleType:
    """Import and return an optional dependency, or raise an actionable error.

    Args:
        module: The importable module name (e.g. ``"torch"``, ``"google.genai"``).
        extra: The pyproject extra that provides it (e.g. ``"train"``).

    Returns:
        The imported module.

    Raises:
        ImportError: If the module is not installed, with a hint to install the extra.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        hint = INSTALL_HINTS.get(extra, f"install the '{extra}' extra")
        raise ImportError(
            f"'{module}' is required for this step but is not installed. "
            f"Install the '{extra}' extra:  {hint}"
        ) from exc
