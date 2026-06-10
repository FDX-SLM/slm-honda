"""Runtime bootstrap for CLI entrypoints.

Call :func:`bootstrap` at the very start of each ``scripts/*.py`` command. On Windows it
re-launches the interpreter in UTF-8 mode (``-X utf8``) so libraries that read packaged text
files without an explicit encoding (e.g. TRL's ``.jinja`` chat templates) don't crash on the
cp1252 default. It loads ``.env`` (so judge + Langfuse keys reach the process), and quiets noisy
third-party warnings (Hugging Face cache symlinks and bitsandbytes/torch ``FutureWarning``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings


def bootstrap() -> None:
    """Enable UTF-8 mode on Windows, load ``.env``, and silence noisy third-party warnings.

    UTF-8 mode must be set at interpreter startup, so on Windows (when not already active) the
    interpreter is re-launched once with ``-X utf8`` and this process exits with the child's
    return code. On other platforms — and on the relaunched process — ``.env`` is loaded (existing
    environment values win) and warnings are tuned.
    """
    if sys.platform == "win32" and not sys.flags.utf8_mode:
        completed = subprocess.run([sys.executable, "-X", "utf8", *sys.argv])
        raise SystemExit(completed.returncode)

    from dotenv import load_dotenv

    load_dotenv()  # secrets from .env (OPENAI/GOOGLE/LANGFUSE keys); does not override real env

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", category=FutureWarning)
