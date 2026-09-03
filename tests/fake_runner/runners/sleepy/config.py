# SPDX-License-Identifier: MIT
"""Settings, read from this repository's own `.env`.

**Nothing is hard-coded here that differs between machines.** Paths to weights,
paths to an upstream clone, and any setting worth changing come from `.env`, so
the source is the same on every installation.

hearth points at this repository through `HEARTH_RUNNER_<NAME>_CWD`, so **this
file is read from the runner's own directory, not from hearth's**.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# runners/example/config.py -> the repository root.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")


def _str(key: str, default: str = "") -> str:
    raw = os.getenv(key)
    return raw.strip() if raw is not None and raw.strip() != "" else default


def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None and raw.strip() != "" else default


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw is not None and raw.strip() != "" else default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Where the weights are. **Always absolute**: the working directory is not yours
# to rely on.
WEIGHTS_DIR: Path = Path(_str("EXAMPLE_WEIGHTS_DIR"))

# Reported in `capabilities`, so a caller can tell two installations apart.
MODEL_VERSION: str = _str("EXAMPLE_MODEL_VERSION", "0.0")

# The model's own settings. **A default that came from a measurement should say
# so**, and one that did not should say that too.
STEPS: int = _int("EXAMPLE_STEPS", 30)
GUIDANCE_SCALE: float = _float("EXAMPLE_GUIDANCE_SCALE", 5.0)

# Cap the VRAM the process may use, if your framework can. **Going past the
# card's dedicated memory does not fail; it silently spills and becomes several
# times slower**, so failing loudly is the cheaper outcome.
VRAM_LIMIT_GB: float = _float("EXAMPLE_VRAM_LIMIT_GB", 0.0)

# How often to report that the process is alive during a long stage.
HEARTBEAT_SEC: float = _float("EXAMPLE_HEARTBEAT_SEC", 10.0)

# Whether preprocessing runs. **Check before turning it off**: for some models
# the cut-out silently does nothing useful without it.
PREPROCESS: bool = _bool("EXAMPLE_PREPROCESS", True)
