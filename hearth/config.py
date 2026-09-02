# SPDX-License-Identifier: MIT
"""hearth's configuration, read from `.env`. **Never name a model in code.**

Runners are declared in `.env` through `HEARTH_RUNNERS`, and each one names its
python, its module and its working directory separately. **The working directory
is what lets a runner live in its own repository**: point it at the clone and
nothing else changes (see `docs/runner_contract.md` §7).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


def _str(key: str, default: str = "") -> str:
    raw = os.getenv(key)
    return raw.strip() if raw is not None and raw.strip() != "" else default


def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None and raw.strip() != "" else default


def _path(key: str, default: str = "") -> Path:
    return Path(_str(key, default))


# --- Output ------------------------------------------------------------------
OUTPUT_DIR: Path = _path("HEARTH_OUTPUT_DIR")

# --- Image generation (ComfyUI: an external app, reached over HTTP only) ------
COMFY_BASE_URL: str = _str("HEARTH_COMFY_BASE_URL", "http://127.0.0.1:8200").rstrip("/")
COMFY_TIMEOUT_SEC: int = _int("HEARTH_COMFY_TIMEOUT_SEC", 1800)

# The image model used when a request does not name one.
DEFAULT_IMAGE_MODEL: str = _str("HEARTH_IMAGE_MODEL", "sdxl")

# ControlNet weights. Only the SDXL one for now; FLUX needs a different model.
CONTROLNET_MODEL: str = _str("HEARTH_CONTROLNET_MODEL")

_wf_raw = _str("HEARTH_WORKFLOW_DIR", "hearth/workflows")
WORKFLOW_DIR: Path = Path(_wf_raw) if Path(_wf_raw).is_absolute() else REPO_ROOT / _wf_raw

# --- Keeping the GPU to ourselves --------------------------------------------
# A port that, when something is listening on it, means another process already
# holds the GPU. **Only one thing can have the VRAM**, so hearth refuses to load
# a runner rather than letting both fight over it. Zero disables the check.
GPU_BUSY_PORT: int = _int("HEARTH_GPU_BUSY_PORT", 0)


def image_model_names() -> list[str]:
    """Return the image models `.env` declares.

    **The same rule as runners**: the name lives in `.env`, never in code.
    """
    raw = _str("HEARTH_IMAGE_MODELS", "sdxl")
    return [name.strip() for name in raw.split(",") if name.strip()]


def image_model_spec(name: str) -> dict[str, str]:
    """Return one image model's checkpoint and workflows.

    Args:
        name: An image model listed in `HEARTH_IMAGE_MODELS`.

    Returns:
        A dict of `checkpoint` / `txt2img` / `img2img` / `controlnet`.
        **A route the model does not support is an empty string** (FLUX's
        ControlNet, for one).

    Raises:
        ValueError: If the name was never declared.
    """
    if name not in image_model_names():
        raise ValueError(
            f"unknown image model: {name} (HEARTH_IMAGE_MODELS lists {image_model_names()})"
        )
    key = name.upper().replace("-", "_")
    return {
        "checkpoint": _str(f"HEARTH_IMAGE_MODEL_{key}_CHECKPOINT"),
        "txt2img": _str(f"HEARTH_IMAGE_MODEL_{key}_TXT2IMG"),
        "img2img": _str(f"HEARTH_IMAGE_MODEL_{key}_IMG2IMG"),
        "controlnet": _str(f"HEARTH_IMAGE_MODEL_{key}_CONTROLNET"),
    }


def runner_names() -> list[str]:
    """Return the runners `.env` declares."""
    raw = _str("HEARTH_RUNNERS")
    return [name.strip() for name in raw.split(",") if name.strip()]


def runner_spec(name: str) -> dict[str, str]:
    """Return what is needed to start one runner.

    Args:
        name: A runner listed in `HEARTH_RUNNERS`.

    Returns:
        A dict of `python` / `module` / `cwd`. Unset keys are empty strings.
    """
    key = name.upper().replace("-", "_")
    return {
        "python": _str(f"HEARTH_RUNNER_{key}_PYTHON"),
        "module": _str(f"HEARTH_RUNNER_{key}_MODULE"),
        "cwd": _str(f"HEARTH_RUNNER_{key}_CWD", str(REPO_ROOT)),
    }
