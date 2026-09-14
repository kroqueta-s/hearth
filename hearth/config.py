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


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw is not None and raw.strip() != "" else default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _path(key: str, default: str = "") -> Path:
    return Path(_str(key, default))


# --- Output ------------------------------------------------------------------
OUTPUT_DIR: Path = _path("HEARTH_OUTPUT_DIR")

# --- Image generation (ComfyUI: an external app, reached over HTTP only) ------
COMFY_BASE_URL: str = _str("HEARTH_COMFY_BASE_URL", "http://127.0.0.1:8200").rstrip("/")
COMFY_TIMEOUT_SEC: int = _int("HEARTH_COMFY_TIMEOUT_SEC", 1800)

# --- Starting ComfyUI (`hearth/comfy_process.py`) ----------------------------
# **hearth starts it as a child process**, the way `forge` starts `mincut`, and
# stops only the one it started. Nothing is installed into its virtual
# environment and its code is never touched: the only lever is the command line.
# Leave these empty and hearth will only ever adopt one somebody else started.
COMFY_PYTHON: str = _str("HEARTH_COMFY_PYTHON")
COMFY_ROOT: str = _str("HEARTH_COMFY_ROOT")

# **The arguments are how a spill is prevented**, so they are configuration and
# not code: which of them a card needs depends on the card. See `.env.example`.
COMFY_ARGS: str = _str("HEARTH_COMFY_ARGS")

# Start it when hearth starts. Off by default: a caller that only makes meshes
# should not pay a minute of weights loading it will never use.
COMFY_AUTOSTART: bool = _bool("HEARTH_COMFY_AUTOSTART", False)

# How long to wait for `/system_stats` to answer. 300 s is what
# `start-comfyui.ps1` waits; the first run after a custom node changes is slow.
COMFY_START_TIMEOUT_SEC: int = _int("HEARTH_COMFY_START_TIMEOUT_SEC", 300)

# --- Watching the card (`hearth/vram.py`) ------------------------------------
# **What the card really has**, in GB. Every number a GPU library reports here
# includes the shared pool - system RAM the driver spills into - and is
# therefore larger than the card: measured 2026-09-06 on this machine, 43.87 GB
# reported against 32 GB of dedicated VRAM. 0 means "report what is in use and
# claim no total".
VRAM_DEDICATED_GB: float = _float("HEARTH_VRAM_DEDICATED_GB", 0.0)

# How much shared (system) memory one process may use before its work is
# abandoned. **A spill does not fail, it pages** into the system RAM every other
# process wants. **The default is not measured as a threshold**, but the number
# it has to separate is: measured 2026-09-06 on this machine, ComfyUI sat at
# 0.01-0.11 GB of shared while it fitted, and reached 1.14 GB when it did not.
# 0 disables it.
VRAM_SHARED_ABORT_GB: float = _float("HEARTH_VRAM_SHARED_ABORT_GB", 1.0)

# How often the counters are read. **Not measured**: two seconds is short
# against the ten a spill takes to matter and long against the 0.2 ms a sample
# costs (measured 2026-09-06).
VRAM_SAMPLE_SEC: float = _float("HEARTH_VRAM_SAMPLE_SEC", 2.0)

# The image model used when a request does not name one.
DEFAULT_IMAGE_MODEL: str = _str("HEARTH_IMAGE_MODEL", "sdxl")

# ControlNet weights. Only the SDXL one for now; FLUX needs a different model.
CONTROLNET_MODEL: str = _str("HEARTH_CONTROLNET_MODEL")

_wf_raw = _str("HEARTH_WORKFLOW_DIR", "hearth/workflows")
WORKFLOW_DIR: Path = Path(_wf_raw) if Path(_wf_raw).is_absolute() else REPO_ROOT / _wf_raw

# **Free the GPU before generating an image.** ComfyUI and a 3D runner share one
# card, and a 3D model left resident does not leave room for an image model.
# Going over does not fail: it falls back to shared memory and becomes several
# times slower **without saying anything**, which is why this defaults to on.
# The cost of it being on is a reload of the 3D model afterwards, which is a
# number you can measure; the cost of it being off is a silent one.
FREE_MESH_BEFORE_IMAGE: bool = _bool("HEARTH_FREE_MESH_BEFORE_IMAGE", True)

# --- When a runner dies on its own -------------------------------------------
# How many times a generating call is tried again after its runner died without
# answering. **The driver aborts the process from under us**: on gfx1151 a large
# decode hits `PAL failed to submit CMD! result:-5`, the errors are sticky, and
# torch's abort handler ends the process (measured 2026-09-13: the reference
# robot at 1024 died on two consecutive attempts and finished on the third).
# There is nothing to catch inside the runner, because the runner is gone.
#
# A retry is safe here and only here: a generating call produces a file and
# changes nothing else, so running it twice costs time and nothing more. It is
# **not** applied to a cancel - which ends the process on purpose - nor to a
# runner that answered with an error, which is a real answer and will be the
# same the second time. Each retry pays a full load of the weights, so the
# ceiling is low on purpose. 0 turns it off.
GENERATE_RETRIES: int = _int("HEARTH_GENERATE_RETRIES", 1)

# What the HIP runtime writes to stderr, for the runner that died without
# saying why. **The first error is hidden behind the abort**: a HIP kernel
# fault is asynchronous and surfaces at the next synchronising call, the
# context is sticky from then on, and the throw during unwinding reaches
# `std::terminate` - so what a runner leaves behind is `PAL failed to submit
# CMD!` and no name for what actually went wrong. At 2 the runtime names the
# failing API and the error itself, on stderr, which hearth already drains and
# attaches to the failure. 0 leaves the runtime quiet.
#
# **It is not free-form**: AMD's levels are 0 none, 1 errors, 2 warnings and
# errors, 3 and up per-API tracing that would bury the progress lines. Anything
# above 2 is for a person chasing one bug by hand.
RUNNER_HIP_LOG_LEVEL: int = _int("HEARTH_RUNNER_HIP_LOG_LEVEL", 2)

# --- Keeping the GPU to ourselves --------------------------------------------
# A port that, when something is listening on it, means **another application**
# already holds the GPU. **Only one thing can have the VRAM**, so hearth refuses
# to load a runner rather than letting both fight over it. Zero disables it.
GPU_BUSY_PORT: int = _int("HEARTH_GPU_BUSY_PORT", 0)

# A port hearth listens on **itself** while a model is loaded, so that a second
# hearth (a second Blender window, or a command line next to a running one)
# discovers the first instead of quietly loading a second model into the same
# card. **It must not be the port above**: that one belongs to another
# application. Zero disables it, at the cost of nothing detecting that case -
# and the case is real, so this is on by default. **Tests set it to zero**, or
# they would refuse to run whenever a real hearth is up.
LOCK_PORT: int = _int("HEARTH_LOCK_PORT", 8011)

# The version of `docs/protocol.md` this hearth speaks. **A constant, not
# configuration**: it describes the code, so it is not something to set in .env.
PROTOCOL_VERSION: int = 1


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
