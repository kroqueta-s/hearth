# SPDX-License-Identifier: MIT
"""**This is the only file with anything to do.** Everything else is the contract.

Three functions have to work:

- `load` puts the weights on the GPU,
- `unload` takes them off and gives the memory back,
- `image_to_mesh` turns one image into one mesh.

As written they raise `NotImplementedError`, which is deliberate: a template that
returned a plausible empty mesh would let a broken runner look like a working
one. **Replace the bodies, not the shape.**

`steps.py` next to this file counts a loop's progress without touching the
model's code. Use it; a caller with no progress cannot tell work from a hang.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import config
from .steps import StepCounter, count_scheduler, count_tqdm  # noqa: F401

Progress = Callable[..., None]

_MODEL: Any = None
_LOAD_SEC: float = 0.0
# Counts whichever loop is running. Bound to a request, reused across them.
_STEPS = StepCounter()


def load(progress: Progress | None = None) -> float:
    """Load the weights once per process.

    Args:
        progress: Where stage notifications go.

    Returns:
        Seconds spent loading. Zero if they were already loaded.
    """
    global _MODEL, _LOAD_SEC
    if _MODEL is not None:
        return 0.0

    def say(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    started = time.perf_counter()
    say("weights", f"loading the weights from {config.WEIGHTS_DIR}")

    # ---------------------------------------------------------------- replace
    # _MODEL = YourPipeline.from_pretrained(str(config.WEIGHTS_DIR)).to("cuda")
    raise NotImplementedError("fill in load() with your model")
    # ------------------------------------------------------------------------

    _LOAD_SEC = time.perf_counter() - started
    say("loaded", f"loaded in {_LOAD_SEC:.1f}s")
    return _LOAD_SEC


def unload() -> tuple[bool, float]:
    """Release the weights and give the VRAM back.

    **Report what is actually still held.** Only this process can measure that,
    and without the number there is no way to tell a real release from a
    hopeful one.

    Returns:
        (whether anything was released, VRAM still in use in GB).
    """
    global _MODEL, _LOAD_SEC
    if _MODEL is None:
        return False, 0.0
    _MODEL = None
    _LOAD_SEC = 0.0

    import gc

    gc.collect()

    # ---------------------------------------------------------------- replace
    # import torch
    # torch.cuda.empty_cache()
    # free, total = torch.cuda.mem_get_info()
    # return True, round((total - free) / 1024**3, 2)
    return True, 0.0
    # ------------------------------------------------------------------------


def image_to_mesh(params: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """One image to a raw mesh.

    Args:
        params: `image_path` and `out_dir` are always present, both absolute.
            Anything else was declared in `capabilities()["params"]`.
        progress: Where stage notifications go.

    Returns:
        The shape described in `docs/runner_contract.md` §5.

    Raises:
        FileNotFoundError: If the input image is not there.
        ValueError: If an argument was never declared.
    """
    image_path = Path(str(params["image_path"]))
    out_dir = Path(str(params["out_dir"]))
    if not image_path.is_file():
        raise FileNotFoundError(f"input image not found: {image_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # **Reject what was never declared, with a reason.** Silently ignoring an
    # argument means a caller can ask for something and never learn it did
    # nothing.
    allowed = {"steps", "seed"}
    unknown = set(params) - allowed - {"image_path", "out_dir"}
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)} (accepted: {sorted(allowed)})")

    load(progress)

    # **Preprocessing is this runner's job**, and what it needs differs by model.
    progress("preprocess", "preparing the input image")

    progress("generate", "generating the mesh")
    # Name the stage before the loop runs; the counter fills in the numbers.
    _STEPS.bind(progress, "generate", "generating")
    started = time.perf_counter()
    try:
        # ------------------------------------------------------------ replace
        # count_scheduler(_MODEL.scheduler, _STEPS)   # a diffusers pipeline, or
        # count_tqdm(their_module, _STEPS)            # a hand-written loop
        # mesh = _MODEL(image_path, num_inference_steps=params.get("steps", config.STEPS))
        raise NotImplementedError("fill in image_to_mesh() with your model")
        # ----------------------------------------------------------------------
    finally:
        # **Let go of this request's sink.** A step must never be reported
        # against a request that has already been answered.
        _STEPS.bind(None, "generate")
    gen_sec = time.perf_counter() - started

    progress("export", "writing the mesh")
    mesh_path = out_dir / "raw.ply"
    # mesh.export(str(mesh_path))

    return {
        "mesh_path": str(mesh_path),
        "n_vertices": 0,
        "n_faces": 0,
        "extra": {},
        "metrics": {
            "load_sec": round(_LOAD_SEC, 2),
            # **Never use this as a pass/fail signal.** It varies by several
            # times for identical settings, and the first run on a machine can
            # be an order of magnitude slower while kernels are tuned.
            "gen_sec": round(gen_sec, 2),
        },
    }
