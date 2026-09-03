# SPDX-License-Identifier: MIT
"""A runner that sleeps. **It generates nothing and owns no GPU.**

hearth's hardest promises are about what happens when work is *interrupted*:
a shutdown during a generation, a request still queued when one starts, a runner
that has to be killed because a `torch` loop will not stop for anything. None of
that can be tested against a real model - each attempt would cost minutes and a
graphics card - so it is tested against this.

It also serves the other half of `docs/runner_contract.md` §5: set
`SLEEPY_RESULT_KEY=params` and it answers the way a runner written against the
older wording did, which is how the promotion in `Manager` is checked.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

Progress = Callable[..., None]

# A box, in normalized scale, written as ASCII PLY. Nothing reads it but the
# path has to point at a real file: a caller told a mesh exists will open it.
_PLY = """ply
format ascii 1.0
element vertex 8
property float x
property float y
property float z
element face 12
property list uchar int vertex_indices
end_header
-0.5 -0.3 -0.2
0.5 -0.3 -0.2
0.5 0.3 -0.2
-0.5 0.3 -0.2
-0.5 -0.3 0.2
0.5 -0.3 0.2
0.5 0.3 0.2
-0.5 0.3 0.2
3 0 2 1
3 0 3 2
3 4 5 6
3 4 6 7
3 0 1 5
3 0 5 4
3 1 2 6
3 1 6 5
3 2 3 7
3 2 7 6
3 3 0 4
3 3 4 7
"""


def load(progress: Progress) -> float:
    """Pretend to load weights, for as long as `SLEEPY_LOAD_SEC` says."""
    seconds = float(os.environ.get("SLEEPY_LOAD_SEC", "0.2"))
    progress("load", f"loading (a pretend {seconds}s)")
    time.sleep(seconds)
    return seconds


def unload() -> tuple[bool, float]:
    """Pretend to give the VRAM back."""
    return True, 0.0


def image_to_mesh(params: dict[str, Any], progress: Progress) -> dict[str, Any]:
    """Sleep for `seconds`, then write a box.

    **The sleep is the point.** It is a generation long enough to interrupt,
    without a model and without a card.
    """
    out_dir = Path(str(params["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed = {"seconds", "steps", "seed"}
    unknown = set(params) - allowed - {"image_path", "out_dir"}
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)} (accepted: {sorted(allowed)})")

    # **Say which process this is, from inside the generation.** A test cannot
    # ask hearth: `capabilities` starts a runner and stops it again when it was
    # only started to ask, so the process it reports is already gone. The one
    # holding the GPU is this one.
    (out_dir / "runner.pid").write_text(str(os.getpid()), encoding="ascii")

    load(progress)
    seconds = float(params.get("seconds", 1.0))
    steps = max(1, int(params.get("steps", 10)))
    # **`SLEEPY_SILENT` says nothing at all while it works**, which is the only
    # way to test the parent watchdog on its own. A runner that reports progress
    # notices a dead parent the moment its write fails, and a runner that reads
    # stdin notices when that closes - so with either of those in play, the
    # watchdog could be missing and the test would still pass.
    silent = os.environ.get("SLEEPY_SILENT", "0") == "1"
    for step in range(1, steps + 1):
        time.sleep(seconds / steps)
        if not silent:
            progress("shape", "sleeping", step=step, total=steps)

    # **Written beside its final name and renamed** (contract §9): a run killed
    # partway must not leave a half-written file that looks finished.
    mesh_path = out_dir / "raw.ply"
    staging = out_dir / "raw.ply.part"
    staging.write_text(_PLY, encoding="ascii")
    os.replace(staging, mesh_path)

    used = {"seconds": seconds, "steps": steps, "seed": int(params.get("seed", 0))}
    # **The key the older wording used**, so that hearth's promotion is testable.
    key = os.environ.get("SLEEPY_RESULT_KEY", "params_used")
    result: dict[str, Any] = {
        "mesh_path": str(mesh_path),
        "n_vertices": 8,
        "n_faces": 12,
        key: used,
        "extra": {},
        "metrics": {"load_sec": 0.0, "gen_sec": seconds},
    }
    # Axes are reported unless the test asks for a runner that does not, which is
    # what an older runner looks like. **Never a placeholder**: absent means
    # unknown, and hearth passes that on rather than inventing one.
    if os.environ.get("SLEEPY_REPORT_AXES", "1") == "1":
        result["up_axis"] = "z"
        result["forward_axis"] = "y"
    return result
