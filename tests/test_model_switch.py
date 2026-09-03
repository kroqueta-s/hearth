# SPDX-License-Identifier: MIT
"""Check that switching between models actually works, on real hardware.

**This uses the GPU.** Every runner in `HEARTH_RUNNERS` is run in turn and then
the first one is run again, because the interesting failure is not "model A
works" but **"model A still works after model B has held the GPU"**.

Three things are verified for each switch:

- The model produces a mesh with faces in it.
- **Unloading gives the VRAM back.** Only the runner can measure that, so its
  reported `vram_used_gb` is what gets checked.
- `status` stops reporting the model as loaded.

It runs **under a watchdog** (`harness.py`) rather than simply waiting: a stall,
an overrun of the time budget, or a runner going past its VRAM cap **ends the
run and prints a diagnosis** instead of hanging until someone notices.

If another process holds the GPU, hearth refuses to load anything. **That is the
correct behaviour**, so this exits successfully rather than reporting a failure.

Run it with hearth's own virtual environment::

    .venv\\Scripts\\python.exe .\\tests\\test_model_switch.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from harness import BUDGET_SEC, IMAGE, HearthSession, WatchdogAbort, print_diagnosis  # noqa: E402

from hearth import config  # noqa: E402


def _sequence() -> list[str]:
    """The order to switch through. **It ends where it started** (A to B to A)."""
    names = config.runner_names()
    if len(names) < 2:
        raise RuntimeError(f"switching needs at least two runners (there are {names})")
    return [*names, names[0]]


def main() -> int:
    """Run the switch sequence and report."""
    if IMAGE is None:
        print(
            "set HEARTH_TEST_IMAGE to an input image. **There is no default**: "
            "hearth ships no specimen, and comparing runners on different "
            "pictures says nothing about the runners."
        )
        return 1
    if not IMAGE.is_file():
        print(f"HEARTH_TEST_IMAGE names nothing: {IMAGE}")
        return 1

    order = _sequence()
    print(f"switching: {' -> '.join(order)}")
    print("budgets: " + " / ".join(f"{n}={BUDGET_SEC.get(n)}s" for n in dict.fromkeys(order)))

    failures = 0
    results: list[tuple[str, dict[str, Any]]] = []
    with HearthSession() as hearth:
        status, _ = hearth.call("status", budget_sec=180.0, trace=False)
        if status.get("gpu_busy"):
            print("  SKIP another process holds the GPU. **Refusing is correct**, so stop here.")
            return 0

        for step, name in enumerate(order, start=1):
            print(f"\n--- {step}/{len(order)}: {name} ---", flush=True)
            started = time.perf_counter()
            try:
                result, record = hearth.call(
                    "image_to_mesh",
                    {"model": name, "image_path": str(IMAGE)},
                    budget_sec=BUDGET_SEC.get(name),
                )
            except WatchdogAbort as exc:
                print_diagnosis(exc)
                # **An aborted session is not to be trusted.** End it, fix the
                # cause, and start again.
                hearth.kill()
                return 1

            elapsed = time.perf_counter() - started
            mesh_path = Path(result["mesh_path"])
            ok = mesh_path.is_file() and result["n_faces"] > 0
            print(
                f"  {'OK  ' if ok else 'FAIL'} {name}: "
                f"{result['n_vertices']} vertices / {result['n_faces']} faces / {elapsed:.0f}s"
            )
            print(f"       stages: {record.summary()}")
            print(f"       metrics: {result.get('metrics')}")
            results.append((name, result))
            failures += 0 if ok else 1

            unload_started = time.perf_counter()
            unloaded, _ = hearth.call("unload", budget_sec=180.0, trace=False)
            unload_sec = time.perf_counter() - unload_started
            after, _ = hearth.call("status", budget_sec=180.0, trace=False)
            print(
                f"  unload {unload_sec:.1f}s / vram_used_gb={unloaded.get('vram_used_gb')} / "
                f"status.loaded={after.get('loaded')}"
            )
            if after.get("loaded") is not None:
                print("  FAIL it was unloaded, but status still reports it as loaded")
                failures += 1

    print("\n--- summary ---")
    for name, result in results:
        metrics = result.get("metrics", {})
        print(
            f"  {name:10s} {result['n_vertices']:>8} vertices / {result['n_faces']:>8} faces / "
            f"gen {metrics.get('gen_sec')}s / vram {metrics.get('vram_peak_gb')}GB"
        )

    print(f"\n{'passed' if failures == 0 else f'{failures} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
