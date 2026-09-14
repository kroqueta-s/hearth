# SPDX-License-Identifier: MIT
"""Where a request writes is always an absolute path. **No GPU needed.**

The directory is handed to a runner, and a runner resolves a relative path
against its own working directory, which is its own repository. With
`HEARTH_OUTPUT_DIR` unset - a checkout without `.env` - the default was `.`, and
results were written into the runner's checkout (found 2026-09-15: meshes under
`tests/fake_runner/<time>/`). What is pinned:

- **No setting lands in hearth's own `output/`**, as an absolute path.
- **A relative `out_dir` is resolved by hearth**, against hearth's working
  directory, before any runner sees it.
- An absolute `out_dir` is kept.

Run it with hearth's own virtual environment::

    .venv\\Scripts\\python.exe .\\tests\\test_run_dir.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hearth import config, worker  # noqa: E402


def test_an_unset_output_dir_lands_in_hearth_s_own_output() -> None:
    saved = config.OUTPUT_DIR
    config.OUTPUT_DIR = Path("")
    made: Path | None = None
    try:
        made = worker._run_dir({})
        assert made.is_absolute(), made
        assert made.parent == (REPO_ROOT / "output").resolve(), made
    finally:
        config.OUTPUT_DIR = saved
        if made is not None and made.is_dir() and not any(made.iterdir()):
            made.rmdir()


def test_a_relative_out_dir_is_resolved_by_hearth_not_by_the_runner() -> None:
    with tempfile.TemporaryDirectory() as raw:
        here = os.getcwd()
        os.chdir(raw)
        try:
            made = worker._run_dir({"out_dir": "relative_run"})
        finally:
            os.chdir(here)
        assert made.is_absolute(), made
        assert made == (Path(raw) / "relative_run").resolve(), made
        assert made.is_dir(), made


def test_an_absolute_out_dir_is_kept() -> None:
    with tempfile.TemporaryDirectory() as raw:
        target = Path(raw) / "absolute_run"
        assert worker._run_dir({"out_dir": str(target)}) == target.resolve()


def main() -> int:
    """Run every test."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  OK   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
