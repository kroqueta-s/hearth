# SPDX-License-Identifier: MIT
"""Pin the rules that keep hearth a relay rather than a model host.

Four things are checked, and each one is a rule that is easy to break by
accident and expensive to discover later:

- **No model is named in code.** Runners and image models come from `.env`, so
  adding a fourth model is configuration rather than a patch.
- **hearth never imports torch.** Only a runner holds the GPU. The moment hearth
  imports torch, the separation that lets runners have conflicting dependencies
  is gone.
- **Nothing imports `bpy`.** Blender's python is GPL; importing it would cost
  this repository its MIT licence.
- **`.env` and `.env.example` declare the same keys**, so the example stays a
  usable starting point instead of drifting into fiction.

Run it with hearth's own virtual environment::

    .venv\\Scripts\\python.exe .\\tests\\test_config.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hearth import config  # noqa: E402

_BPY_IMPORT = re.compile(r"^\s*(?:import\s+bpy|from\s+bpy[\s.])", re.MULTILINE)
_TORCH_IMPORT = re.compile(r"^\s*(?:import\s+torch|from\s+torch[\s.])", re.MULTILINE)
_SOURCE_DIRS = ("hearth", "tools", "tests", "templates")


def _sources(sub: str) -> list[Path]:
    """Every .py file under a directory."""
    directory = REPO_ROOT / sub
    if not directory.is_dir():
        return []
    return sorted(f for f in directory.rglob("*.py") if "__pycache__" not in f.parts)


def test_runners_come_from_env() -> None:
    """The list of runners comes from `.env`, never from code."""
    names = config.runner_names()
    assert names, "HEARTH_RUNNERS is empty (check .env)"


def test_runner_spec_is_complete() -> None:
    """Every declared runner names a python, a module and a directory."""
    for name in config.runner_names():
        spec = config.runner_spec(name)
        assert spec["python"], f"{name}: no python configured"
        assert spec["module"], f"{name}: no module configured"
        assert spec["cwd"], f"{name}: no working directory configured"


def test_image_models_come_from_env() -> None:
    """The same rule for image models, and the default is one of them."""
    names = config.image_model_names()
    assert names, "HEARTH_IMAGE_MODELS is empty (check .env)"
    assert config.DEFAULT_IMAGE_MODEL in names, (
        f"the default image model {config.DEFAULT_IMAGE_MODEL} is not in {names}"
    )


def test_hearth_never_imports_torch() -> None:
    """**hearth holds no torch.** Only a runner touches the GPU."""
    offenders = [
        str(f.relative_to(REPO_ROOT))
        for f in _sources("hearth")
        if _TORCH_IMPORT.search(f.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"hearth imports torch: {offenders}"


def test_nothing_imports_bpy() -> None:
    """**Nothing here imports `bpy`.** Doing so would end the MIT licence."""
    offenders = [
        str(f.relative_to(REPO_ROOT))
        for sub in _SOURCE_DIRS
        for f in _sources(sub)
        if _BPY_IMPORT.search(f.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"these import bpy: {offenders}"


def test_env_example_matches_env() -> None:
    """`.env` and `.env.example` declare the same keys.

    **`HEARTH_RUNNER_*` entries are exempt**: they are written per installation
    by the installer, and hard-coding somebody's paths into the example would
    make it a worse starting point, not a better one.

    Skipped when there is no `.env`, which is the case on a fresh clone and in
    CI. **The example is the file that is checked in**, so it is the one that
    has to stay honest.
    """

    def keys(path: Path) -> set[str]:
        out = set()
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.lstrip("﻿").strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name = line.split("=", 1)[0].strip()
            if not name.startswith("HEARTH_RUNNER_"):
                out.add(name)
        return out

    env = REPO_ROOT / ".env"
    if not env.is_file():
        print("       (no .env; nothing to compare against)")
        return
    example = keys(REPO_ROOT / ".env.example")
    actual = keys(env)
    assert example == actual, (
        f"only in .env.example={example - actual} / only in .env={actual - example}"
    )


def main() -> int:
    """Run every test."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
