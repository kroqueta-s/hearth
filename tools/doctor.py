# SPDX-License-Identifier: MIT
"""Check an installation, and say what is wrong with it in one place.

**The knowledge of what a working install looks like was scattered**: the
installer knew how to register a runner, `status` knew what was loaded, and
nobody checked the parts in between - that the output directory can be written
to, that a declared workflow file exists, that the python in `.env` is still
there after an environment was rebuilt.

    .venv\\Scripts\\python.exe tools\\doctor.py

It changes nothing. Every line is `ok`, `warn` or `FAIL`:

- **FAIL** is something that will not work.
- **warn** is something that will work but not as intended - a route declared
  with no weights behind it, ComfyUI not running when image routes are wanted.

The exit code is 0 unless something failed. **Runners are asked what they can
do**, which starts each one's python for a moment; that is the only slow part.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "client"))

from hearth import config, imagegen  # noqa: E402
from hearth.comfy import ComfyUIClient  # noqa: E402
from hearth_client import Hearth, RequestFailed  # noqa: E402


class Report:
    """Collects findings and counts the ones that matter."""

    def __init__(self) -> None:
        self.failed = 0
        self.warned = 0

    def ok(self, what: str, detail: str = "") -> None:
        print(f"  ok    {what}{f': {detail}' if detail else ''}")

    def warn(self, what: str, detail: str) -> None:
        self.warned += 1
        print(f"  warn  {what}: {detail}")

    def fail(self, what: str, detail: str) -> None:
        self.failed += 1
        print(f"  FAIL  {what}: {detail}")

    def section(self, title: str) -> None:
        print(f"\n{title}")


def check_env(report: Report) -> None:
    """The settings file itself: present, complete, and pointing at real things."""
    report.section("settings")
    env = REPO_ROOT / ".env"
    if not env.is_file():
        report.fail(".env", "not found. Run install.ps1, or copy .env.example")
        return
    report.ok(".env", str(env))

    def keys(path: Path) -> set[str]:
        out = set()
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                name = line.split("=", 1)[0].strip()
                if not name.startswith("HEARTH_RUNNER_"):
                    out.add(name)
        return out

    missing = keys(REPO_ROOT / ".env.example") - keys(env)
    if missing:
        # **Missing keys fall back to the default in code**, which is a working
        # program with a setting nobody chose. Say so before it surprises anyone.
        report.warn(".env", f"these are only in .env.example: {sorted(missing)}")
    else:
        report.ok(".env", "the same keys as .env.example")

    out_dir = config.OUTPUT_DIR
    if str(out_dir) in ("", "."):
        report.fail("HEARTH_OUTPUT_DIR", "not set")
    else:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            probe = out_dir / ".hearth-doctor"
            probe.write_text("", encoding="utf-8")
            probe.unlink()
            report.ok("output directory", str(out_dir))
        except OSError as exc:
            report.fail("output directory", f"{out_dir}: {exc}")


def check_ports(report: Report) -> None:
    """The two ports, which mean opposite things."""
    report.section("ports")

    def answering(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    busy = config.GPU_BUSY_PORT
    if busy <= 0:
        report.warn(
            "HEARTH_GPU_BUSY_PORT", "0: nothing detects another application holding the GPU"
        )
    elif answering(busy):
        report.warn(
            "HEARTH_GPU_BUSY_PORT",
            f"something is listening on {busy}, so hearth will refuse to load a model",
        )
    else:
        report.ok("HEARTH_GPU_BUSY_PORT", f"{busy}, nothing listening")

    lock = config.LOCK_PORT
    if lock <= 0:
        report.warn("HEARTH_LOCK_PORT", "0: a second hearth would not notice the first")
    elif lock == busy:
        report.fail(
            "HEARTH_LOCK_PORT",
            f"{lock} is also HEARTH_GPU_BUSY_PORT, and they do not mean the same thing",
        )
    elif answering(lock):
        report.warn("HEARTH_LOCK_PORT", f"{lock} answers: another hearth may have a model loaded")
    else:
        report.ok("HEARTH_LOCK_PORT", f"{lock}, free")


def check_runners(report: Report, hearth: Hearth) -> None:
    """Every declared runner: configured, startable, and able to say what it does."""
    report.section("runners")
    names = config.runner_names()
    if not names:
        report.warn("HEARTH_RUNNERS", "empty: nothing can make a mesh. Use tools/add_runner.ps1")
        return
    for name in names:
        spec = config.runner_spec(name)
        if not Path(spec["python"]).is_file():
            report.fail(name, f"python not found: {spec['python']}")
            continue
        if not spec["module"]:
            report.fail(name, "no module configured")
            continue
        if not Path(spec["cwd"]).is_dir():
            report.fail(name, f"working directory not found: {spec['cwd']}")
            continue
        try:
            caps = hearth.call("capabilities", {"model": name}, timeout=180)
        except (RequestFailed, OSError) as exc:
            report.fail(name, f"would not answer capabilities: {exc}")
            continue
        able = sorted(k for k, v in (caps.get("capabilities") or {}).items() if v)
        report.ok(name, f"contract {caps.get('contract', 1)}, can: {', '.join(able) or 'nothing'}")
        if not caps.get("params"):
            report.warn(name, "declares no params, so a caller can build no form for it")


def check_images(report: Report, hearth: Hearth) -> None:
    """The image side: ComfyUI, the declared models, and the files behind them."""
    report.section("image generation")
    alive = ComfyUIClient().is_alive()
    if alive:
        report.ok("ComfyUI", config.COMFY_BASE_URL)
    else:
        report.warn("ComfyUI", f"{config.COMFY_BASE_URL} does not answer: image routes will fail")

    for name, caps in imagegen.all_capabilities().items():
        if "error" in caps:
            report.fail(name, caps["error"])
            continue
        able = sorted(k for k, v in caps["capabilities"].items() if v)
        if not able:
            report.fail(name, "no route has a workflow declared in .env")
            continue
        report.ok(name, f"can: {', '.join(able)}")
        if not caps.get("checkpoint"):
            report.warn(name, "no checkpoint declared")
        spec = config.image_model_spec(name)
        for route in ("txt2img", "img2img", "controlnet"):
            workflow = spec[route]
            if workflow and not (config.WORKFLOW_DIR / workflow).is_file():
                report.fail(name, f"{route} names a workflow that is not there: {workflow}")
        if spec["controlnet"] and not config.CONTROLNET_MODEL:
            report.warn(
                name, "a ControlNet workflow is declared but HEARTH_CONTROLNET_MODEL is empty"
            )


def main() -> int:
    """Run every check and report."""
    print(f"hearth doctor - {REPO_ROOT}")
    report = Report()
    check_env(report)
    check_ports(report)

    report.section("hearth")
    python = config._str("HEARTH_PYTHON", sys.executable)  # noqa: SLF001 - one setting, read once
    if not Path(python).is_file():
        report.fail("HEARTH_PYTHON", f"not found: {python}")
        print("\ncannot go further without a python to run hearth")
        return 1
    try:
        with Hearth.start(python, REPO_ROOT) as hearth:
            alive = hearth.call("ping", timeout=60)
            report.ok("hearth starts", f"python {alive.get('python')}, pid {alive.get('pid')}")
            spoken = int(alive.get("protocol", 0))
            if spoken != config.PROTOCOL_VERSION:
                report.warn(
                    "protocol",
                    f"hearth speaks {spoken}, this tool expects {config.PROTOCOL_VERSION}",
                )
            else:
                report.ok("protocol", str(spoken))
            check_runners(report, hearth)
            check_images(report, hearth)
    except (OSError, RequestFailed, RuntimeError) as exc:
        report.fail("hearth", str(exc))

    print(f"\n{report.failed} failed, {report.warned} to look at")
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
