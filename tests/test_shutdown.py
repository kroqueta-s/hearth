# SPDX-License-Identifier: MIT
"""Shutting hearth down leaves nothing holding the card.

**This is the failure that had no symptom.** Stopping hearth while a generation
ran did not error, did not hang visibly, and did not warn: the runner simply
carried on in another process with the VRAM, and everything afterwards was
several times slower for no reason anybody could see.

Three things go wrong together, and each is checked here:

1. `shutdown` used to ask the runner to unload, which waits on the lock the
   generation holds. So it looked like a hang.
2. A caller's answer to a hang is to kill hearth - and on Windows killing a
   process does not kill its children. The runner outlived it.
3. Requests still in the GPU queue kept running during the shutdown, so a
   `load` that arrived a moment too late started a **new** runner that nothing
   was left to stop.

`tests/fake_runner/` provides a runner that sleeps, so all of this is testable
without a model and without a graphics card.

Run it with hearth's own virtual environment::

    .venv\\Scripts\\python.exe .\\tests\\test_shutdown.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE = REPO_ROOT / "tests" / "fake_runner"
OUT = REPO_ROOT / "output" / "_shutdown_test"


def _env() -> dict[str, str]:
    """An environment with one runner that sleeps, and no lock port.

    `load_dotenv` never overwrites what is already set, so `.env` is left alone.

    **`HEARTH_LOCK_PORT=0` is not optional.** The default is a real port, and a
    hearth started by the operator's Blender would already hold it - the test
    would then fail with `GpuBusyError` and say nothing about shutting down.
    """
    return {
        **os.environ,
        "HEARTH_RUNNERS": "sleepy",
        "HEARTH_RUNNER_SLEEPY_PYTHON": sys.executable,
        "HEARTH_RUNNER_SLEEPY_MODULE": "runners.sleepy",
        "HEARTH_RUNNER_SLEEPY_CWD": str(FAKE),
        "HEARTH_LOCK_PORT": "0",
        "HEARTH_GPU_BUSY_PORT": "0",
        "SLEEPY_LOAD_SEC": "0.1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }


class Session:
    """One hearth, spoken to directly so that timing can be measured."""

    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "hearth"],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=_env(),
        )
        self._id = 0

    def send(self, method: str, params: dict | None = None) -> int:
        """Send one request and return its id."""
        self._id += 1
        assert self.proc.stdin is not None
        self.proc.stdin.write(
            json.dumps({"id": self._id, "method": method, "params": params or {}}) + "\n"
        )
        self.proc.stdin.flush()
        return self._id

    def until(self, request_id: int, timeout: float = 30.0) -> tuple[str, dict]:
        """Read until this request is answered. Other ids may arrive first."""
        assert self.proc.stdout is not None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise AssertionError("hearth closed its stdout without answering")
            event = json.loads(line)
            if event.get("id") != request_id or event.get("event") == "progress":
                continue
            return event["event"], event.get("result") or event.get("error") or {}
        raise AssertionError(f"request {request_id} went unanswered for {timeout}s")

    def kill(self) -> None:
        """End it, whatever state it is in."""
        if self.proc.poll() is None:
            self.proc.kill()


def _alive(pid: int) -> bool:
    """Whether a process id is still running. **No new dependency for this.**"""
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    done = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return str(pid) in done.stdout



def _kill(pid: int) -> None:
    """End one process and **not its children**. No new dependency for this."""
    if sys.platform != "win32":
        os.kill(pid, 9)
        return
    subprocess.run(
        ["taskkill", "/F", "/PID", str(pid)],
        capture_output=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _generating(session: Session, out_dir: Path, seconds: float = 30.0) -> int:
    """Start a generation and return the process actually running it.

    **The pid comes from the runner, not from hearth.** `capabilities` starts a
    runner and stops it again when it was only started to ask, so the process it
    names is already gone by the time anyone looks. The one that matters is the
    one holding the card, and it says so from inside the generation.
    """
    marker = out_dir / "runner.pid"
    marker.unlink(missing_ok=True)
    session.send(
        "image_to_mesh",
        {"model": "sleepy", "image_path": "x", "out_dir": str(out_dir), "seconds": seconds},
    )
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if marker.is_file():
            return int(marker.read_text(encoding="ascii"))
        time.sleep(0.05)
    raise AssertionError("the runner never started generating")


def test_shutdown_during_a_generation_answers_at_once() -> None:
    """**A shutdown must not wait for the generation it is ending.**

    It used to, because it asked the runner to unload and that call queues behind
    the generation. Looking like a hang is what led to hearth being killed, which
    is what orphaned the runner.
    """
    session = Session()
    try:
        session.send("image_to_mesh", {"model": "sleepy", "image_path": "x", "seconds": 30})
        time.sleep(1.0)
        started = time.perf_counter()
        kind, payload = session.until(session.send("shutdown"), timeout=10)
        elapsed = time.perf_counter() - started
        assert kind == "result" and payload.get("bye") is True, payload
        assert elapsed < 2.0, f"shutdown took {elapsed:.1f}s while a generation was running"
        session.proc.wait(timeout=10)
        assert session.proc.returncode is not None
    finally:
        session.kill()


def test_the_runner_does_not_outlive_hearth() -> None:
    """**The whole point.** A runner left alive keeps the card and says nothing."""
    session = Session()
    try:
        pid = _generating(session, OUT / "outlive")
        assert _alive(pid), "the runner did not start"
        session.until(session.send("shutdown"), timeout=10)
        session.proc.wait(timeout=10)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.1)
        assert not _alive(pid), f"the runner ({pid}) outlived hearth and still holds the GPU"
    finally:
        session.kill()


def test_a_queued_request_does_not_start_a_new_runner() -> None:
    """A request that arrives a moment too late is answered, not run.

    Running it would start a runner **after** shutdown took its list of them,
    which is the same orphan by another route.
    """
    session = Session()
    try:
        pid = _generating(session, OUT / "queued")
        queued = session.send("load", {"model": "sleepy"})  # waits behind it
        session.until(session.send("shutdown"), timeout=10)
        kind, payload = session.until(queued, timeout=10)
        assert kind == "error", payload
        assert "shutting down" in str(payload.get("message", "")), payload
        session.proc.wait(timeout=10)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.1)
        assert not _alive(pid), "a runner was left behind"
    finally:
        session.kill()


def test_an_older_runners_params_are_promoted() -> None:
    """**Contract §5**: `params_used`, whatever the runner called it.

    A runner written against the earlier wording reports `params`. That is a
    rename, so promoting it invents nothing - and without it every setting a
    generation ran with is lost to the caller.
    """
    session = Session()
    try:
        os.environ["SLEEPY_RESULT_KEY"] = "params"
        session.proc.kill()
        session.__init__()  # type: ignore[misc] - restart with the new environment
        kind, out = session.until(
            session.send(
                "image_to_mesh", {"model": "sleepy", "image_path": "x", "seconds": 0.2, "steps": 2}
            ),
            timeout=60,
        )
        assert kind == "result", out
        assert out.get("params_used", {}).get("steps") == 2, out
    finally:
        os.environ.pop("SLEEPY_RESULT_KEY", None)
        session.kill()


def test_axes_are_passed_on_as_unknown_rather_than_invented() -> None:
    """**Never fill in an axis.**

    A mesh imported on the wrong axis renders perfectly correctly, so nobody
    finds the mistake by looking - the first sign is a mirrored joint, and by
    then it has been printed. "Unknown" has to survive the journey.
    """
    session = Session()
    try:
        os.environ["SLEEPY_REPORT_AXES"] = "0"
        session.proc.kill()
        session.__init__()  # type: ignore[misc] - restart with the new environment
        kind, out = session.until(
            session.send(
                "image_to_mesh", {"model": "sleepy", "image_path": "x", "seconds": 0.2, "steps": 2}
            ),
            timeout=60,
        )
        assert kind == "result", out
        assert "up_axis" not in out, f"hearth invented an axis: {out.get('up_axis')!r}"
    finally:
        os.environ.pop("SLEEPY_REPORT_AXES", None)
        session.kill()


def test_a_mesh_is_never_left_half_written() -> None:
    """**Contract §9.** A cancelled run must not leave a file that looks finished."""
    session = Session()
    try:
        out_dir = OUT / "atomic"
        kind, out = session.until(
            session.send(
                "image_to_mesh",
                {
                    "model": "sleepy",
                    "image_path": "x",
                    "out_dir": str(out_dir),
                    "seconds": 0.2,
                    "steps": 2,
                },
            ),
            timeout=60,
        )
        assert kind == "result", out
        assert Path(out["mesh_path"]).is_file(), out
        leftovers = list(out_dir.glob("*.tmp"))
        assert not leftovers, f"staging files were left behind: {leftovers}"
    finally:
        session.kill()


def test_a_runner_ends_itself_when_hearth_crashes() -> None:
    """**The orphan case nothing else covers.**

    `shutdown` stops a runner, and killing hearth from outside kills the tree.
    Neither happens when hearth **crashes**: on Windows the child simply carries
    on with a new parent, holding the whole card, and nothing anywhere errors.
    Everything afterwards is several times slower for a reason nobody can see.

    So the runner watches its own parent. This kills hearth without letting it
    tidy up - the crash, as far as the runner is concerned - and waits.
    """
    # **Silent on purpose.** A runner that reports progress finds out its caller
    # is gone the moment a write fails, and one reading stdin finds out when
    # that closes. With either of those in play this test passes whether the
    # watchdog exists or not - which was true of the first version of it.
    os.environ["SLEEPY_SILENT"] = "1"
    session = Session()
    try:
        # **hearth's own pid, not the one `Popen` returned.** Measured
        # 2026-09-03: a venv `python.exe` on this machine re-executes the base
        # interpreter, so `Popen` holds a launcher and hearth is its child.
        # Killing the launcher leaves hearth running and proves nothing; killing
        # the tree takes the runner with it and proves nothing either.
        kind, hello = session.until(session.send("ping"), timeout=15)
        assert kind == "result", hello
        hearth_pid = int(hello["pid"])

        pid = _generating(session, OUT / "orphan", seconds=60.0)
        assert _alive(pid), "the runner did not start"
        assert pid != hearth_pid, "the runner and hearth are the same process"

        # **Killed, not shut down**, and killed on its own: that is a crash as
        # far as everything downstream is concerned.
        _kill(hearth_pid)
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _alive(hearth_pid):
            time.sleep(0.1)
        assert not _alive(hearth_pid), "hearth did not die"

        # The watchdog looks every two seconds; ten is room to spare.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.2)
        assert not _alive(pid), (
            f"the runner ({pid}) outlived a crashed hearth and still holds the GPU"
        )
    finally:
        os.environ.pop("SLEEPY_SILENT", None)
        session.kill()


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
