# SPDX-License-Identifier: MIT
r"""Starting, adopting and stopping ComfyUI. **Without ComfyUI, and in seconds.**

The four states `comfy_start` moves through are the whole of what an interface
draws, and each of them was easy to get wrong in a way that only shows up on a
machine with a GPU:

1. **`absent` is not `failed`.** Nothing listening is the ordinary case, not an
   error, and an interface that says "failed to start" when nobody has pressed
   Start yet is telling the user to go and read a log that is empty.
2. **A ComfyUI that is already running is adopted, never duplicated** - there is
   one port - and `owned: false` is what makes that visible.
3. **`comfy_stop` stops only what hearth started.** Stopping somebody else's
   process is the one thing hearth never does on a user's behalf
   (`docs/protocol.md` §6), and it has to *say* it did not rather than answer
   "stopped" and leave it running.
4. **A child that never answers ends in `failed`, and is not left holding the
   card.** It is the case that costs the most and happens least often.

ComfyUI is replaced by a stand-in HTTP server, and "the process hearth started"
by a python that sleeps. **No GPU, no weights, no ComfyUI.**

Run it with hearth's own virtual environment::

    .venv\Scripts\python.exe .\tests\test_comfy_process.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _free_port() -> int:
    """A port nothing is on, so this test never fights the real ComfyUI."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# **Set before hearth is imported.** `config` reads the environment once, and
# pointing it at the operator's real ComfyUI would make this test stop it.
PORT = _free_port()
os.environ["HEARTH_COMFY_BASE_URL"] = f"http://127.0.0.1:{PORT}"
os.environ["HEARTH_COMFY_AUTOSTART"] = "0"
os.environ["HEARTH_COMFY_START_TIMEOUT_SEC"] = "8"

from hearth import comfy_process  # noqa: E402


class _Comfy:
    """Something that answers `/system_stats` with a 200. That is all `ready` means."""

    def __init__(self, port: int) -> None:
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:
                """Silent: the test's own output is the only thing worth reading."""

            def do_GET(self) -> None:  # noqa: N802 - the base class names it
                body = json.dumps({"system": {}}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def _fresh() -> comfy_process.ComfyProcess:
    """A ComfyProcess with nothing remembered."""
    return comfy_process.ComfyProcess()


def _wait_for(process: comfy_process.ComfyProcess, state: str, seconds: float) -> str:
    """Wait for a state, and return whatever it actually reached."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        now = process.state()
        if now == state:
            return now
        time.sleep(0.1)
    return process.state()


def test_nothing_listening_is_absent_not_failed() -> None:
    """**The ordinary case.** Nobody has started it, and that is not an error."""
    process = _fresh()
    status = process.status()
    assert status["state"] == comfy_process.ABSENT, status
    assert status["owned"] is False, status
    assert status["pid"] is None, status
    assert status["url"].endswith(str(PORT)), status


def test_a_running_comfyui_is_adopted_and_survives_stop() -> None:
    """**One port, so there is nothing else to do but adopt it.**

    And the whole reason adopting is a state of its own: a FLUX already loaded is
    about a minute, and a `shutdown` asked of hearth must not take it.
    """
    fake = _Comfy(PORT)
    try:
        process = _fresh()
        started = process.start()
        assert started["state"] == comfy_process.READY, started
        assert started["owned"] is False, "an adopted ComfyUI must not be reported as ours"

        stopped = process.stop()
        assert stopped["stopped"] is False, "hearth stopped a ComfyUI it did not start"
        assert "not started by hearth" in str(stopped["why"]), stopped
        # Still answering: nothing was killed.
        assert process.status()["state"] == comfy_process.READY
    finally:
        fake.stop()


def test_the_watcher_notices_one_started_by_hand() -> None:
    """A person starts ComfyUI while hearth is up. **The state has to follow.**

    Without this the only way to notice was to ask, and asking meant `netstat` on
    the thread that reads stdin - which is the thread a `cancel` arrives on.
    """
    process = _fresh()
    assert process.state() == comfy_process.ABSENT
    fake = _Comfy(PORT)
    try:
        process.watch()
        assert _wait_for(process, comfy_process.READY, 20.0) == comfy_process.READY, (
            "a ComfyUI started by hand was never noticed"
        )
        assert process.status()["owned"] is False
    finally:
        process.shutdown()
        fake.stop()


def test_an_adopted_comfyui_going_away_is_absent_again() -> None:
    """Whoever started it stopped it. **That is not a failure of hearth's.**"""
    fake = _Comfy(PORT)
    process = _fresh()
    try:
        process.start()
        assert process.state() == comfy_process.READY
        process.watch()
        fake.stop()
        assert _wait_for(process, comfy_process.ABSENT, 20.0) == comfy_process.ABSENT, (
            "hearth went on reporting a ComfyUI that had gone"
        )
    finally:
        process.shutdown()


def test_starting_returns_at_once_and_the_state_follows() -> None:
    """**`comfy_start` answers in milliseconds**, and `starting` is a real state.

    Loading FLUX is about a minute. A control method that waited for it would
    stop `cancel` being read for that minute (`docs/protocol.md` §2), so the
    answer is "starting" and the caller watches `status`.
    """
    # A python that sleeps stands in for ComfyUI: it is started the same way and
    # it never answers, which is exactly the `starting` state.
    root = REPO_ROOT / "output" / "test-comfy-process"
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    comfy_process.config.COMFY_PYTHON = sys.executable
    comfy_process.config.COMFY_ROOT = str(root)
    comfy_process.config.COMFY_ARGS = ""
    process = _fresh()
    try:
        began = time.perf_counter()
        answer = process.start()
        elapsed = time.perf_counter() - began
        assert answer["state"] == comfy_process.STARTING, answer
        assert answer["owned"] is True, answer
        assert answer["pid"], "a started ComfyUI has a pid"
        # **It spawns a process; it does not wait for weights.** The floor is
        # one probe for a ComfyUI that might already be running, and on this
        # machine a closed port is dropped rather than refused, so that probe
        # costs its timeout (measured 2026-09-06). A minute is the number this
        # is guarding against, not a second.
        assert elapsed < 3.0, f"comfy_start took {elapsed:.2f}s, which a caller feels"

        stopped = process.stop()
        assert stopped["stopped"] is True, stopped
        assert process.state() == comfy_process.ABSENT
    finally:
        process.shutdown()


def test_one_that_never_answers_ends_in_failed_and_is_not_left_running() -> None:
    """**The expensive case.** A half-started ComfyUI still holds the whole card.

    So the timeout is not just reported: the process is ended. Leaving it would
    take 32 GB with nothing to show for it and nothing to say so.
    """
    root = REPO_ROOT / "output" / "test-comfy-process"
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text("import time\ntime.sleep(300)\n", encoding="utf-8")
    comfy_process.config.COMFY_PYTHON = sys.executable
    comfy_process.config.COMFY_ROOT = str(root)
    comfy_process.config.COMFY_ARGS = ""
    process = _fresh()
    try:
        answer = process.start()
        pid = int(answer["pid"] or 0)
        assert _wait_for(process, comfy_process.FAILED, 30.0) == comfy_process.FAILED, (
            "a ComfyUI that never answered was left reported as starting"
        )
        assert "did not answer" in str(process.status().get("why", "")), process.status()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.1)
        assert not _alive(pid), f"the ComfyUI that failed to start ({pid}) is still running"
    finally:
        process.shutdown()


def _alive(pid: int) -> bool:
    """Whether a process id is still running. **No new dependency for this.**"""
    if not pid:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    import subprocess  # noqa: PLC0415 - only this helper needs it

    done = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return str(pid) in done.stdout


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
