# SPDX-License-Identifier: MIT
r"""Starting, adopting and stopping ComfyUI. **The process, not the protocol.**

`comfy.py` is the conversation; this is the child process it happens with.
They are separate because the rules are different: everything in `comfy.py` goes
over HTTP to an application hearth does not own, and everything here is a
`Popen` hearth does.

## Why hearth starts it at all

ComfyUI is still an external application - **nothing is ever installed into its
virtual environment, and its code is never touched**. But the GPU is hearth's to
allocate: `free_models`, unloading a runner before an image, and the refusal to
hold two models at once all live here already. Leaving the *start* of the only
other process that takes the whole card to a script the operator runs by hand
meant that the one thing hearth could not do was the one thing that decided how
much VRAM ComfyUI would take. **The launch arguments are the fix for a spill**
(`HEARTH_COMFY_ARGS`), and they belong beside everything else hearth knows about
sharing one card.

This is the same shape as `forge` starting `mincut`: a child process of its own,
started for one purpose, not a second backend.

## Adopting one that is already running

There is one port, so "start my own instead" is not on the menu. An already
running ComfyUI is **adopted**: reported as `ready` with `owned: false`, and
never stopped by hearth. Reloading FLUX costs about a minute, and an operator
who started it by hand across several Blender sessions should not lose it to a
`shutdown` they asked of something else.
"""

from __future__ import annotations

import contextlib
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from . import config, vram

# The states, in the order they can happen (`docs/protocol.md` §4).
ABSENT = "absent"
STARTING = "starting"
READY = "ready"
FAILED = "failed"

# How often the watcher looks. **Not measured**: a person pressing Start on a
# ComfyUI somebody else launched should see it within a few seconds, and the
# look costs one refused connection when nothing is there.
_WATCH_SEC = 5.0


def _port() -> int:
    """The port `HEARTH_COMFY_BASE_URL` names. **One source for the address.**"""
    parsed = urlparse(config.COMFY_BASE_URL)
    return parsed.port or (443 if parsed.scheme == "https" else 80)


def listening_pid(port: int) -> int:
    """Which process is listening on a port, or 0.

    Parsed out of `netstat -ano` rather than asked of a library: **hearth's
    virtual environment holds three packages** and this is not worth a fourth.
    """
    if sys.platform != "win32":
        return 0
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP" or parts[3].upper() != "LISTENING":
            continue
        local = parts[1]
        if local.rsplit(":", 1)[-1] != str(port):
            continue
        try:
            return int(parts[4])
        except ValueError:
            continue
    return 0


def _kill_tree(pid: int, proc: subprocess.Popen[bytes] | None = None) -> None:
    """End a process **and its children**, then collect it.

    **The children matter on Windows.** A venv's `python.exe` re-executes the
    base interpreter, so the process holding the weights is a child of the one
    that was started (measured 2026-09-03; `runner_client` relies on the same
    fact). Killing only the launcher would leave ComfyUI running with the whole
    card and nothing would report it.

    **Collecting it matters everywhere else.** A killed child that is never
    waited for stays in the process table as a zombie, and every way of asking
    "is it still running" says yes - `os.kill(pid, 0)` included. That is not an
    academic point: it failed this repository's own test on Linux, where the
    process had in fact been killed.
    """
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=20,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except (OSError, subprocess.SubprocessError):
            pass  # Fall through to the plain kill below.
    else:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    if proc is not None:
        with contextlib.suppress(subprocess.SubprocessError, OSError):
            proc.wait(timeout=10)


# How long to wait for `/system_stats` before calling ComfyUI absent.
# **The cost of "nothing is there" is this whole timeout on this machine**:
# measured 2026-09-06, connecting to a closed port on 127.0.0.1 is not refused,
# it is dropped, so the answer arrives when the timeout does. A ComfyUI that is
# up answers in single-digit milliseconds, so the number only decides how long a
# negative takes - which is why the probe happens off the control thread.
_PROBE_SEC = 1.0


def _alive(url: str) -> bool:
    """Whether `/system_stats` answers, which is what "loaded" means here."""
    try:
        return httpx.get(f"{url}/system_stats", timeout=_PROBE_SEC).status_code == 200
    except httpx.HTTPError:
        return False


@dataclass
class _State:
    state: str = ABSENT
    owned: bool = False
    pid: int = 0
    why: str = ""


class ComfyProcess:
    """The ComfyUI hearth started, or the one it found. **All the state is here.**"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._s = _State()
        self._proc: subprocess.Popen[bytes] | None = None
        self._logs: list[Any] = []
        self._watching: threading.Thread | None = None
        self._stop_watch = threading.Event()

    # --- Noticing what happened without hearth --------------------------------
    def watch(self) -> None:
        """Keep the state true when ComfyUI is started or stopped elsewhere.

        **This is on a thread because `status` must not block.** Control methods
        are answered on the thread that reads stdin (`docs/protocol.md` §2), so
        a `status` that ran `netstat` would put tens of milliseconds between a
        person pressing cancel and hearth reading it - and `status` is the method
        an interface polls.
        """
        if self._watching is not None:
            return
        self._watching = threading.Thread(target=self._watch, name="hearth-comfy-watch",
                                          daemon=True)
        self._watching.start()

    def _watch(self) -> None:
        while not self._stop_watch.is_set():
            try:
                self._reconcile()
            except OSError as exc:
                print(f"[hearth] could not check on ComfyUI: {exc}", file=sys.stderr)
            self._stop_watch.wait(_WATCH_SEC)

    def _reconcile(self) -> None:
        """One look at whether the world still matches what we last recorded."""
        with self._lock:
            state, owned, pid = self._s.state, self._s.owned, self._s.pid
        if state == STARTING:
            return  # `_await_ready` owns this state until it resolves.
        alive = _alive(config.COMFY_BASE_URL)
        if state in (ABSENT, FAILED) and alive:
            # **Somebody started one by hand.** Adopted, never stopped by us.
            found = listening_pid(_port())
            with self._lock:
                self._s = _State(state=READY, owned=False, pid=found)
            if found:
                vram.SAMPLER.watch(found, "comfyui")
            print(f"[hearth] adopted a ComfyUI already running (pid {found}).", file=sys.stderr)
            return
        if state == READY and not alive:
            # It went away. An owned one that dies is a failure worth naming; an
            # adopted one being stopped by whoever started it is not.
            with self._lock:
                self._s = _State(
                    state=FAILED if owned else ABSENT,
                    why="ComfyUI stopped on its own (see comfyui.err.log)" if owned else "",
                )
                self._proc = None
            vram.SAMPLER.forget(pid)
            self._close_logs()

    # --- What it is doing -----------------------------------------------------
    def status(self) -> dict[str, Any]:
        """The `status.comfy` value (`docs/protocol.md` §4). **Reads state only.**"""
        with self._lock:
            state, owned, pid, why = self._s.state, self._s.owned, self._s.pid, self._s.why
        out: dict[str, Any] = {
            "state": state,
            "owned": owned,
            "pid": pid or None,
            "url": config.COMFY_BASE_URL,
        }
        if why:
            out["why"] = why
        return out

    def state(self) -> str:
        """Just the state, for a caller deciding whether to wait."""
        return str(self.status()["state"])

    # --- Starting -------------------------------------------------------------
    def start(self) -> dict[str, Any]:
        """Start ComfyUI, or adopt the one that is already there. **Returns at once.**

        Waiting for the weights to load takes about a minute, and this is
        answered on the thread that reads stdin (`docs/protocol.md` §2), so the
        wait happens on a thread of its own and the caller watches `status`.

        Returns:
            `state`, `owned` and `pid`, as `comfy_start` promises.

        Raises:
            RuntimeError: If `.env` does not say where ComfyUI is.
        """
        with self._lock:
            if self._s.state in (STARTING, READY):
                return {"state": self._s.state, "owned": self._s.owned, "pid": self._s.pid or None}
        # **Answering is the test, not the port being bound.** A port that is
        # held but silent is not a ComfyUI to submit a workflow to.
        if _alive(config.COMFY_BASE_URL):
            # **Not ours, and never stopped by us.** One port, so there is no
            # second one to start; and a FLUX already loaded is a minute saved.
            existing = listening_pid(_port())
            with self._lock:
                self._s = _State(state=READY, owned=False, pid=existing)
            if existing:
                vram.SAMPLER.watch(existing, "comfyui")
            return {"state": READY, "owned": False, "pid": existing or None}
        self._spawn()
        with self._lock:
            return {"state": self._s.state, "owned": self._s.owned, "pid": self._s.pid or None}

    def _spawn(self) -> None:
        """Launch ComfyUI and start watching for it to answer."""
        python = Path(config.COMFY_PYTHON)
        root = Path(config.COMFY_ROOT)
        if not config.COMFY_PYTHON or not config.COMFY_ROOT:
            raise RuntimeError(
                "hearth cannot start ComfyUI: set HEARTH_COMFY_PYTHON and "
                "HEARTH_COMFY_ROOT in .env, or start ComfyUI yourself."
            )
        main = root / "main.py"
        if not python.is_file():
            raise FileNotFoundError(f"ComfyUI's python is missing: {python}")
        if not main.is_file():
            raise FileNotFoundError(f"ComfyUI is missing: {main}")

        log_dir = root / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        # **The same two files `start-comfyui.ps1` writes**, appended to. An
        # operator diagnosing a spill should not have to know which of the two
        # ways it was started to know where to look.
        out_log = (log_dir / "comfyui.log").open("ab")
        err_log = (log_dir / "comfyui.err.log").open("ab")
        self._logs = [out_log, err_log]

        parsed = urlparse(config.COMFY_BASE_URL)
        # `-u`: unbuffered. Redirected to a file, python block-buffers and the
        # log arrives minutes late, which makes it useless while starting.
        argv = [
            str(python),
            "-u",
            str(main),
            "--listen",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(_port()),
            # **The whole reason hearth starts it.** On this machine
            # `--fp8_e4m3fn-unet` and `--reserve-vram` are what keep FLUX inside
            # the card; see `.env` for the measured values.
            *shlex.split(config.COMFY_ARGS, posix=False),
        ]
        proc = subprocess.Popen(
            argv,
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=out_log,
            stderr=err_log,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self._proc = proc
        with self._lock:
            self._s = _State(state=STARTING, owned=True, pid=proc.pid)
        print(f"[hearth] starting ComfyUI: {' '.join(argv)}", file=sys.stderr)
        threading.Thread(target=self._await_ready, name="hearth-comfy-start", daemon=True).start()

    def _await_ready(self) -> None:
        """Wait for `/system_stats`, then record the pid that is really listening."""
        proc = self._proc
        deadline = time.monotonic() + config.COMFY_START_TIMEOUT_SEC
        while time.monotonic() < deadline:
            # **This thread outlives what it was watching.** A `comfy_stop` a
            # second after a `comfy_start` leaves it polling, and without this it
            # would go on to declare `ready` - resurrecting a ComfyUI that had
            # been stopped, or claiming a later one as ours. Seen on 2026-09-06
            # while writing `tests/test_comfy_process.py`.
            with self._lock:
                mine = self._proc is proc and self._s.state == STARTING
            if not mine or self._stop_watch.is_set():
                return
            if proc is not None and proc.poll() is not None:
                self._failed(proc, f"ComfyUI exited with code {proc.returncode}")
                return
            if _alive(config.COMFY_BASE_URL):
                # **The listening pid, not the one that was spawned.** A venv
                # launcher starts the real interpreter as a child, and it is the
                # child that holds the VRAM.
                real = listening_pid(_port()) or (proc.pid if proc else 0)
                with self._lock:
                    # Looked at again under the lock: `_alive` takes up to three
                    # seconds and a `comfy_stop` fits inside it.
                    if self._proc is not proc or self._s.state != STARTING:
                        return
                    self._s = _State(state=READY, owned=True, pid=real)
                vram.SAMPLER.watch(real, "comfyui")
                print(f"[hearth] ComfyUI is up (pid {real}).", file=sys.stderr)
                return
            time.sleep(2.0)
        self._failed(
            proc,
            f"ComfyUI did not answer within {config.COMFY_START_TIMEOUT_SEC} seconds "
            f"(see {Path(config.COMFY_ROOT) / 'logs' / 'comfyui.err.log'})"
        )

    def _failed(self, proc: subprocess.Popen[bytes] | None, why: str) -> None:
        with self._lock:
            if self._proc is not proc or self._s.state != STARTING:
                # Somebody stopped it while we were waiting, or started another.
                # **Not a failure**, and not this thread's state to write.
                return
            pid = self._s.pid
            self._s = _State(state=FAILED, owned=False, pid=0, why=why)
            self._proc = None
        print(f"[hearth] {why}", file=sys.stderr)
        # A ComfyUI that half-started still holds the card, so it does not get
        # to stay just because it never answered.
        if pid:
            _kill_tree(pid, proc)
            vram.SAMPLER.forget(pid)
        self._close_logs()

    # --- Stopping -------------------------------------------------------------
    def stop(self) -> dict[str, Any]:
        """Stop ComfyUI, **but only the one hearth started**.

        Returns:
            `stopped`, and `why` when it was not.
        """
        with self._lock:
            state, owned, pid = self._s.state, self._s.owned, self._s.pid
            proc = self._proc
        if state in (ABSENT, FAILED):
            return {"stopped": False, "why": "ComfyUI is not running"}
        if not owned:
            return {
                "stopped": False,
                "why": (
                    f"ComfyUI (pid {pid}) was not started by hearth, so it is left running. "
                    "Stop it where it was started."
                ),
            }
        # The spawned process is the launcher; `/T` takes the interpreter with it.
        _kill_tree(proc.pid if proc is not None else pid, proc)
        if pid:
            vram.SAMPLER.forget(pid)
        with self._lock:
            self._s = _State()
            self._proc = None
        self._close_logs()
        return {"stopped": True, "why": None}

    def shutdown(self) -> None:
        """Stop watching, and take an owned ComfyUI down with hearth.

        **Only an owned one.** An adopted ComfyUI outlives the hearth that found
        it, which is the whole reason adopting is a state of its own.
        """
        self._stop_watch.set()
        with self._lock:
            owned = self._s.owned
        if owned:
            self.stop()

    def _close_logs(self) -> None:
        for handle in self._logs:
            try:
                handle.close()
            except OSError:
                pass
        self._logs = []

    # --- What a spill looks like ---------------------------------------------
    def spilled(self) -> tuple[float, float, int] | None:
        """Whether ComfyUI is over the shared-memory threshold right now.

        Returns:
            `(shared_gb, dedicated_gb, pid)` when it is, otherwise None.
        """
        with self._lock:
            pid = self._s.pid if self._s.state == READY else 0
        if not pid:
            return None
        over = vram.SAMPLER.spilled(pid)
        if over is None:
            return None
        return over[0], over[1], pid


# **One ComfyUI per hearth**, because there is one port and one card.
COMFY = ComfyProcess()
