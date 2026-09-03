# SPDX-License-Identifier: MIT
"""Talking to a runner, which is a child process. **Blocking is fine**: requests are serial.

Two things are not optional, whatever the caller looks like:

- **A thread has to drain stderr.** Left unread, the pipe fills and the runner
  stops in place.
- **Progress has to be relayed onward.** A generation can take many minutes; if
  the caller sees nothing for that long it has no way to tell work from a hang.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Protocol

# Where progress goes: `(stage, message)`, plus **counted** steps (`step`, and
# `total` only when the length is known).
# **Nothing estimated arrives here** - no ETA, no overall percentage (contract §8).
class Relay(Protocol):
    def __call__(
        self,
        stage: str,
        message: str = "",
        *,
        step: int | None = None,
        total: int | None = None,
    ) -> None: ...


def _as_int(value: Any) -> int | None:
    """Let through only integers usable as a step count (**drop anything else**).

    A runner emitting a broken `step` can still be generating perfectly well.
    **Never fail a generation over its progress display.**
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number >= 0 else None


class RunnerError(RuntimeError):
    """A runner would not start, or the conversation with it failed."""


class RunnerProcess:
    """One runner. **Only one is ever alive at a time**, because there is one GPU."""

    def __init__(self, name: str, spec: dict[str, str]) -> None:
        self.name = name
        self._spec = spec
        self._proc: subprocess.Popen[str] | None = None
        self._stderr: deque[str] = deque(maxlen=400)
        self._next_id = 1
        # **One conversation at a time with this process.** Control methods are
        # answered on another thread (`docs/protocol.md` §2), and two requests
        # interleaving their lines on one pipe is not protocol.
        self._call_lock = threading.Lock()
        # **Two threads reach here.** `capabilities` is answered on the control
        # thread while the GPU thread generates, and both may find the process
        # stopped; without this they would each start one.
        self._start_lock = threading.Lock()

    def is_running(self) -> bool:
        """Whether the process is alive."""
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Start the runner as a child process.

        Raises:
            RunnerError: If the configuration is incomplete or the python is missing.
        """
        with self._start_lock:
            self._start()

    def _start(self) -> None:
        """Start it, with the start lock already held."""
        if self.is_running():
            return
        python = Path(self._spec.get("python", ""))
        module = self._spec.get("module", "")
        cwd = self._spec.get("cwd", "")
        if not module:
            raise RunnerError(f"{self.name}: no module configured (check .env)")
        if not python.is_file():
            raise RunnerError(f"{self.name}: python not found: {python}")

        env = dict(os.environ)
        # **Force UTF-8.** On the default code page, any non-ASCII progress
        # message raises on the way out.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [str(python), "-m", module],
            cwd=cwd or None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def stop(self, *, timeout: float = 10.0) -> None:
        """Ask it to shut down, and end it if it does not."""
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"id": 0, "method": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.stdin.close()
            proc.wait(timeout=timeout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            proc.kill()

    def kill(self) -> None:
        """End the process now, without asking it to stop.

        **This is how a generation is cancelled** (`docs/runner_contract.md`
        §9): a torch loop does not check for anything, so ending the process is
        the only thing that stops it and the only thing that reliably returns the
        VRAM. A `call` waiting on this process fails as soon as its stdout
        closes.
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.kill()

    def call(
        self, method: str, params: dict[str, Any] | None = None, *, relay: Relay | None = None
    ) -> dict[str, Any]:
        """Send one request and wait for its `result`, relaying progress as it arrives.

        Args:
            method: The method name.
            params: Its arguments.
            relay: Where progress goes.

        Returns:
            The contents of `result`.

        Raises:
            RunnerError: If the runner is not running, died partway, or answered
                with an error.
        """
        # **Serialised per process.** More than one thread reaches a runner:
        # `capabilities` is answered on the control thread while the GPU thread
        # is generating.
        with self._call_lock:
            return self._converse(method, params, relay)

    def _converse(
        self, method: str, params: dict[str, Any] | None, relay: Relay | None
    ) -> dict[str, Any]:
        """Send one request and read until this runner answers it."""
        if not self.is_running() or self._proc is None or self._proc.stdin is None:
            raise RunnerError(f"{self.name}: the runner is not running")
        assert self._proc.stdout is not None

        request_id = self._next_id
        self._next_id += 1
        payload = {"id": request_id, "method": method, "params": params or {}}
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        for raw in self._proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                # A line that is not protocol: a sign the runner's stdout guard
                # has been breached by something printing directly.
                self._stderr.append(f"[protocol] unparsable line: {line[:200]}")
                continue
            if int(event.get("id", -1)) != request_id:
                continue
            kind = event.get("event")
            if kind == "progress":
                if relay is not None:
                    # **Pass the counts on.** Dropping them leaves everything
                    # downstream knowing only that the runner is alive.
                    relay(
                        str(event.get("stage", "")),
                        str(event.get("message", "")),
                        step=_as_int(event.get("step")),
                        total=_as_int(event.get("total")),
                    )
            elif kind == "result":
                return dict(event.get("result") or {})
            elif kind == "error":
                err = event.get("error") or {}
                raise RunnerError(f"{self.name}: {err.get('type')}: {err.get('message')}")

        raise RunnerError(f"{self.name}: the runner exited:\n{self.stderr_tail()}")

    def stderr_tail(self, lines: int = 20) -> str:
        """The tail of the runner's stderr, for diagnosis."""
        return "\n".join(list(self._stderr)[-lines:])

    def _drain_stderr(self) -> None:
        """Drain stderr. **Left unread, the pipe fills and the runner stops.**"""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        for raw in proc.stderr:
            self._stderr.append(raw.rstrip())
