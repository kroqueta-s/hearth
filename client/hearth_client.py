# SPDX-License-Identifier: MIT
"""A client for hearth. **One file, no dependencies. Copy it into your application.**

This is the reference implementation of `docs/protocol.md`. It is a single file
with nothing but the standard library behind it **so that it can be vendored**:
dropped into a Blender add-on, a script, or anything else that cannot install
packages into the python it is running on.

There are two ways to use it.

**Blocking**, for a script or a command line, where waiting is the point::

    with Hearth.start(python) as hearth:
        state = hearth.call("status")
        out = hearth.call("text_to_image", {"prompt": "a small brass key"},
                          on_progress=print_it)

**Non-blocking**, for a user interface, where waiting is the one thing you must
not do::

    hearth = Hearth.start(python)
    hearth.send("image_to_mesh", params, done=on_done, on_progress=on_progress)
    ...
    hearth.poll()   # from a timer, on the thread that owns the interface

**Both live on one process.** Starting hearth for each request throws away the
loaded model and pays the load again - tens of seconds, every time - so **keep
one instance for as long as the user is working**.

## Two things that are not optional

- **stderr is drained on a thread.** Left unread, the pipe fills and hearth
  stops. This class does it for you.
- **Replies are matched by `id`, never by order.** Control methods are answered
  while a generation runs (`docs/protocol.md` §2), so the next line to arrive is
  routinely not the answer to the last thing you sent.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

# The version of `docs/protocol.md` this client was written against.
PROTOCOL_VERSION = 1

# Called with `(ok, payload)`: the result when ok, the error object when not.
Done = Callable[[bool, dict[str, Any]], None]
# Called with `(stage, message, step, total)`. **`step` and `total` are None
# unless they were counted**, and a percentage without a `total` is an invention
# (`docs/runner_contract.md` §8).
OnProgress = Callable[[str, str, "int | None", "int | None"], None]


class HearthError(RuntimeError):
    """hearth would not start, or the conversation with it failed."""


class RequestFailed(RuntimeError):
    """A request was answered with an error.

    Attributes:
        type: The name hearth gave it, such as `CanceledError`. **Branch on
            this**, not on the message.
        message: What to show a person.
    """

    def __init__(self, error: dict[str, Any]) -> None:
        self.type = str(error.get("type", "Error"))
        self.message = str(error.get("message", ""))
        super().__init__(f"{self.type}: {self.message}")


class Hearth:
    """One hearth process, and every request in flight to it."""

    def __init__(self, proc: subprocess.Popen[str]) -> None:
        self._proc = proc
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=400)
        self._pending: dict[int, tuple[Done, OnProgress | None]] = {}
        self._next_id = 1
        self._lock = threading.Lock()
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    # --- Starting and stopping ----------------------------------------------
    @classmethod
    def start(cls, python: str | Path, repo_root: str | Path | None = None) -> Hearth:
        """Start hearth as a child process.

        Args:
            python: The python of hearth's own virtual environment. **Not the
                one your application runs on**: hearth has its own, and in a
                Blender add-on the two are different versions.
            repo_root: hearth's directory. Defaults to the parent of this file's
                directory, which is right when this file has not been vendored.

        Returns:
            A live client.

        Raises:
            HearthError: If the python or the package is not there.
        """
        python = Path(python)
        root = Path(repo_root) if repo_root else Path(__file__).resolve().parent.parent
        if not python.is_file():
            raise HearthError(f"python not found: {python}")
        if not (root / "hearth" / "__main__.py").is_file():
            raise HearthError(f"the hearth package is not in {root}")

        env = dict(os.environ)
        # **Force UTF-8.** On the default code page a non-ASCII progress message
        # raises on the way out.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            [str(python), "-m", "hearth"],
            cwd=str(root),
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
        return cls(proc)

    def is_running(self) -> bool:
        """Whether hearth is alive."""
        return self._proc.poll() is None

    def stop(self, *, timeout: float = 10.0) -> None:
        """Ask hearth to shut down, and end it if it will not.

        **hearth ends its runners on the way out**, so the VRAM comes back
        without anything else being asked for.
        """
        if not self.is_running():
            return
        try:
            if self._proc.stdin is not None:
                self._write({"id": 0, "method": "shutdown"})
                self._proc.stdin.close()
            self._proc.wait(timeout=timeout)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            self._proc.kill()
        finally:
            self._fail_pending("hearth was stopped")

    def __enter__(self) -> Hearth:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # --- Sending -------------------------------------------------------------
    def send(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        done: Done,
        on_progress: OnProgress | None = None,
    ) -> int:
        """Send one request. **This does not wait**; the answer arrives in `poll`.

        Args:
            method: The method name.
            params: Its arguments.
            done: Called with `(True, result)` or `(False, error)`.
            on_progress: Called for each progress line.

        Returns:
            The request id.

        Raises:
            HearthError: If hearth is not running.
        """
        if not self.is_running() or self._proc.stdin is None:
            raise HearthError("hearth is not running")
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._pending[request_id] = (done, on_progress)
        self._write({"id": request_id, "method": method, "params": params or {}})
        return request_id

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        on_progress: OnProgress | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send one request and wait for its answer.

        **Do not call this from a thread that draws a user interface.** A
        generation takes minutes, and this blocks for all of them.

        Args:
            method: The method name.
            params: Its arguments.
            on_progress: Called for each progress line.
            timeout: Seconds to wait. None waits as long as it takes, which is
                usually right: a generation has no useful deadline.

        Returns:
            The result.

        Raises:
            RequestFailed: If hearth answered with an error.
            HearthError: If hearth died, or the timeout ran out.
        """
        answer: queue.Queue[tuple[bool, dict[str, Any]]] = queue.Queue(maxsize=1)
        self.send(method, params, done=lambda ok, payload: answer.put((ok, payload)),
                  on_progress=on_progress)
        deadline_step = 0.05
        waited = 0.0
        while True:
            self.poll()
            try:
                ok, payload = answer.get(timeout=deadline_step)
            except queue.Empty:
                waited += deadline_step
                if timeout is not None and waited >= timeout:
                    raise HearthError(f"{method} did not answer within {timeout}s") from None
                continue
            if ok:
                return payload
            raise RequestFailed(payload)

    # --- Receiving -----------------------------------------------------------
    def poll(self) -> int:
        """Deliver whatever has arrived. **Call this from your timer.**

        In an interface, this is the only place your callbacks run, which is what
        keeps them on the thread that is allowed to touch it.

        Returns:
            How many events were delivered.
        """
        handled = 0
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break
            handled += 1
            self._dispatch(event)
        # **If hearth died, nothing will ever answer.** Fail what is waiting
        # rather than leaving an interface spinning forever.
        if self._pending and not self.is_running():
            self._fail_pending("hearth exited")
        return handled

    def has_pending(self) -> bool:
        """Whether anything is still waiting for an answer."""
        with self._lock:
            return bool(self._pending)

    def stderr_tail(self, lines: int = 20) -> str:
        """The tail of hearth's stderr, for diagnosis."""
        return "\n".join(list(self._stderr)[-lines:])

    # --- Internals -----------------------------------------------------------
    def _write(self, payload: dict[str, Any]) -> None:
        """Write one request line."""
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise HearthError(f"could not write to hearth: {exc}") from exc

    def _dispatch(self, event: dict[str, Any]) -> None:
        """Deliver one event to the request it belongs to."""
        try:
            request_id = int(event.get("id", -1))
        except (TypeError, ValueError):
            return
        with self._lock:
            entry = self._pending.get(request_id)
        if entry is None:
            return
        done, on_progress = entry
        kind = event.get("event")
        if kind == "progress":
            if on_progress is not None:
                on_progress(
                    str(event.get("stage", "")),
                    str(event.get("message", "")),
                    _as_int(event.get("step")),
                    _as_int(event.get("total")),
                )
            return
        with self._lock:
            self._pending.pop(request_id, None)
        if kind == "result":
            done(True, dict(event.get("result") or {}))
        elif kind == "error":
            done(False, dict(event.get("error") or {}))

    def _fail_pending(self, why: str) -> None:
        """Fail everything still waiting, with the tail of stderr attached."""
        with self._lock:
            waiting = list(self._pending.values())
            self._pending.clear()
        if not waiting:
            return
        payload = {"type": "HearthError", "message": f"{why}:\n{self.stderr_tail()}"}
        for done, _ in waiting:
            done(False, dict(payload))

    def _pump_stdout(self) -> None:
        """Read protocol lines onto the queue."""
        stream = self._proc.stdout
        if stream is None:
            return
        for raw in stream:
            line = raw.lstrip("﻿").strip()
            if not line:
                continue
            try:
                self._events.put(json.loads(line))
            except ValueError:
                # Not protocol: a sign something printed to stdout past the
                # guard. Keep it for diagnosis rather than dropping it.
                self._stderr.append(f"[protocol] unparsable line: {line[:200]}")

    def _pump_stderr(self) -> None:
        """Drain stderr. **Left unread, the pipe fills and hearth stops.**"""
        stream = self._proc.stderr
        if stream is None:
            return
        for raw in stream:
            self._stderr.append(raw.rstrip())


def _as_int(value: Any) -> int | None:
    """Let through only what can be a counted step. **Drop anything else.**

    A broken `step` must never be the thing that fails a generation.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number >= 0 else None


class Flow:
    """Runs steps one after another, handing each one's output to the next.

    **Chaining belongs to the caller** (`docs/protocol.md` §3.2), and this is the
    caller's side of it: a list of steps, run in order, where the image a step
    produced becomes the input of the one after it.

    It is deliberately small. **It does not decide anything**: which models, which
    steps, and where to stop are the application's to choose, because only the
    application knows whether a person is about to look at the intermediate
    image. What this does is spare every application from writing the same
    bookkeeping.

    Each step is `{"method": ..., "params": {...}}`. Before a step runs, the
    output of the one before it is filled in:

    - a step that takes `image_path` gets the `image_path` that came out,
    - a step that takes `mesh_path` gets the `mesh_path` that came out,

    and **anything the step already names is left alone**, so a step can always
    override what it is given.
    """

    # What one step's result is called on the way into the next step's arguments.
    CARRIES = ("image_path", "mesh_path")

    def __init__(self, hearth: Hearth, steps: list[dict[str, Any]]) -> None:
        self.hearth = hearth
        self.steps = list(steps)
        self.results: list[dict[str, Any]] = []

    def run(
        self,
        *,
        on_progress: OnProgress | None = None,
        on_step: Callable[[int, str, dict[str, Any]], None] | None = None,
        warm_next: bool = True,
    ) -> list[dict[str, Any]]:
        """Run every step in order, blocking until they are done.

        Args:
            on_progress: Passed through to each request.
            on_step: Called as `(index, method, result)` when a step finishes.
                **This is where an interface shows the intermediate image.**
            warm_next: Ask hearth to get the next step's model ready while this
                one runs. Advice only, and free when it is declined.

        Returns:
            One result per step, in order.

        Raises:
            RequestFailed: From the step that failed. **The steps before it
                still happened**, and their results are in `self.results`.
        """
        for index, step in enumerate(self.steps):
            method = str(step["method"])
            params = dict(step.get("params") or {})
            if self.results:
                self._carry(self.results[-1], params)
            if warm_next:
                self._warm_after(index)
            result = self.hearth.call(method, params, on_progress=on_progress)
            self.results.append(result)
            if on_step is not None:
                on_step(index, method, result)
        return self.results

    def _carry(self, previous: dict[str, Any], params: dict[str, Any]) -> None:
        """Fill this step's inputs in from the last step's outputs."""
        for key in self.CARRIES:
            if key in previous and key not in params:
                params[key] = previous[key]

    def _warm_after(self, index: int) -> None:
        """Get the next step's model ready, if it is a different one.

        **Nothing here waits and nothing here fails.** A warm that does not
        happen costs the load it would have saved and nothing else.
        """
        following = self.steps[index + 1 : index + 2]
        if not following:
            return
        model = str(following[0].get("params", {}).get("model") or "")
        current = str(self.steps[index].get("params", {}).get("model") or "")
        if not model or model == current:
            return
        try:
            self.hearth.send("warm", {"model": model}, done=lambda ok, payload: None)
        except HearthError:
            pass
