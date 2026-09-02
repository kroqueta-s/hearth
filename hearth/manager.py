# SPDX-License-Identifier: MIT
"""Starts runners, switches between them, and holds the state.

**Only one model is loaded at a time.** There is one GPU, so switching means
**unloading before loading**; overlapping the two goes past the VRAM the card
actually has.

**Two threads reach this class** (`docs/protocol.md` §2): the one draining the
GPU queue, and the one answering control methods. Everything that touches the
state below takes `_lock`, and the sections that hold it are short - **a load is
not performed under the lock**, or a `status` during a load would wait for it.

hearth does not interpret a runner's `params`. **Only the runner knows what its
own values mean**, so validation is left to it and the arguments pass straight
through (`docs/runner_contract.md` §3).
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Any

from . import config
from .runner_client import Relay, RunnerError, RunnerProcess


class GpuBusyError(RuntimeError):
    """Another process already holds the GPU, so no runner can be loaded."""


class CanceledError(RuntimeError):
    """The request was cancelled by the caller (`docs/runner_contract.md` §10)."""


def assert_gpu_free() -> None:
    """Check that nothing is listening on the "GPU is busy" port.

    Two processes sharing the VRAM does not halve the speed; both fall back to
    paging and become drastically slower. **Nothing is stopped for you**: ending
    someone else's process is the operator's call, not this program's.

    The check is skipped when `HEARTH_GPU_BUSY_PORT` is 0.

    Raises:
        GpuBusyError: If the port answers.
    """
    port = config.GPU_BUSY_PORT
    if port <= 0:
        return
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        reachable = sock.connect_ex(("127.0.0.1", port)) == 0
    if reachable:
        raise GpuBusyError(
            f"something is listening on port {port}, which HEARTH_GPU_BUSY_PORT "
            "says means the GPU is taken. Stop it first, or set the port to 0 "
            "in .env to disable this check."
        )


class Manager:
    """Supervises the runners. **All the state lives here.**"""

    def __init__(self) -> None:
        self._runners: dict[str, RunnerProcess] = {}
        self._capabilities: dict[str, dict[str, Any]] = {}
        self._loaded: str | None = None
        self._busy: str | None = None
        self._canceling = False
        self._lock = threading.RLock()
        self._gpu_claim: socket.socket | None = None

    # --- Inventory and capabilities -----------------------------------------
    def available(self) -> list[str]:
        """The runners declared in `.env`."""
        return config.runner_names()

    def known_capabilities(self) -> dict[str, dict[str, Any]]:
        """The capability tables already asked for. **This starts nothing.**

        `status` uses this: asking every runner at startup costs starting every
        runner's python, and that is felt when a window is opening
        (`docs/protocol.md` §4).
        """
        with self._lock:
            return dict(self._capabilities)

    def capabilities(self, name: str) -> dict[str, Any]:
        """Return a runner's capabilities (**without loading its weights**).

        The answer is remembered. Answering `capabilities` without loading a
        model is part of the contract, so asking is cheap - but it does start
        that runner's process, so it is asked for on demand rather than for
        everything at once.

        Args:
            name: The runner's name.

        Returns:
            The shape described in `docs/runner_contract.md` §3.

        Raises:
            RunnerError: If the runner is unknown or would not start.
        """
        with self._lock:
            if name in self._capabilities:
                return self._capabilities[name]
        runner = self._runner(name)
        started_here = not runner.is_running()
        runner.start()
        try:
            caps = runner.call("capabilities")
        finally:
            # If it was only started to ask, put it back down.
            with self._lock:
                spare = started_here and self._loaded != name and self._busy != name
            if spare:
                runner.stop()
        with self._lock:
            self._capabilities[name] = caps
        return caps

    def all_capabilities(self) -> dict[str, dict[str, Any]]:
        """Every declared runner's capabilities. **Failures carry their reason.**"""
        out: dict[str, dict[str, Any]] = {}
        for name in self.available():
            try:
                out[name] = self.capabilities(name)
            except (RunnerError, OSError) as exc:
                out[name] = {"name": name, "error": str(exc)}
        return out

    # --- Loading and switching ----------------------------------------------
    def loaded(self) -> str | None:
        """The runner whose weights are loaded, or None."""
        with self._lock:
            self._forget_dead()
            return self._loaded

    def busy(self) -> str | None:
        """The runner that is generating right now, or None."""
        with self._lock:
            return self._busy

    def load(self, name: str, relay: Relay | None = None) -> dict[str, Any]:
        """Switch to a runner and load its weights.

        **Anything already loaded comes down first.** Two models at once do not
        fit.

        Args:
            name: The runner's name.
            relay: Where progress goes.

        Returns:
            The runner's `load` result plus `loaded` (the name).

        Raises:
            GpuBusyError: If another process, or another hearth, holds the GPU.
            CanceledError: If the caller cancelled while it was loading.
            RunnerError: If the runner is unknown or loading failed.
        """
        assert_gpu_free()
        with self._lock:
            self._forget_dead()
            if self._loaded == name:
                return {"loaded": name, "elapsed_sec": 0.0, "already": True}
            current = self._loaded
        if current is not None:
            if relay is not None:
                relay("unload", f"unloading {current} to free the VRAM")
            self.unload(relay=relay)

        runner = self._runner(name)
        spawn_started = time.perf_counter()
        runner.start()
        spawn_sec = time.perf_counter() - spawn_started
        self._claim_gpu()
        # **A load is cancellable too.** It takes tens of seconds, and inside a
        # flow of several steps that is a real part of the wait; a cancel that
        # answered "nothing is generating" for all of it would be useless
        # exactly when someone is waiting.
        self._begin(name)
        try:
            result = runner.call("load", relay=relay)
        except RunnerError:
            self._release_gpu()
            canceled = self._canceled_instead(f"loading {name}")
            if canceled is not None:
                raise canceled from None
            raise
        except BaseException:
            self._release_gpu()
            raise
        finally:
            self._end()
        with self._lock:
            self._loaded = name
        # **Starting and loading are reported separately.** Without knowing
        # which of the two is the expensive one, there is nothing to act on.
        return {"loaded": name, "spawn_sec": round(spawn_sec, 2), **result}

    def unload(self, relay: Relay | None = None) -> dict[str, Any]:
        """Unload the current model and end its runner.

        **The whole process is ended**, because keeping it alive buys nothing
        and there is no reliable way to make torch's allocator give the VRAM
        back.
        """
        with self._lock:
            name = self._loaded
            self._loaded = None
        if name is None:
            self._release_gpu()
            return {"unloaded": False}
        runner = self._runners.get(name)
        if runner is None:
            self._release_gpu()
            return {"unloaded": False}
        # **Keep what the runner reports.** `vram_used_gb` is the only number
        # that shows whether a switch actually returned the memory, and hearth
        # holds no torch of its own to measure it with.
        reported: dict[str, Any] = {}
        try:
            reported = runner.call("unload", relay=relay) or {}
        except RunnerError:
            pass  # It is about to be ended anyway, so a failure here costs nothing.
        stop_started = time.perf_counter()
        runner.stop()
        self._release_gpu()
        return {
            "unloaded": True,
            "was": name,
            "vram_used_gb": reported.get("vram_used_gb"),
            "stop_sec": round(time.perf_counter() - stop_started, 2),
        }

    # --- Generating ----------------------------------------------------------
    def generate(
        self, name: str, method: str, params: dict[str, Any], relay: Relay | None = None
    ) -> dict[str, Any]:
        """Call a runner's generating method, switching to it first if needed.

        Args:
            name: The runner's name.
            method: `image_to_mesh` and the like. **Only what the contract names.**
            params: Passed through untouched. **hearth does not check them.**
            relay: Where progress goes.

        Returns:
            The runner's result plus `model` (the runner that produced it).

        Raises:
            CanceledError: If the caller cancelled it.
            RunnerError: If the runner does not support the method, or failed.
        """
        caps = self.capabilities(name).get("capabilities", {})
        if not caps.get(method, False):
            raise RunnerError(f"{name} does not support {method} (check its capabilities)")
        with self._lock:
            self._forget_dead()
            needs_load = self._loaded != name
        if needs_load:
            self.load(name, relay=relay)
        self._begin(name)
        try:
            result = self._runners[name].call(method, params, relay=relay)
        except RunnerError:
            canceled = self._canceled_instead(f"{method} on {name}")
            if canceled is not None:
                raise canceled from None
            raise
        finally:
            self._end()
        return {"model": name, **result}

    def cancel(self) -> dict[str, Any]:
        """End whatever is generating right now, by ending its process.

        **There is no gentler way** (`docs/runner_contract.md` §10). The price is
        that the weights go with it and the next generation pays a full load.

        Returns:
            Whether anything was cancelled, and what it was.
        """
        with self._lock:
            name = self._busy
            if name is None:
                return {"canceled": False, "why": "nothing is generating"}
            self._canceling = True
            runner = self._runners.get(name)
        if runner is None:
            return {"canceled": False, "why": "nothing is generating"}
        runner.kill()
        return {"canceled": True, "was": name}

    # --- Internals -----------------------------------------------------------
    def _begin(self, name: str) -> None:
        """Mark a runner as the one holding the GPU, and therefore cancellable."""
        with self._lock:
            self._busy = name
            self._canceling = False

    def _end(self) -> None:
        """It is no longer holding the GPU."""
        with self._lock:
            self._busy = None
            self._canceling = False

    def _canceled_instead(self, what: str) -> CanceledError | None:
        """Was this runner's death a cancellation we asked for?

        **A cancel ends the process** (`docs/runner_contract.md` §10), so it
        reaches the caller as the runner having died. Telling the two apart is
        the difference between an error a person should read and one they asked
        for.

        Args:
            what: What was interrupted, for the message.

        Returns:
            The error to raise instead, or None when the runner died on its own
            and the original failure is the true one.
        """
        with self._lock:
            canceled = self._canceling
        if not canceled:
            return None
        self._forget_dead()
        return CanceledError(f"{what} was cancelled")

    def _runner(self, name: str) -> RunnerProcess:
        """Return a runner, creating it if this is the first time.

        Raises:
            RunnerError: If `.env` never declared the name.
        """
        if name not in self.available():
            raise RunnerError(f"unknown runner: {name} (HEARTH_RUNNERS lists {self.available()})")
        with self._lock:
            if name not in self._runners:
                self._runners[name] = RunnerProcess(name, config.runner_spec(name))
            return self._runners[name]

    def _forget_dead(self) -> None:
        """Drop the memory of a loaded model whose process is no longer there.

        **A runner can die mid-generation**, and without this the name stays in
        `_loaded` forever: the next request for that model skips the load it
        needs and fails against a process that is gone, over and over. Call it
        under the lock.
        """
        name = self._loaded
        if name is None:
            return
        runner = self._runners.get(name)
        if runner is None or not runner.is_running():
            self._loaded = None
            self._release_gpu()

    def _claim_gpu(self) -> None:
        """Listen on the lock port for as long as a model is loaded.

        **This is how two hearths find each other**: a second Blender window, or
        a command line run next to a running one, would otherwise load a second
        model into the same card and both would crawl. Disabled when
        `HEARTH_LOCK_PORT` is 0.

        Raises:
            GpuBusyError: If another hearth already holds it.
        """
        port = config.LOCK_PORT
        if port <= 0 or self._gpu_claim is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # **No SO_REUSEADDR.** On Windows it lets a second bind succeed,
            # which is the one thing this must not do.
            sock.bind(("127.0.0.1", port))
            sock.listen(1)
        except OSError as exc:
            sock.close()
            raise GpuBusyError(
                f"another hearth already holds the GPU (port {port} is taken). "
                "Use the one that is running, or stop it first."
            ) from exc
        self._gpu_claim = sock

    def _release_gpu(self) -> None:
        """Stop holding the lock port."""
        sock = self._gpu_claim
        self._gpu_claim = None
        if sock is not None:
            sock.close()

    def shutdown(self) -> None:
        """End every runner."""
        self.unload()
        for runner in self._runners.values():
            runner.stop()
        self._release_gpu()
