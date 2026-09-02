# SPDX-License-Identifier: MIT
"""Starts runners, switches between them, and holds the state.

**Only one model is loaded at a time.** There is one GPU, so switching means
**unloading before loading**; overlapping the two goes past the VRAM the card
actually has.

hearth does not interpret a runner's `params`. **Only the runner knows what its
own values mean**, so validation is left to it and the arguments pass straight
through (`docs/runner_contract.md` §3).
"""

from __future__ import annotations

import socket
import time
from typing import Any

from . import config
from .runner_client import Relay, RunnerError, RunnerProcess


class GpuBusyError(RuntimeError):
    """Another process already holds the GPU, so no runner can be loaded."""


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

    # --- Inventory and capabilities -----------------------------------------
    def available(self) -> list[str]:
        """The runners declared in `.env`."""
        return config.runner_names()

    def capabilities(self, name: str) -> dict[str, Any]:
        """Return a runner's capabilities (**without loading its weights**).

        The answer is remembered. Answering `capabilities` without loading a
        model is part of the contract, so asking is cheap.

        Args:
            name: The runner's name.

        Returns:
            The shape described in `docs/runner_contract.md` §3.

        Raises:
            RunnerError: If the runner is unknown or would not start.
        """
        if name in self._capabilities:
            return self._capabilities[name]
        runner = self._runner(name)
        started_here = not runner.is_running()
        runner.start()
        try:
            caps = runner.call("capabilities")
        finally:
            # If it was only started to ask, put it back down.
            if started_here and self._loaded != name:
                runner.stop()
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
        return self._loaded

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
            GpuBusyError: If another process holds the GPU.
            RunnerError: If the runner is unknown or loading failed.
        """
        assert_gpu_free()
        if self._loaded == name:
            return {"loaded": name, "elapsed_sec": 0.0, "already": True}
        if self._loaded is not None:
            if relay is not None:
                relay("unload", f"unloading {self._loaded} to free the VRAM")
            self.unload(relay=relay)

        runner = self._runner(name)
        spawn_started = time.perf_counter()
        runner.start()
        spawn_sec = time.perf_counter() - spawn_started
        result = runner.call("load", relay=relay)
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
        name = self._loaded
        self._loaded = None
        if name is None:
            return {"unloaded": False}
        runner = self._runners.get(name)
        if runner is None:
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
            RunnerError: If the runner does not support the method.
        """
        caps = self.capabilities(name).get("capabilities", {})
        if not caps.get(method, False):
            raise RunnerError(f"{name} does not support {method} (check its capabilities)")
        if self._loaded != name:
            self.load(name, relay=relay)
        result = self._runners[name].call(method, params, relay=relay)
        return {"model": name, **result}

    # --- Internals -----------------------------------------------------------
    def _runner(self, name: str) -> RunnerProcess:
        """Return a runner, creating it if this is the first time.

        Raises:
            RunnerError: If `.env` never declared the name.
        """
        if name not in self.available():
            raise RunnerError(
                f"unknown runner: {name} (HEARTH_RUNNERS lists {self.available()})"
            )
        if name not in self._runners:
            self._runners[name] = RunnerProcess(name, config.runner_spec(name))
        return self._runners[name]

    def shutdown(self) -> None:
        """End every runner."""
        self.unload()
        for runner in self._runners.values():
            runner.stop()
