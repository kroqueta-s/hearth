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

import contextlib
import socket
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from . import config, vram
from .comfy import ComfyUIClient
from .comfy_process import COMFY
from .runner_client import STDERR_LINES, Relay, RunnerError, RunnerProcess
from .vram import VramOverError

# What `busy` is called while the work is in another application's process.
EXTERNAL_PREFIX = "image:"


class GpuBusyError(RuntimeError):
    """Another process already holds the GPU, so no runner can be loaded."""


class CanceledError(RuntimeError):
    """The request was cancelled by the caller (`docs/runner_contract.md` §9)."""


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


class GpuBusyWatch:
    """Whether another application holds the GPU, looked at on a timer.

    **`assert_gpu_free` is not free.** It connects to `HEARTH_GPU_BUSY_PORT`,
    and on the machine hearth was written for a closed port on 127.0.0.1 is
    dropped rather than refused (measured 2026-09-06), so learning "nobody is
    there" costs the whole connect timeout. `status` asks that question every
    time it is called, and `status` is what an interface polls while a
    generation runs - it was answering in **500 ms**, on the same thread that
    reads `cancel`.

    So the probe happens here, on a timer, and `status` reports the last look.
    **`load` still asks for itself**: it is about to spend a minute, so half a
    second buys a fresh answer, and a stale "free" would put two models on one
    card.
    """

    def __init__(self) -> None:
        self._busy = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None or config.GPU_BUSY_PORT <= 0:
            return
        self._thread = threading.Thread(target=self._run, name="hearth-gpu-busy", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                assert_gpu_free()
            except GpuBusyError:
                self._busy = True
            else:
                self._busy = False
            self._stop.wait(GPU_BUSY_POLL_SEC)

    def busy(self) -> bool:
        """The last look. **False when the check is disabled**, as it always was."""
        return self._busy


# How often the "somebody else has the GPU" port is looked at. **Not measured**:
# another application taking the card is not something that happens between two
# frames of an interface, and each look costs a connect timeout.
GPU_BUSY_POLL_SEC = 5.0

GPU_BUSY_WATCH = GpuBusyWatch()


def _contract_shape(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Bring an older runner's result up to contract §5, and no further.

    Exactly one thing is repaired: a runner written against the earlier wording
    reports `params` where the contract says `params_used`. That is a rename, so
    promoting it invents nothing.

    **The axes are not repaired.** A runner that does not say which way is up
    has not told anyone, and filling in `"z"` would turn "unknown" into
    "known and possibly wrong" - which is worse, because the mistake it hides is
    invisible: a mesh imported on the wrong axis renders perfectly correctly.
    The absence travels downstream, and `forge` says so.
    """
    if "params_used" not in result and isinstance(result.get("params"), dict):
        print(
            f"[hearth] {name}: reports `params` where the contract says `params_used` "
            "(runner_contract.md §5)",
            file=sys.stderr,
        )
        result = {**result, "params_used": result["params"]}
    # **Only a result that carries a mesh owes an axis.** §5 is about the shape
    # of a mesh result, and a method that produces none - `segment_mesh` answers
    # with labels - has nothing to be oriented. Saying so anyway made a correct
    # runner look defective once per call.
    if "mesh_path" in result and "up_axis" not in result:
        print(
            f"[hearth] {name}: reports no `up_axis` (runner_contract.md §5). "
            "Passing it on as unknown rather than guessing",
            file=sys.stderr,
        )
    return result


def _vram_run_up() -> list[str]:
    """The card's memory over the last minute, one line a reading, for a death record.

    **What the whole card held matters as much as what the runner did.** A
    driver that fails a submit for want of GPU memory while the runner's own
    total is far below the card is saying something about what else was on it,
    or about how the memory was laid out - and only the counters from outside
    see the first.
    """
    history = vram.SAMPLER.history()
    if not history:
        return ["vram: no readings (the counters are unavailable here)"]
    now = time.time()
    lines = ["vram (dedicated used / total, shared used; then each watched process):"]
    for sample in history:
        watched = ", ".join(
            f"pid {pid} {entry['dedicated_gb']:.2f}/{entry['shared_gb']:.2f}"
            for pid, entry in sorted(sample.by_pid.items())
        )
        lines.append(
            f"  {sample.sampled_at - now:+6.1f}s  {sample.dedicated_used_gb:.2f}/"
            f"{sample.dedicated_total_gb:.2f} GB, shared {sample.shared_used_gb:.2f} GB"
            + (f"  [{watched}]" if watched else "")
        )
    return lines


# How long a cancel waits on ComfyUI. **Short on purpose**: cancelling is
# interactive, it is answered on the thread that reads stdin, and a caller that
# has just pressed cancel usually presses stop next. The default of thirty
# seconds turned that into a minute of silence when ComfyUI was wedged.
CANCEL_TIMEOUT_SEC = 5.0


class _SpillWatch:
    """Ends a generation that has spilled out of the card into system memory.

    **Nothing raises when a GPU runs out on this machine.** The driver falls back
    to shared memory and the work carries on several times slower, which is the
    kind of failure nobody attributes to its cause. This is the outside view of
    it: the runner's process family is measured against
    `HEARTH_VRAM_SHARED_ABORT_GB`, and going over ends the process - the only
    thing that stops a torch loop (`docs/runner_contract.md` §9).

    The kill surfaces as `RunnerError` in `generate`, exactly as a cancel does,
    and `spilled` is what tells the two apart.
    """

    def __init__(self, runner: RunnerProcess, name: str) -> None:
        self._runner = runner
        self._name = name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.spilled: VramOverError | None = None

    def start(self) -> None:
        pid = self._runner.pid()
        if pid <= 0 or config.VRAM_SHARED_ABORT_GB <= 0:
            return
        vram.SAMPLER.watch(pid, self._name)
        self._pid = pid
        self._thread = threading.Thread(target=self._run, name="hearth-spill", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(config.VRAM_SAMPLE_SEC):
            over = vram.SAMPLER.spilled(self._pid)
            if over is None:
                continue
            shared_gb, dedicated_gb = over
            self.spilled = VramOverError(
                f"{self._name} (pid {self._pid}) has spilled {shared_gb:.1f} GB into shared "
                f"system memory on top of {dedicated_gb:.1f} GB of dedicated VRAM. It would "
                "finish, several times slower. Ask for less, or free the card first.",
                shared_gb=shared_gb,
                dedicated_gb=dedicated_gb,
                pid=self._pid,
            )
            print(f"[hearth] {self.spilled}", file=sys.stderr)
            self._runner.kill()
            return

    def stop(self) -> None:
        self._stop.set()
        pid = getattr(self, "_pid", 0)
        if pid:
            vram.SAMPLER.forget(pid)


class Manager:
    """Supervises the runners. **All the state lives here.**"""

    def __init__(self) -> None:
        self._runners: dict[str, RunnerProcess] = {}
        self._capabilities: dict[str, dict[str, Any]] = {}
        self._loaded: str | None = None
        self._busy: str | None = None
        # The ComfyUI prompt an image route is waiting on, so that cancelling
        # takes out **that one** and not whatever else is running there.
        self._prompt_id = ""
        self._canceling = False
        # **Set once, and never unset.** While it is true nothing new may be
        # loaded or generated: a request already in the GPU queue would start a
        # runner that hearth is about to stop caring about, and on Windows that
        # runner outlives it holding the card.
        self._shutting_down = False
        self._lock = threading.RLock()
        self._gpu_claim: socket.socket | None = None

    @property
    def shutting_down(self) -> bool:
        """Whether hearth is on its way out."""
        with self._lock:
            return self._shutting_down

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
        self._refuse_if_shutting_down()
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
        self._refuse_if_shutting_down()
        caps = self.capabilities(name).get("capabilities", {})
        if not caps.get(method, False):
            raise RunnerError(f"{name} does not support {method} (check its capabilities)")
        attempts = 0
        kept: list[str] = []
        while True:
            try:
                result = self._generate_once(name, method, params, relay, attempts)
                if kept:
                    result["runner_logs"] = kept
                return result
            except RunnerError as exc:
                attempts += 1
                died = self._died_unasked(name)
                if died:
                    record = self._keep_stderr(name, method, params, attempts)
                    if record:
                        kept.append(record)
                if attempts > max(config.GENERATE_RETRIES, 0) or not died:
                    if kept:
                        raise RunnerError(
                            f"{exc}\nstderr of each death: {', '.join(kept)}"
                        ) from exc
                    raise
                if relay is not None:
                    relay(
                        "retry",
                        f"{name} died without answering; loading it again and asking once more "
                        f"(attempt {attempts + 1})",
                    )

    def _generate_once(
        self,
        name: str,
        method: str,
        params: dict[str, Any],
        relay: Relay | None,
        attempt: int,
    ) -> dict[str, Any]:
        """One attempt at a generating call, loading the runner if it is not up."""
        self._refuse_if_shutting_down()
        with self._lock:
            self._forget_dead()
            needs_load = self._loaded != name
        if needs_load:
            self.load(name, relay=relay)
        self._begin(name)
        runner = self._runners[name]
        # **The same ruler as ComfyUI's.** A runner that holds torch watches its
        # own VRAM (`apply_vram_limit`), but not every runner holds torch -
        # partfield does not - and a spill there was invisible. Watching from
        # outside covers both, and covers a runner whose own limit is wrong.
        watch = _SpillWatch(runner, name)
        watch.start()
        try:
            result = runner.call(method, params, relay=relay)
        except RunnerError:
            over = watch.spilled
            if over is not None:
                raise over from None
            canceled = self._canceled_instead(f"{method} on {name}")
            if canceled is not None:
                raise canceled from None
            raise
        finally:
            watch.stop()
            self._end()
        shaped = {"model": name, **_contract_shape(name, result)}
        if attempt:
            # **Say that it took more than one go.** A caller comparing times
            # against the table would otherwise see a load it cannot explain.
            shaped["attempts"] = attempt + 1
        return shaped

    def _keep_stderr(self, name: str, method: str, params: dict[str, Any], attempt: int) -> str:
        """Write a dead runner's stderr beside the output, and say where.

        **In memory it lasted until the next start**, and only its last twenty
        lines reached the caller - so a death that a retry covered left nothing
        anyone could read, and a fault that happens one run in several could
        not be counted. The request's `out_dir` is where the rest of that run
        already is, which keeps a death with the work it interrupted.

        Returns:
            The file written, or "" when there was nowhere to write it. **Failing
            to keep a record never fails the request**; it is said on stderr.
        """
        runner = self._runners.get(name)
        out = str(params.get("out_dir") or "").strip()
        if runner is None or not out:
            return ""
        code = runner.last_exit_code
        named = "unknown" if code is None else f"{code} (0x{code & 0xFFFFFFFF:08X})"
        lines = runner.stderr_all()
        header = [
            f"runner: {name}",
            f"method: {method}",
            f"attempt: {attempt}",
            f"exit code: {named}",
            f"recorded: {datetime.now().isoformat(timespec='seconds')}",
            f"stderr lines: {len(lines)} (only the last {STDERR_LINES} are ever kept)",
            *_vram_run_up(),
            "",
        ]
        target = Path(out) / f"runner_stderr_{attempt}.txt"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"[hearth] could not keep {name}'s stderr at {target}: {exc}", file=sys.stderr)
            return ""
        return str(target)

    def _died_unasked(self, name: str) -> bool:
        """Did the runner's process go away on its own, rather than being ended?

        **A cancel and a shutdown both end the process** (contract §9), and
        neither is something to try again. Nor is an error the runner answered
        with: it is still running, and the answer will be the same. What is left
        is the driver taking the process out from under us, which the next
        attempt may well survive.
        """
        with self._lock:
            if self._shutting_down or self._canceling:
                return False
        runner = self._runners.get(name)
        return runner is not None and not runner.is_running()

    def cancel(self) -> dict[str, Any]:
        """End whatever is generating right now, by ending its process.

        **There is no gentler way** (`docs/runner_contract.md` §9). The price is
        that the weights go with it and the next generation pays a full load.

        Returns:
            Whether anything was cancelled, and what it was.
        """
        with self._lock:
            name = self._busy
            prompt_id = self._prompt_id
            if name is None:
                return {"canceled": False, "why": "nothing is generating"}
            self._canceling = True
            runner = self._runners.get(name)

        if name.startswith(EXTERNAL_PREFIX):
            # **Somebody else's process.** Only this prompt is taken out of
            # ComfyUI's queue; nothing is killed on the user's behalf (§6).
            # **A short timeout, because this is somebody pressing a button.**
            # The default is thirty seconds, and `cancel` is answered on the
            # thread that reads stdin - so an unresponsive ComfyUI would hold up
            # every control method behind it, including the `shutdown` a caller
            # sends next.
            dropped = (
                ComfyUIClient(timeout_sec=CANCEL_TIMEOUT_SEC).cancel_prompt(prompt_id)
                if prompt_id
                else False
            )
            # **The request ends either way.** `_canceling` is set above, and the
            # route asks about it between polls of ComfyUI's history, so the wait
            # stops within a poll whatever the queue says. What differs is
            # whether the work itself was still in there to be taken out.
            if not prompt_id:
                why = "the workflow had not reached ComfyUI yet; it is dropped as it arrives"
            elif not dropped:
                why = "ComfyUI was no longer holding this prompt, so only the wait was stopped"
            else:
                why = ""
            return {
                "canceled": True,
                "was": name,
                # The image model's own load is not paid again. **The 3D model's
                # is**: it was unloaded to make room before the image started.
                "image_model_reload": False,
                "dropped_from_queue": bool(dropped),
                **({"why": why} if why else {}),
            }

        if runner is None:
            return {"canceled": False, "why": "nothing is generating"}
        runner.kill()
        return {"canceled": True, "was": name}

    # --- Internals -----------------------------------------------------------
    def note_external_prompt(self, prompt_id: str) -> bool:
        """Record which prompt the running image route submitted.

        **The id does not exist when the work starts.** `begin_external` runs
        before the workflow is sent, because a caller asking `status` in that
        second deserves to be told an image is being made. So the id arrives
        afterwards, and this is the only place it may be written - `cancel` may
        have been asked for in between, and starting over here would forget it.

        Returns:
            Whether a cancel is already pending. **The caller must then drop the
            prompt it has just submitted**, because nothing else is going to:
            `cancel` has already looked and found no id to act on.
        """
        with self._lock:
            self._prompt_id = prompt_id
            return self._canceling

    def begin_external(self, label: str, prompt_id: str = "") -> None:
        """Mark work that is running somewhere else as the busy one.

        ComfyUI is another application: hearth does not own its process and
        cannot kill it. But the person waiting has no way to know that, and
        `cancel` answering "nothing is generating" during an eight-minute image
        is the same lie either way. So an image route marks itself busy, and
        `cancel` asks ComfyUI to drop **that prompt** (§5).
        """
        with self._lock:
            self._busy = label
            self._prompt_id = prompt_id
            # **A shutdown already under way is not cleared by work starting.**
            # `_serve_gpu` refuses what is still queued, but a request that got
            # past that check reaches here a moment later, and starting fresh
            # would forget the shutdown - leaving a prompt running in ComfyUI
            # that nobody is left to collect. Cancelling is already true of it.
            self._canceling = self._shutting_down

    def is_canceling(self) -> bool:
        """Whether a cancel has been asked for and not yet taken effect.

        **Asked from inside the wait on ComfyUI.** An image route polls this
        between polls of the queue, so a cancel ends the wait in seconds rather
        than after `COMFY_TIMEOUT_SEC`.
        """
        with self._lock:
            return self._canceling

    def end_external(self) -> None:
        """The work somewhere else has finished."""
        with self._lock:
            self._busy = None
            self._prompt_id = ""
            self._canceling = False

    def _begin(self, name: str) -> None:
        """Mark a runner as the one holding the GPU, and therefore cancellable."""
        with self._lock:
            self._busy = name
            self._prompt_id = ""
            self._canceling = False

    def _end(self) -> None:
        """It is no longer holding the GPU."""
        with self._lock:
            self._busy = None
            self._prompt_id = ""
            self._canceling = False

    def _canceled_instead(self, what: str) -> CanceledError | None:
        """Was this runner's death a cancellation we asked for?

        **A cancel ends the process** (`docs/runner_contract.md` §9), so it
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

    def _refuse_if_shutting_down(self) -> None:
        """Raise if hearth is on its way out.

        Raises:
            RunnerError: Always, when shutting down. **A request that arrives
                too late is answered**, rather than starting a process nobody
                will be left to stop.
        """
        if self.shutting_down:
            raise RunnerError("hearth is shutting down")

    def shutdown(self) -> None:
        """End every runner. **Nothing may be left holding the card.**

        The order matters, and it is the whole of the fix for the failure this
        method used to cause:

        1. **The flag first.** A queued `load` or generation would otherwise
           start a runner while this is running, and that runner is not in the
           list below.
        2. **A running generation is killed, not asked.** `unload` would wait on
           the call lock that generation holds, so a shutdown during one looked
           like a hang - and the caller's answer to a hang is to kill hearth,
           which on Windows leaves the runner alive with the VRAM.
        3. **An image is stopped too, in the only way that is available.**
           ComfyUI is another application and nothing of its is killed, but the
           prompt is taken out of its queue and the waiting side is told to
           stop. Leaving it would mean shutting down while still producing an
           image nobody will collect.
        4. Then the ordinary unload, and every remaining process stopped.
        """
        with self._lock:
            self._shutting_down = True
            busy = self._busy
            prompt_id = self._prompt_id
            self._canceling = True
        if busy is not None:
            if busy.startswith(EXTERNAL_PREFIX):
                if prompt_id:
                    with contextlib.suppress(Exception):
                        # **A shutdown must not fail over an unreachable
                        # ComfyUI.** Everything below still has to happen.
                        ComfyUIClient(timeout_sec=CANCEL_TIMEOUT_SEC).cancel_prompt(prompt_id)
            else:
                runner = self._runners.get(busy)
                if runner is not None:
                    runner.kill()
        self.unload()
        for runner in list(self._runners.values()):
            runner.stop()
        self._release_gpu()
        GPU_BUSY_WATCH.stop()
        # **Last, and only if hearth started it.** An adopted ComfyUI belongs to
        # whoever launched it and holding a loaded FLUX across Blender sessions
        # is the point of adopting one at all.
        COMFY.shutdown()
        vram.SAMPLER.stop()
