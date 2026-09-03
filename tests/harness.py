# SPDX-License-Identifier: MIT
"""Hold one hearth session open, drive it, and **watch that it is making progress**.

**This exists to stop waiting for things to finish.** Generations that run for
ten minutes in silence get waited out to the end, and the waiting teaches you
nothing: whatever went wrong is just as visible at minute one as at minute ten.

Two things are watched:

1. **Stalling** (`stall_sec`). A runner emits a `heartbeat` every ten seconds,
   so a gap in them means it is not progressing. **This is a better signal than
   a time budget**, because generation time varies by several times for
   identical settings and a budget alone gives false alarms.
2. **A budget** (`budget_sec`), where a real measurement exists to base one on.
   Where none does, it stays `None` and the stall detector does the work.

**When it gives up, it collects a diagnosis before killing anything**: which
stage, how long ago, and what the last heartbeat said. Without that, stopping
early tells you no more than waiting did.

A `vram_over` report **ends the call immediately**. Once a runner has gone past
its VRAM cap the outcome is decided - it will only get slower - so there is
nothing to wait for.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# **One input, used for every model.** Comparing models on different inputs says
# nothing about the models. Point HEARTH_TEST_IMAGE at whatever specimen you use.
IMAGE = Path(os.environ.get("HEARTH_TEST_IMAGE", str(REPO_ROOT / "assets" / "sample.png")))

# **A budget only goes in here when a measurement backs it.** A number with
# nothing behind it hardens into "that is how long it takes" and stops anyone
# asking why.
#
# Anything not listed has no budget and relies on stall detection alone, which
# is the safer default. Override one with HEARTH_BUDGET_<NAME>_SEC.
BUDGET_SEC: dict[str, float | None] = {}
for _name in os.environ:
    if _name.startswith("HEARTH_BUDGET_") and _name.endswith("_SEC"):
        BUDGET_SEC[_name[len("HEARTH_BUDGET_") : -len("_SEC")].lower()] = float(os.environ[_name])

# How long without any progress counts as stalled. Runners beat every ten
# seconds by default, so this leaves plenty of room for a slow stage.
STALL_SEC = float(os.environ.get("HEARTH_STALL_SEC", "60"))


class WatchdogAbort(RuntimeError):
    """Raised when the watchdog gives up. **It carries the diagnosis.**"""

    def __init__(self, reason: str, diagnosis: dict[str, Any]) -> None:
        super().__init__(reason)
        self.diagnosis = diagnosis


@dataclass
class Trace:
    """What happened during one call."""

    stages: list[tuple[float, str, str]] = field(default_factory=list)
    last_heartbeat: tuple[float, str] | None = None
    vram_over: bool = False

    def summary(self) -> str:
        """One line: how long each stage took."""
        out = []
        for i, (at, stage, _) in enumerate(self.stages):
            nxt = self.stages[i + 1][0] if i + 1 < len(self.stages) else None
            span = f"{nxt - at:.1f}s" if nxt is not None else "…"
            out.append(f"{stage} {span}")
        return " / ".join(out)


class HearthSession:
    """Start one hearth process and send it many requests.

    `tools/rpc_call.py` starts a fresh process per request, so it cannot show
    anything about switching between models within a session.
    """

    def __init__(self, python: str | None = None) -> None:
        self._python = python or sys.executable
        self._proc: subprocess.Popen[str] | None = None
        self._next_id = 1

    def __enter__(self) -> HearthSession:
        self._proc = subprocess.Popen(
            [self._python, "-m", "hearth"],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # **Discarded.** Left unread, the pipe fills.
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=self._env(),
        )
        return self

    @staticmethod
    def _env() -> dict[str, str]:
        """The environment a test's hearth runs in.

        **`HEARTH_LOCK_PORT=0` is not optional.** The default is a real port, and
        the operator's Blender holds it whenever their hearth is up: a test would
        then fail with `GpuBusyError` and say nothing at all about what it was
        checking. `load_dotenv` never overwrites what is already set, so `.env`
        is left alone.
        """
        return {
            **os.environ,
            "HEARTH_LOCK_PORT": "0",
            "HEARTH_GPU_BUSY_PORT": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
        }

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Shut down cleanly, or end the process if it will not."""
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"id": 0, "method": "shutdown"}) + "\n")
                proc.stdin.flush()
            proc.wait(timeout=20)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            proc.kill()

    def kill(self) -> None:
        """**Kill it without waiting.** Used when giving up."""
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            proc.kill()

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        budget_sec: float | None = None,
        stall_sec: float = STALL_SEC,
        trace: bool = True,
    ) -> tuple[Any, Trace]:
        """Send one request and wait for it **under the watchdog**.

        Args:
            method: The method to call.
            params: Its arguments.
            budget_sec: Give up past this. None relies on stall detection alone.
            stall_sec: Give up after this long with no progress at all.
            trace: Whether to print progress as it arrives.

        Returns:
            `(result, Trace)`.

        Raises:
            WatchdogAbort: On a stall, an overrun, or a VRAM overflow.
            RuntimeError: If the runner answered with an error.
        """
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        request_id = self._next_id
        self._next_id += 1
        payload = {"id": request_id, "method": method, "params": params or {}}
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        started = time.monotonic()
        last_event = started
        record = Trace()

        while True:
            line = self._readline_with_watchdog(started, last_event, budget_sec, stall_sec, record)
            now = time.monotonic()
            message = json.loads(line)
            if message.get("id") != request_id:
                continue
            last_event = now
            event = message.get("event")

            if event == "progress":
                stage = str(message.get("stage", ""))
                text = str(message.get("message", ""))
                if stage == "heartbeat":
                    record.last_heartbeat = (now - started, text)
                else:
                    record.stages.append((now - started, stage, text))
                if trace:
                    print(f"      [+{now - started:6.1f}s] {stage:11s} {text}", flush=True)
                if stage == "vram_over":
                    record.vram_over = True
                    raise WatchdogAbort(
                        "went past the VRAM cap; waiting can only make it slower",
                        self._diagnose(method, params, started, record, "vram_over"),
                    )
                continue

            if event == "error":
                raise RuntimeError(f"{method}: {message['error']}")
            return message["result"], record

    def _readline_with_watchdog(
        self,
        started: float,
        last_event: float,
        budget_sec: float | None,
        stall_sec: float,
        record: Trace,
    ) -> str:
        """Read one line, **giving up if it stalls or overruns while waiting**.

        `readline` does not return until a line arrives, so the checks happen
        around it rather than during. That is enough because runners beat every
        ten seconds and `stall_sec` is well above that interval.
        """
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        now = time.monotonic()
        if not line:
            raise WatchdogAbort(
                "hearth exited without answering",
                self._diagnose("", None, started, record, "eof"),
            )
        if budget_sec is not None and now - started > budget_sec:
            raise WatchdogAbort(
                f"went past the {budget_sec:.0f}s budget (took {now - started:.0f}s)",
                self._diagnose("", None, started, record, "budget"),
            )
        if now - last_event > stall_sec:
            raise WatchdogAbort(
                f"no progress for {stall_sec:.0f}s (the heartbeat stopped too)",
                self._diagnose("", None, started, record, "stall"),
            )
        return line

    def _diagnose(
        self,
        method: str,
        params: dict[str, Any] | None,
        started: float,
        record: Trace,
        kind: str,
    ) -> dict[str, Any]:
        """The diagnosis, collected **before anything is killed**."""
        return {
            "kind": kind,
            "method": method,
            "params": params,
            "elapsed_sec": round(time.monotonic() - started, 1),
            "stages": [(round(at, 1), stage, text) for at, stage, text in record.stages],
            "last_heartbeat": record.last_heartbeat,
            "vram_over": record.vram_over,
        }


def print_diagnosis(exc: WatchdogAbort) -> None:
    """Print a watchdog diagnosis in a readable form."""
    d = exc.diagnosis
    print(f"\n  ** gave up: {exc} **", flush=True)
    print(f"     {d['kind']} after {d['elapsed_sec']}s", flush=True)
    for at, stage, text in d["stages"]:
        print(f"     [+{at:6.1f}s] {stage:11s} {text}", flush=True)
    if d["last_heartbeat"] is not None:
        at, text = d["last_heartbeat"]
        print(f"     last heartbeat [+{at:.1f}s] {text}", flush=True)
