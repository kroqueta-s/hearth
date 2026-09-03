# SPDX-License-Identifier: MIT
"""The protocol loop. **Copy this file as it is; there is nothing to change but
`capabilities()`.**

It implements every rule in `docs/runner_contract.md` §1 and §2:

- one JSON object per line, over stdin and stdout,
- **the stdout guard**, installed before anything else,
- exactly one `result` or `error` per request, and no exception escaping,
- `capabilities` answered **without loading the model**.

Run it by hand to see it work::

    echo {"id":1,"method":"capabilities"} | python -m runners.example
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

from . import config
from .rpc import install_stdout_guard

NAME = "sleepy"

_EMIT_LOCK = threading.Lock()


def emit(out: Any, payload: dict[str, Any]) -> None:
    """Write one protocol line.

    **Locked**, because progress can be reported from a watcher thread while the
    main thread is answering.
    """
    with _EMIT_LOCK:
        out.write(json.dumps(payload, ensure_ascii=False) + "\n")
        out.flush()


def m_capabilities(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Say what this runner can do. **Answer without loading the model.**

    Everything a caller needs to decide what to offer is here, as data.
    **Nothing downstream should ever branch on the runner's name.**
    """
    return {
        "name": NAME,
        "version": config.MODEL_VERSION,
        # The version of `docs/runner_contract.md` this was written against.
        # **A caller uses it to explain an absence**, never to refuse a runner.
        "contract": 3,
        "capabilities": {
            "image_to_mesh": True,
            "text_to_mesh": False,
            "multi_image_to_mesh": False,
            "texture": False,
            "texture_mesh": False,
        },
        "params": {
            "seconds": {"type": "float", "default": 1.0, "min": 0.0, "max": 600.0},
            "steps": {"type": "int", "default": 10, "min": 1, "max": 200},
            "seed": {"type": "int", "default": 0, "min": 0},
        },
        # The runner's own process id, so a test can check it is gone.
        "pid": os.getpid(),
        "notes": "A runner that sleeps. It owns no GPU and generates nothing.",
    }


def m_load(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Load the weights."""
    from . import pipeline

    elapsed = pipeline.load(progress)
    return {"loaded": True, "elapsed_sec": round(elapsed, 2)}


def m_unload(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """Release the weights. **Give the VRAM back and mean it.**"""
    from . import pipeline

    freed, used_gb = pipeline.unload()
    return {"unloaded": freed, "vram_used_gb": used_gb}


def m_image_to_mesh(params: dict[str, Any], progress: Any) -> dict[str, Any]:
    """One image to a raw mesh. **Preprocessing is this runner's job.**"""
    from . import pipeline

    return pipeline.image_to_mesh(params, progress)


METHODS = {
    "capabilities": m_capabilities,
    "load": m_load,
    "unload": m_unload,
    "image_to_mesh": m_image_to_mesh,
}



def watch_parent(interval_sec: float = 2.0) -> None:
    """End this process if the caller that started it goes away.

    **This is the orphan case nothing else covers.** hearth stops its runners
    when it shuts down, and a caller that kills hearth kills the whole tree -
    but a hearth that *crashes* does neither. On Windows the child simply
    carries on, holding the entire card, and **nothing anywhere errors**:
    everything afterwards is several times slower for a reason nobody can see.

    Reporting progress or reading stdin is not enough on its own. Both fail once
    the caller's pipes close, which covers most of a run - but not the middle of
    a long kernel, which is exactly when there is most to lose.

    Two things about how this is done, both measured rather than assumed:

    - **The process to watch is the one `HEARTH_PARENT_PID` names**, not this
      process's own parent. A venv's `python.exe` re-executes the base
      interpreter, so the runner's parent is a launcher that outlives hearth by
      design; watching it would never fire. `os.getppid()` is the fallback for
      being run by hand.
    - **`os.getppid()` cannot detect a dead parent on Windows.** A process whose
      parent dies is not reparented there, so the field keeps naming the dead
      one. Holding a handle from the start and waiting on it does work: the
      handle stays valid after the process exits, and a reused id cannot fool
      it.

    `os._exit` rather than a clean exit on purpose: this fires on a thread while
    a generation may be mid-kernel, and unwinding a model from another thread is
    not something to attempt. The weights are in VRAM, not on disk, so there is
    nothing to lose by leaving abruptly.

    Args:
        interval_sec: How often to look, where waiting on a handle is not
            available. Two seconds is far below the cost of noticing an orphan
            any other way.
    """
    named = os.environ.get("HEARTH_PARENT_PID", "").strip()
    watched = int(named) if named.isdigit() else os.getppid()

    def gone() -> None:
        # **Saying so must never stop it leaving.** stderr is a pipe to the
        # process that just died, so writing to it raises - and an exception
        # here would kill this thread and leave the runner holding the card,
        # which is the entire failure being prevented.
        try:
            print(
                f"[{NAME}] the process that started this runner is gone; "
                "exiting so the card is freed",
                file=sys.stderr,
                flush=True,
            )
        except OSError:
            pass
        os._exit(0)

    def watch() -> None:
        if sys.platform == "win32":
            import ctypes  # noqa: PLC0415 - only needed here, and only on Windows
            from ctypes import wintypes  # noqa: PLC0415 - absent on other platforms

            synchronize = 0x00100000
            infinite = 0xFFFFFFFF
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            handle = kernel32.OpenProcess(synchronize, False, watched)
            if handle:
                # Blocks until that process exits, however long that takes.
                kernel32.WaitForSingleObject(handle, infinite)
                gone()
                return
            # No handle: fall through to polling, which is worse but not nothing.
        while True:
            time.sleep(interval_sec)
            if os.getppid() != watched:
                gone()

    threading.Thread(target=watch, name=f"{NAME}-parent-watch", daemon=True).start()


def main() -> int:
    """Handle requests one at a time, in order.

    Returns:
        The exit code. 0 on a clean exit.
    """
    out = install_stdout_guard()
    # **Before anything is loaded.** A runner that has already taken the card is
    # exactly the one worth ending.
    watch_parent()
    print(f"[{NAME}] runner started.", file=sys.stderr)

    for raw in sys.stdin:
        # Windows tools like to put a BOM at the start of a line.
        line = raw.lstrip("﻿").strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = int(request["id"])
            method_name = str(request["method"])
        except (ValueError, KeyError, TypeError) as exc:
            print(f"[{NAME}] skipped an unparsable request: {exc}", file=sys.stderr)
            continue

        if method_name == "shutdown":
            emit(out, {"id": request_id, "event": "result", "result": {"bye": True}})
            break

        method = METHODS.get(method_name)
        if method is None:
            emit(
                out,
                {
                    "id": request_id,
                    "event": "error",
                    "error": {"type": "ValueError", "message": f"unknown method: {method_name}"},
                },
            )
            continue

        def progress(
            stage: str,
            message: str = "",
            _id: int = request_id,
            **extra: Any,
        ) -> None:
            # `extra` carries `step` and, when the length is known, `total`.
            # **Nothing estimated ever goes in here** (contract §8).
            emit(
                out,
                {"id": _id, "event": "progress", "stage": stage, "message": message, **extra},
            )

        try:
            emit(
                out,
                {
                    "id": request_id,
                    "event": "result",
                    "result": method(dict(request.get("params") or {}), progress),
                },
            )
        except Exception as exc:  # noqa: BLE001 - always answer, whatever happens
            # **Never let this escape.** A runner that dies leaves its caller
            # waiting; one that answers with an error lets the caller say why.
            import traceback

            traceback.print_exc()
            emit(
                out,
                {
                    "id": request_id,
                    "event": "error",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
            )

    print(f"[{NAME}] runner exiting.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
