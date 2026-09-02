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
import sys
import threading
from typing import Any

from . import config
from .rpc import install_stdout_guard

NAME = "example"

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
        "contract": 2,
        "capabilities": {
            "image_to_mesh": True,
            "text_to_mesh": False,
            "multi_image_to_mesh": False,
            "texture": False,
            "texture_mesh": False,
        },
        "params": {
            "steps": {"type": "int", "default": config.STEPS, "min": 1, "max": 200},
            "seed": {"type": "int", "default": 0, "min": 0},
        },
        "notes": "Replace this with anything a caller should know that the fields cannot say.",
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


def main() -> int:
    """Handle requests one at a time, in order.

    Returns:
        The exit code. 0 on a clean exit.
    """
    out = install_stdout_guard()
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
