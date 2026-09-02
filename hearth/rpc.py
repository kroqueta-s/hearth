# SPDX-License-Identifier: MIT
"""The stdio JSON-RPC frame: one message per line.

**No HTTP server and no port.** The caller starts hearth as a child process and
talks to it over stdin and stdout. **hearth talks to its runners the same way**
(`docs/runner_contract.md` §1).

## The shape of a message

A request (caller to hearth)::

    {"id": 1, "method": "image_to_mesh", "params": {...}}

A reply. **Zero or more `progress` for one request, then exactly one `result`
or `error`**::

    {"id": 1, "event": "progress", "stage": "shape", "message": "..."}
    {"id": 1, "event": "result", "result": {...}}
    {"id": 1, "event": "error", "error": {"type": "...", "message": "..."}}

## Promises that must not be broken

1. **Nothing else may write to the protocol's stdout.** Model dependencies print
   to stdout as a matter of course - version banners, notices, progress an
   author wanted a human to see - and a single such line breaks the protocol. `install_stdout_guard()` duplicates the
   real stdout, hides it, and points `sys.stdout` at stderr. **Call it first.**
2. **stderr is never parsed.** It is for people and for logs, and it is what
   gets read back after a crash.
3. **Requests are handled strictly one at a time.** There is one GPU.
4. **There is no cancellation.** A running job runs to the end.
5. **No bytes on the wire.** Images and meshes are passed as **absolute paths**.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, TextIO


@dataclass(frozen=True)
class Request:
    """One request.

    id: Identifies the request. Replies carry it back unchanged.
    method: The method to call.
    params: Its arguments.
    """

    id: int
    method: str
    params: dict[str, Any]


def install_stdout_guard() -> TextIO:
    """Duplicate the real stdout, hide it, and point `sys.stdout` at stderr.

    **Call this first, before anything else.** It is the only thing standing
    between a vendor `print` and a broken protocol, and model dependencies do
    print to stdout.

    Returns:
        The protocol's stream. **Nothing else may write to stdout.**
    """
    fd = os.dup(sys.stdout.fileno())
    protocol = os.fdopen(fd, "w", encoding="utf-8", newline="\n", buffering=1)
    sys.stdout = sys.stderr
    return protocol


class Responder:
    """Writes the replies to one request.

    **Every line is flushed.** Progress that has not arrived is progress the
    caller cannot distinguish from a hang.
    """

    def __init__(self, out: TextIO, request_id: int) -> None:
        self._out = out
        self._id = request_id
        self._closed = False

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("this request is already answered (result/error happens once)")
        line = json.dumps({"id": self._id, **payload}, ensure_ascii=False)
        self._out.write(line + "\n")
        self._out.flush()

    def progress(
        self,
        stage: str,
        message: str = "",
        *,
        step: int | None = None,
        total: int | None = None,
    ) -> None:
        """Send one progress line. Call it as often as you like.

        Args:
            stage: The name of the stage.
            message: Something a person can read.
            step: The **counted** step, from one. Only when it is known.
            total: How many steps there are. **Only when the length is known.**
                Without it the receiver must not show a percentage (contract §8).
        """
        payload: dict[str, Any] = {"event": "progress", "stage": stage, "message": message}
        if step is not None:
            payload["step"] = step
            if total is not None:
                payload["total"] = total
        self._emit(payload)

    def result(self, result: dict[str, Any]) -> None:
        """Finish successfully. The responder cannot be used again."""
        self._emit({"event": "result", "result": result})
        self._closed = True

    def error(self, exc: BaseException) -> None:
        """Finish with a failure. The responder cannot be used again."""
        self._emit({"event": "error", "error": {"type": type(exc).__name__, "message": str(exc)}})
        self._closed = True


def read_requests(stream: TextIO) -> Iterator[Request]:
    """Read requests from a stream, one line at a time.

    A broken or malformed line is skipped rather than raised: there is no id, so
    there is nowhere to send an answer. It is written to stderr for diagnosis
    instead of vanishing.

    Args:
        stream: Where to read from (`sys.stdin` in the worker).

    Yields:
        Every request that parsed.
    """
    for raw in stream:
        # Windows tools like to put a BOM at the start of a line. Parsing fails
        # unless it is stripped.
        line = raw.lstrip("﻿").strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            request = Request(
                id=int(obj["id"]), method=str(obj["method"]), params=dict(obj.get("params") or {})
            )
        except (ValueError, KeyError, TypeError) as exc:
            print(f"[rpc] skipped an unparsable request: {exc}: {line[:200]}", file=sys.stderr)
            continue
        yield request
