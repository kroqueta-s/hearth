# SPDX-License-Identifier: MIT
"""hearth's entry point. **It talks over stdin and stdout, and serves no port.**

Start it as a child process::

    & $env:HEARTH_PYTHON -m hearth

**Each model's virtual environment belongs to its runner.** hearth itself never
imports torch: images come from ComfyUI over HTTP, meshes from a runner over
stdio.
"""

from __future__ import annotations

import sys

from .rpc import Responder, install_stdout_guard, read_requests
from .worker import MANAGER, handle


def main() -> int:
    """Handle requests one at a time, in order, until stdin closes.

    Returns:
        The exit code. 0 on a clean exit.
    """
    protocol_out = install_stdout_guard()
    print("[hearth] started, waiting for requests on stdin.", file=sys.stderr)

    try:
        for request in read_requests(sys.stdin):
            if request.method == "shutdown":
                Responder(protocol_out, request.id).result({"bye": True})
                break
            handle(request, Responder(protocol_out, request.id))
    finally:
        # **Leave no runner behind.** One that outlives hearth keeps the VRAM.
        MANAGER.shutdown()

    print("[hearth] exiting.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
